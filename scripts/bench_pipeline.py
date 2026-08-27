"""Бенчмарк голосового пайплайна Гоги: STT / LLM / TTS по отдельности и оценка
полного цикла реплики (E2E).

Зачем: ROADMAP объявляет STT на CPU главным блокером live talk, но замеры
`small`/`base` и LLM time-to-first-token не сняты. Пока нет воспроизводимого
измерителя, прогресс по «STT < 1 c» оценивается на глаз. Этот скрипт —
измеритель: гоняет одни и те же входные данные через каждую ступень, печатает
таблицу и складывает JSON в benchmarks/, чтобы версии проекта можно было
сравнивать между собой и на другом железе (NVIDIA).

Запуск:
    gogi-bench                      # всё: STT (base/small/medium) + LLM + TTS
    gogi-bench --stt-only           # только распознавание
    gogi-bench --no-llm --no-tts    # то же самое
    gogi-bench --stt-models small,medium --runs 5
    gogi-bench --nfe 8,16           # какие nfe_step мерить у TTS

Ничего не мокается — нужны живые Ollama и GPU. Секция, для которой окружение
недоступно (Ollama не запущен, модель F5 не на месте), пропускается с пометкой,
остальные отрабатывают.
"""

import argparse
import json
import statistics
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Тестовое аудио: та же запись 9.61 c, на которой снят базовый замер в ROADMAP
# (whisper medium int8 CPU -> 5.86 c, RTF 0.61). Транскрипт известен -> считаем CER.
_STT_CLIP = _ROOT / "voices" / "gogi_voice_16k.wav"
_STT_CLIP_REFERENCE = (
    "Слушай, дорогой, я тебе честно скажу. Такого голоса ты еще не слышал. "
    "Садись. Покажу. Расскажу."
)

# Короткие команды — то, ради чего ассистент существует. LLM-латентность меряем
# на них (реальная длина запроса), а не на «расскажи сказку».
_LLM_PROMPTS = [
    "Открой калькулятор.",
    "Который час?",
    "Поставь громкость на пятьдесят процентов.",
]

# Фраза для TTS — одно предложение примерно той длины, которой StreamingSpeaker
# отдаёт куски в синтез.
_TTS_SENTENCE = "Секунду, сейчас посмотрю и всё сделаю."

_DEFAULT_STT_MODELS = ["base", "small", "medium"]
_DEFAULT_NFE_STEPS = [8, 16, 32]


# --------------------------------------------------------------------------- #
# утилиты
# --------------------------------------------------------------------------- #

def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate, нормализованный: нижний регистр, схлопнутые пробелы,
    без финальной пунктуации — интересует смысловое совпадение, не форматирование."""
    def norm(s: str) -> str:
        s = s.lower()
        for ch in ".,!?…":
            s = s.replace(ch, " ")
        return " ".join(s.split())

    ref = norm(reference)
    hyp = norm(hypothesis)
    if not ref:
        return 0.0
    return _levenshtein(hyp, ref) / len(ref)


def _timed(fn, runs: int) -> dict:
    """Один прогрев + `runs` замеров. Возвращает медиану/мин/макс и результат
    последнего вызова (для проверки корректности)."""
    fn()  # warmup: первый вызов тянет ленивую инициализацию / компиляцию ядер
    samples = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    return {
        "median_s": round(statistics.median(samples), 3),
        "min_s": round(min(samples), 3),
        "max_s": round(max(samples), 3),
        "runs": runs,
        "result": result,
    }


def _section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# --------------------------------------------------------------------------- #
# STT
# --------------------------------------------------------------------------- #

def bench_stt(model_sizes: list[str], runs: int) -> dict:
    _section("STT — faster-whisper, CPU int8")
    if not _STT_CLIP.exists():
        print(f"  пропуск: нет тестового аудио {_STT_CLIP}")
        return {"skipped": "нет тестового аудио"}

    import soundfile as sf
    from faster_whisper import WhisperModel

    audio, sr = sf.read(str(_STT_CLIP), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = _wav_duration_seconds(_STT_CLIP)
    print(f"  аудио: {_STT_CLIP.name}  {duration:.2f} c  {sr} Гц\n")

    header = f"  {'модель':<8} {'загрузка':>9} {'распозн.':>9} {'RTF':>7} {'CER':>7}   текст"
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = {}
    for size in model_sizes:
        try:
            t0 = time.perf_counter()
            model = WhisperModel(size, device="cpu", compute_type="int8")
            load_s = time.perf_counter() - t0

            def run(m=model):
                segments, _ = m.transcribe(
                    audio, language="ru", beam_size=1, vad_filter=True
                )
                return " ".join(s.text.strip() for s in segments).strip()

            timed = _timed(run, runs)
            text = timed.pop("result")
            rtf = timed["median_s"] / duration
            cer = _cer(text, _STT_CLIP_REFERENCE)

            preview = (text[:52] + "…") if len(text) > 53 else text
            print(
                f"  {size:<8} {load_s:>8.2f}c {timed['median_s']:>8.2f}c "
                f"{rtf:>6.2f}x {cer:>6.1%}   {preview}"
            )
            results[size] = {
                "load_s": round(load_s, 3),
                "transcribe": timed,
                "rtf": round(rtf, 3),
                "cer": round(cer, 4),
                "text": text,
                "audio_duration_s": round(duration, 3),
            }
            del model
        except Exception as e:  # noqa: BLE001 — бенчмарк не должен падать целиком
            print(f"  {size:<8} ошибка: {e}")
            results[size] = {"error": str(e)}

    return results


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #

def bench_llm(runs: int) -> dict:
    _section("LLM — Ollama, time-to-first-token")
    try:
        import ollama

        from config import LLM_MODEL, SYSTEM_PROMPT
    except Exception as e:  # noqa: BLE001
        print(f"  пропуск: {e}")
        return {"skipped": str(e)}

    try:
        ollama.list()
    except Exception as e:  # noqa: BLE001
        print(f"  пропуск: Ollama недоступен ({e})")
        return {"skipped": f"Ollama недоступен: {e}"}

    print(f"  модель: {LLM_MODEL}\n")
    header = f"  {'TTFT':>8} {'полный':>9} {'ответ':>14}   запрос"
    print(header)
    print("  " + "-" * (len(header) - 2))

    from tools import TOOL_SCHEMA

    def one_turn(prompt: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        t0 = time.perf_counter()
        ttft = None  # время до первого осмысленного чанка: текст ИЛИ вызов инструмента
        tokens = 0
        tool_calls = 0
        for chunk in ollama.chat(
            model=LLM_MODEL, messages=messages, tools=TOOL_SCHEMA, stream=True
        ):
            msg = chunk.get("message", {})
            content = msg.get("content", "")
            calls = msg.get("tool_calls") or []
            if ttft is None and (content or calls):
                ttft = time.perf_counter() - t0
            if content:
                tokens += 1
            tool_calls += len(calls)
        total = time.perf_counter() - t0
        return {"ttft_s": ttft, "total_s": total, "tokens": tokens, "tool_calls": tool_calls}

    def _median(vals: list) -> float | None:
        vals = [v for v in vals if v is not None]
        return statistics.median(vals) if vals else None

    results = {}
    for prompt in _LLM_PROMPTS:
        try:
            one_turn(prompt)  # warmup / загрузка модели в VRAM
            samples = [one_turn(prompt) for _ in range(runs)]
            ttft = _median([s["ttft_s"] for s in samples])
            total = _median([s["total_s"] for s in samples])
            tok = int(_median([s["tokens"] for s in samples]) or 0)
            calls = int(_median([s["tool_calls"] for s in samples]) or 0)
            kind = f"{tok} ток." + (f" +{calls} tool" if calls else "")
            ttft_str = f"{ttft:>7.2f}c" if ttft is not None else "   —   "
            print(f"  {ttft_str} {total:>8.2f}c {kind:>14}   {prompt}")
            results[prompt] = {
                "ttft_s": round(ttft, 3) if ttft is not None else None,
                "total_s": round(total, 3) if total is not None else None,
                "tokens": tok,
                "tool_calls": calls,
                "runs": runs,
            }
        except Exception as e:  # noqa: BLE001
            print(f"  ошибка на «{prompt}»: {e}")
            results[prompt] = {"error": str(e)}

    return results


# --------------------------------------------------------------------------- #
# TTS
# --------------------------------------------------------------------------- #

def bench_tts(nfe_steps: list[int], runs: int) -> dict:
    _section("TTS — F5-TTS, GPU")
    try:
        from tts_engine import TTSEngine
    except Exception as e:  # noqa: BLE001
        print(f"  пропуск: не удалось импортировать TTSEngine ({e})")
        return {"skipped": str(e)}

    try:
        engine = TTSEngine()
    except Exception as e:  # noqa: BLE001
        print(f"  пропуск: не удалось инициализировать F5-TTS ({e})")
        return {"skipped": str(e)}

    print(f'  фраза: «{_TTS_SENTENCE}»  ({len(_TTS_SENTENCE)} симв.)\n')
    header = f"  {'nfe_step':>9} {'синтез':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    def synth(nfe: int):
        wav, sr, _ = engine.tts.infer(
            ref_file=str(engine.ref_audio),
            ref_text=engine.ref_text,
            gen_text=_TTS_SENTENCE,
            nfe_step=nfe,
            cfg_strength=engine.cfg_strength,
        )
        return len(wav) / sr  # длительность синтезированного аудио, c

    results = {}
    for nfe in nfe_steps:
        try:
            timed = _timed(lambda n=nfe: synth(n), runs)
            audio_len = timed.pop("result")
            print(f"  {nfe:>9} {timed['median_s']:>8.2f}c   (аудио {audio_len:.1f}c)")
            results[str(nfe)] = {"synth": timed, "audio_len_s": round(audio_len, 2)}
        except Exception as e:  # noqa: BLE001
            print(f"  {nfe:>9} ошибка: {e}")
            results[str(nfe)] = {"error": str(e)}

    return results


# --------------------------------------------------------------------------- #
# сводка E2E
# --------------------------------------------------------------------------- #

def summarize_e2e(stt: dict, llm: dict, tts: dict) -> dict:
    _section("Оценка E2E — от конца речи до первого звука ответа (TTFA)")

    def best_stt():
        ok = {k: v for k, v in stt.items() if isinstance(v, dict) and "transcribe" in v}
        if not ok:
            return None, None
        # «приемлемое качество» = CER <= 15%; из подходящих берём самую быструю
        good = {k: v for k, v in ok.items() if v["cer"] <= 0.15} or ok
        name = min(good, key=lambda k: good[k]["transcribe"]["median_s"])
        return name, good[name]

    stt_name, stt_best = best_stt()
    llm_ttft = None
    if isinstance(llm, dict):
        vals = [v["ttft_s"] for v in llm.values() if isinstance(v, dict) and v.get("ttft_s")]
        llm_ttft = statistics.median(vals) if vals else None
    tts_fast = None
    if isinstance(tts, dict):
        cand = [v["synth"]["median_s"] for v in tts.values()
                if isinstance(v, dict) and "synth" in v]
        tts_fast = min(cand) if cand else None

    if not (stt_name and llm_ttft and tts_fast):
        print("  недостаточно данных — прогони все три секции")
        return {"available": False}

    stt_s = stt_best["transcribe"]["median_s"]
    ttfa = stt_s + llm_ttft + tts_fast
    print(f"  STT ({stt_name}, CER {stt_best['cer']:.1%})   {stt_s:.2f}c")
    print(f"  LLM time-to-first-token          {llm_ttft:.2f}c")
    print(f"  TTS первая фраза (min nfe)       {tts_fast:.2f}c")
    print("  " + "-" * 40)
    print(f"  TTFA (оценка)                    {ttfa:.2f}c")
    print("\n  NB: STT здесь — обработка всего клипа 9.6 c. На реальной команде")
    print("  в 2-3 c и с VAD-эндпоинтингом ступень STT будет кратно меньше.")

    return {
        "available": True,
        "stt_model": stt_name,
        "stt_s": stt_best["transcribe"]["median_s"],
        "stt_cer": stt_best["cer"],
        "llm_ttft_s": round(llm_ttft, 3),
        "tts_first_sentence_s": round(tts_fast, 3),
        "ttfa_estimate_s": round(ttfa, 3),
    }


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Бенчмарк пайплайна Гоги (STT/LLM/TTS).")
    parser.add_argument("--stt-only", action="store_true", help="только STT")
    parser.add_argument("--no-stt", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--stt-models", default=",".join(_DEFAULT_STT_MODELS),
                        help="через запятую, напр. small,medium")
    parser.add_argument("--nfe", default=",".join(map(str, _DEFAULT_NFE_STEPS)),
                        help="nfe_step для TTS через запятую")
    parser.add_argument("--runs", type=int, default=3, help="замеров на точку (медиана)")
    parser.add_argument("--json", type=str, default=None,
                        help="путь для JSON (по умолчанию benchmarks/<timestamp>.json)")
    args = parser.parse_args()

    do_stt = not args.no_stt
    do_llm = not args.no_llm and not args.stt_only
    do_tts = not args.no_tts and not args.stt_only

    stt_models = [s.strip() for s in args.stt_models.split(",") if s.strip()]
    nfe_steps = [int(s) for s in args.nfe.split(",") if s.strip()]

    print("=" * 60)
    print("GOGI PIPELINE BENCHMARK")
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    report: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runs_per_point": args.runs,
    }

    try:
        import gpu
        gpu.configure_env()
        import torch  # noqa: F401
        report["gpu_backend"] = gpu.backend_label(gpu.detect_backend())
        print(f"GPU: {report['gpu_backend']}")
    except Exception as e:  # noqa: BLE001
        report["gpu_backend"] = f"неизвестно ({e})"

    stt_res = bench_stt(stt_models, args.runs) if do_stt else {"skipped": "флаг"}
    llm_res = bench_llm(args.runs) if do_llm else {"skipped": "флаг"}
    tts_res = bench_tts(nfe_steps, args.runs) if do_tts else {"skipped": "флаг"}

    report["stt"] = stt_res
    report["llm"] = llm_res
    report["tts"] = tts_res
    report["e2e"] = summarize_e2e(stt_res, llm_res, tts_res)

    out = Path(args.json) if args.json else (
        _ROOT / "benchmarks" / f"{datetime.now():%Y-%m-%d_%H%M%S}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out.relative_to(_ROOT) if out.is_relative_to(_ROOT) else out}")


if __name__ == "__main__":
    main()
