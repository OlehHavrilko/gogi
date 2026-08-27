"""Локальный TTS на F5-TTS-Russian (hotstone228/F5-TTS-Russian), GPU через
NVIDIA CUDA или AMD ROCm (бэкенд определяется автоматически, см. gpu.py).

Важные особенности среды, без которых модель не запустится или даёт мусор:
- torch.distributed отсутствует в ROCm-сборке torch для Windows — encodec (вокодер
  внутри F5-TTS) обращается к ReduceOp на этапе импорта, патчим заглушкой.
  На CUDA-сборках distributed обычно доступен, патч в этом случае не срабатывает.
- torchaudio.load() в некоторых сборках требует torchcodec, а его .dll собран под
  другой ABI torch и не грузится — грузим аудио через soundfile напрямую всегда,
  это безопасно независимо от вендора GPU.
- Чекпоинт обучен на архитектуре F5TTS_Base (НЕ F5TTS_v1_Base, это дефолт
  библиотеки) — с неправильным конфигом модель генерирует нечленораздельный шум.
"""

import queue
import re
import threading
from pathlib import Path

import gpu

gpu.configure_env()

import sounddevice as sd
import soundfile as sf
import torch
import torchaudio

if not torch.distributed.is_available():
    class _StubReduceOp:
        SUM = None
    torch.distributed.ReduceOp = _StubReduceOp


def _load_via_soundfile(filepath, **kwargs):
    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


torchaudio.load = _load_via_soundfile

from f5_tts.api import F5TTS

from config import TTS_CFG_STRENGTH, TTS_DEFAULT_VOICE, TTS_NFE_STEP, TTS_VOICES

_PROJECT_ROOT = Path(__file__).parent
VOICES_DIR = _PROJECT_ROOT / "voices"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
# Для первого куска ответа режем раньше — по запятой/тире/двоеточию, — чтобы
# «Секунду,» ушло в синтез сразу, а не ждало первой точки. LLM
# time-to-first-token сейчас доминирует в задержке (см. ROADMAP), и этот кусок
# закрывает паузу перед голосом.
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:—])\s+")
# Аварийный предел: если ни точки, ни запятой долго нет — не копим бесконечно.
_FIRST_CHUNK_HARD_CAP = 60


class TTSEngine:
    def __init__(self):
        backend = gpu.detect_backend()
        print(f"[TTS] GPU-бэкенд: {gpu.backend_label(backend)}")
        device = "cuda" if backend in ("cuda", "rocm") else "cpu"

        self.tts = F5TTS(
            model="F5TTS_Base",
            ckpt_file=str(VOICES_DIR / "f5_ru" / "model_last.safetensors"),
            vocab_file=str(VOICES_DIR / "f5_ru" / "vocab.txt"),
            device=device,
        )

        self.voice_id: str = ""
        self.ref_audio: Path | None = None
        self.ref_text: str = ""
        self.set_voice(TTS_DEFAULT_VOICE)

        self.nfe_step = TTS_NFE_STEP
        self.cfg_strength = TTS_CFG_STRENGTH

        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def available_voices(self) -> dict[str, str]:
        """{voice_id: отображаемое имя} — для выпадающего списка в GUI."""
        return {vid: v.get("name", vid) for vid, v in TTS_VOICES.items()}

    def set_voice(self, voice_id: str) -> None:
        """Переключить референс-голос. Синтез берёт self.ref_audio/ref_text
        на каждый вызов (не запечён в модель при загрузке), поэтому
        переключение работает без перезагрузки F5TTS."""
        voice = TTS_VOICES.get(voice_id)
        if voice is None:
            raise ValueError(f"Голос '{voice_id}' не найден в конфиге (tts.voices).")
        self.ref_audio = _PROJECT_ROOT / voice["ref_audio"]
        self.ref_text = (_PROJECT_ROOT / voice["ref_text"]).read_text(encoding="utf-8").strip()
        self.voice_id = voice_id

    def set_synthesis_params(
        self, nfe_step: int | None = None, cfg_strength: float | None = None
    ) -> None:
        """Изменить параметры синтеза на лету (слайдеры в Settings) — влияет
        только на следующие фразы, применяется без перезагрузки модели."""
        if nfe_step is not None:
            self.nfe_step = nfe_step
        if cfg_strength is not None:
            self.cfg_strength = cfg_strength

    def interrupt(self) -> None:
        """Barge-in: остановить текущее воспроизведение и выбросить ещё не
        озвученные фразы из очереди — пользователь начал говорить снова, не
        дожидаясь, пока ассистент договорит. Best-effort: фраза, которую
        воркер уже успел выхватить из очереди в момент вызова, доиграет."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()
        sd.stop()

    def _synthesize(self, text: str):
        text = text.strip()
        if not text:
            return
        try:
            wav, sr, _ = self.tts.infer(
                ref_file=str(self.ref_audio),
                ref_text=self.ref_text,
                gen_text=text,
                nfe_step=self.nfe_step,
                cfg_strength=self.cfg_strength,
            )
            sd.play(wav, sr)
            sd.wait()
        except Exception as e:
            print(f"[TTS error] {e}")

    def _run(self):
        while True:
            text = self._queue.get()
            if text is None:
                self._queue.task_done()
                break
            self._synthesize(text)
            self._queue.task_done()

    def say(self, text: str):
        """Поставить фразу в очередь на озвучку (неблокирующе)."""
        self._queue.put(text)

    def wait_until_done(self):
        self._queue.join()

    def stop(self):
        self._queue.put(None)


class StreamingSpeaker:
    """Собирает токены в предложения и сразу отдаёт их в TTS, пока LLM
    генерирует следующие. Первый кусок каждого ответа отдаётся по более
    раннему рубежу (запятая/двоеточие/тире или предел по длине) — чтобы
    сократить паузу от конца речи пользователя до первого звука ответа."""

    def __init__(self, engine: TTSEngine):
        self.engine = engine
        self._buffer = ""
        self._first_chunk_sent = False

    def feed(self, token: str):
        self._buffer += token
        if not self._first_chunk_sent and self._try_emit_first_chunk():
            return
        parts = _SENTENCE_SPLIT.split(self._buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                if sentence.strip():
                    self.engine.say(sentence.strip())
                    self._first_chunk_sent = True
            self._buffer = parts[-1]

    def _try_emit_first_chunk(self) -> bool:
        """Ранний рубеж для самого первого куска ответа. True — что-то ушло в
        синтез, буфер обрезан."""
        head, sep, tail = _split_once(self._buffer)
        if sep is None and len(self._buffer) < _FIRST_CHUNK_HARD_CAP:
            return False
        if sep is None:  # предел по длине — режем по последнему пробелу
            cut = self._buffer.rfind(" ", 0, _FIRST_CHUNK_HARD_CAP)
            if cut <= 0:
                return False
            head, tail = self._buffer[:cut], self._buffer[cut + 1:]
        if not head.strip():
            self._buffer = tail
            return False
        self.engine.say(head.strip())
        self._first_chunk_sent = True
        self._buffer = tail
        return True

    def flush(self):
        if self._buffer.strip():
            self.engine.say(self._buffer.strip())
        self._buffer = ""
        self._first_chunk_sent = False  # следующий ход — снова ранний рубеж

    def reset(self):
        """Barge-in: выбросить недособранный хвост предложения, чтобы он не
        уехал в синтез в начале следующего хода."""
        self._buffer = ""
        self._first_chunk_sent = False


def _split_once(text: str) -> tuple[str, str | None, str]:
    """Первый рубеж — конец предложения ИЛИ конец клаузы. Возвращает
    (голова, разделитель|None, хвост). Разделитель None — рубежа нет."""
    best = None
    for rx in (_SENTENCE_SPLIT, _CLAUSE_SPLIT):
        m = rx.search(text)
        if m and (best is None or m.start() < best.start()):
            best = m
    if best is None:
        return text, None, ""
    return text[: best.start()], text[best.start(): best.end()], text[best.end():]
