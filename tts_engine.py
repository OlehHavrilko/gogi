"""Локальный TTS на F5-TTS-Russian (hotstone228/F5-TTS-Russian), GPU через ROCm.

Важные особенности среды, без которых модель не запустится или даёт мусор:
- torch.distributed отсутствует в ROCm-сборке torch для Windows — encodec (вокодер
  внутри F5-TTS) обращается к ReduceOp на этапе импорта, патчим заглушкой.
- torchaudio.load() в этой версии требует torchcodec, а его .dll собран под другой
  ABI torch и не грузится — грузим аудио через soundfile напрямую.
- Чекпоинт обучен на архитектуре F5TTS_Base (НЕ F5TTS_v1_Base, это дефолт
  библиотеки) — с неправильным конфигом модель генерирует нечленораздельный шум.
"""

import os
import re
import threading
import queue
from pathlib import Path

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torchaudio
import soundfile as sf
import sounddevice as sd

if not torch.distributed.is_available():
    class _StubReduceOp:
        SUM = None
    torch.distributed.ReduceOp = _StubReduceOp


def _load_via_soundfile(filepath, **kwargs):
    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


torchaudio.load = _load_via_soundfile

from f5_tts.api import F5TTS

VOICES_DIR = Path(__file__).parent / "voices"
REF_AUDIO = VOICES_DIR / "gogi_voice.wav"
REF_TEXT_FILE = VOICES_DIR / "gogi_transcript.txt"
NFE_STEP = 32
CFG_STRENGTH = 3.0

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


class TTSEngine:
    def __init__(self):
        self.tts = F5TTS(
            model="F5TTS_Base",
            ckpt_file=str(VOICES_DIR / "f5_ru" / "model_last.safetensors"),
            vocab_file=str(VOICES_DIR / "f5_ru" / "vocab.txt"),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.ref_text = REF_TEXT_FILE.read_text(encoding="utf-8").strip()

        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _synthesize(self, text: str):
        text = text.strip()
        if not text:
            return
        try:
            wav, sr, _ = self.tts.infer(
                ref_file=str(REF_AUDIO),
                ref_text=self.ref_text,
                gen_text=text,
                nfe_step=NFE_STEP,
                cfg_strength=CFG_STRENGTH,
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
    """Собирает токены в предложения и сразу отдаёт их в TTS,
    пока LLM продолжает генерировать следующие."""

    def __init__(self, engine: TTSEngine):
        self.engine = engine
        self._buffer = ""

    def feed(self, token: str):
        self._buffer += token
        parts = _SENTENCE_SPLIT.split(self._buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                if sentence.strip():
                    self.engine.say(sentence.strip())
            self._buffer = parts[-1]

    def flush(self):
        if self._buffer.strip():
            self.engine.say(self._buffer.strip())
        self._buffer = ""
