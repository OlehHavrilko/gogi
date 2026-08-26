"""Запись голоса с микрофона и распознавание через faster-whisper.

Два способа записи: record_until_enter() — push-to-talk для консольного
main.py; start_recording()/stop_recording() — для GUI, где старт/стоп
управляются кликом по орбу, а не блокирующим input()."""

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from config import STT_MODEL_SIZE

SAMPLE_RATE = 16000


class STTEngine:
    def __init__(self, model_size: str = STT_MODEL_SIZE):
        # CPU + int8: разумный компромисс скорость/точность на Ryzen 5600 без CUDA
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []

    def start_recording(self) -> None:
        self._frames = []

        def callback(indata, frame_count, time_info, status):
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
        )
        self._stream.start()

    def stop_recording(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None

        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames, axis=0).flatten()

    def record_until_enter(self) -> np.ndarray:
        print("\n[Говорите] Нажмите Enter, когда закончите...")
        self.start_recording()
        input()
        return self.stop_recording()

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        segments, _ = self.model.transcribe(audio, language="ru", beam_size=1, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()
