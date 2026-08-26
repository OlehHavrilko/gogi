"""Запись голоса с микрофона (push-to-talk) и распознавание через faster-whisper."""

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from config import STT_MODEL_SIZE

SAMPLE_RATE = 16000


class STTEngine:
    def __init__(self, model_size: str = STT_MODEL_SIZE):
        # CPU + int8: разумный компромисс скорость/точность на Ryzen 5600 без CUDA
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def record_until_enter(self) -> np.ndarray:
        print("\n[Говорите] Нажмите Enter, когда закончите...")
        frames = []

        def callback(indata, frame_count, time_info, status):
            frames.append(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
            input()

        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames, axis=0).flatten()

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        segments, _ = self.model.transcribe(audio, language="ru", beam_size=1, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()
