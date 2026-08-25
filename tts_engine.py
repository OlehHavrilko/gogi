"""Локальный TTS на Silero — полностью офлайн после первой загрузки модели."""

import re
import threading
import queue

import torch
import sounddevice as sd

SAMPLE_RATE = 48000
SPEAKER = "eugene"  # мужской голос; варианты: aidar, baya, kseniya, xenia, eugene, random

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


class TTSEngine:
    def __init__(self):
        torch.set_num_threads(4)
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v4_ru",
        )
        self.model.to(torch.device("cpu"))
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _synthesize(self, text: str):
        text = text.strip()
        if not text:
            return
        try:
            audio = self.model.apply_tts(text=text, speaker=SPEAKER, sample_rate=SAMPLE_RATE)
            sd.play(audio.numpy(), SAMPLE_RATE)
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
