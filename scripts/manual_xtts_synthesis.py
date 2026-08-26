"""Архив: сравнительный тест XTTS-v2, отклонён в пользу F5-TTS-Russian
(латентность 30-65с против 7-10с на этом железе, см. README). Требует
`pip install coqui-tts[codec]` — эта зависимость сознательно НЕ входит в
requirements.txt, так как в продукте не используется."""

import os
import time

os.environ.setdefault("COQUI_TOS_AGREED", "1")

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

from TTS.api import TTS

print("Загружаю XTTS-v2 на GPU...")
t0 = time.time()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print(f"Загружено за {time.time()-t0:.1f}с")

t0 = time.time()
wav = tts.tts(
    text="Конечно, открываю блокнот. Дай мне пару секунд, пожалуйста.",
    speaker_wav="voices/gogi_voice.wav",
    language="ru",
)
print(f"Синтез занял {time.time()-t0:.1f}с")

sf.write("voices/xtts_test_gogi.wav", wav, 24000)
print("Сохранено в voices/xtts_test_gogi.wav")
