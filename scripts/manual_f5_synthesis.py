import time

import soundfile as sf
import torch
import torchaudio

if not torch.distributed.is_available():
    # ROCm-сборка torch для Windows не включает torch.distributed;
    # encodec (вокодер внутри F5-TTS) обращается к ReduceOp на этапе импорта,
    # хотя реально distributed-режим никогда не используется при одиночном GPU.
    class _StubReduceOp:
        SUM = None

    torch.distributed.ReduceOp = _StubReduceOp


def _load_via_soundfile(filepath, **kwargs):
    # torchaudio.load() в этой версии требует torchcodec, а его нативный .dll
    # собран под другой ABI torch и не грузится с нашей ROCm-сборкой.
    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


torchaudio.load = _load_via_soundfile

from f5_tts.api import F5TTS

print("Загружаю F5-TTS-Russian...")
t0 = time.time()
tts = F5TTS(
    model="F5TTS_Base",
    ckpt_file="voices/f5_ru/model_last.safetensors",
    vocab_file="voices/f5_ru/vocab.txt",
    device="cuda",
)
print(f"Загружено за {time.time()-t0:.1f}с")

with open("voices/gogi_transcript.txt", encoding="utf-8") as f:
    ref_text = f.read().strip()
gen_text = "Конечно, открываю блокнот. Дай мне пару секунд, пожалуйста."

t0 = time.time()
wav, sr, _ = tts.infer(
    ref_file="voices/gogi_voice.wav",
    ref_text=ref_text,
    gen_text=gen_text,
    file_wave="voices/f5_test_out_gogi_nfe32_cfg3.wav",
    nfe_step=32,
    cfg_strength=3.0,
)
print(f"Синтез занял {time.time()-t0:.1f}с (sr={sr}, samples={len(wav)})")
