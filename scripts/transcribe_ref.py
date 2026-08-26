import soundfile as sf

from stt_engine import STTEngine

audio, sr = sf.read("voices/gogi_voice_16k.wav", dtype="float32")

stt = STTEngine(model_size="medium")
segments, info = stt.model.transcribe(audio, beam_size=1, vad_filter=True)
text = " ".join(seg.text.strip() for seg in segments).strip()
print("Detected language:", info.language)
print("TRANSCRIPT:", text)

with open("voices/gogi_transcript.txt", "w", encoding="utf-8") as f:
    f.write(text)
