"""Гоги: локальный голосовой ассистент.
STT (faster-whisper) -> LLM (gpt-oss:20b через Ollama, streaming + tool calling)
-> TTS (F5-TTS-Russian, GPU, потоково по предложениям)."""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import ollama

from stt_engine import STTEngine
from tts_engine import TTSEngine, StreamingSpeaker
from tools import TOOL_SCHEMA, execute_tool_call

MODEL = "gpt-oss:20b"

SYSTEM_PROMPT = (
    "Ты — голосовой ассистент по имени Гоги. Отвечай на русском языке, "
    "кратко и естественно, как в живом разговоре — без списков и markdown-разметки, "
    "это будет озвучено вслух. Если пользователь просит открыть приложение, "
    "используй инструмент open_app."
)


def run_turn(messages: list, speaker: StreamingSpeaker) -> str:
    stream = ollama.chat(model=MODEL, messages=messages, tools=TOOL_SCHEMA, stream=True)

    full_content = ""
    tool_calls = []

    for chunk in stream:
        msg = chunk.get("message", {})
        content = msg.get("content", "")
        if content:
            print(content, end="", flush=True)
            full_content += content
            speaker.feed(content)
        if msg.get("tool_calls"):
            tool_calls.extend(msg["tool_calls"])

    speaker.flush()
    print()

    if tool_calls:
        messages.append({"role": "assistant", "content": full_content, "tool_calls": tool_calls})
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            result = execute_tool_call(name, args)
            print(f"[tool] {name}({args}) -> {result}")
            messages.append({"role": "tool", "content": result, "name": name})
        # второй проход — модель озвучивает результат выполнения
        return run_turn(messages, speaker)

    messages.append({"role": "assistant", "content": full_content})
    return full_content


def main():
    print("Загружаю STT (faster-whisper)...")
    stt = STTEngine(model_size="medium")
    print("Загружаю TTS (F5-TTS-Russian)...")
    tts = TTSEngine()
    speaker = StreamingSpeaker(tts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n=== Гоги готов. Ctrl+C для выхода. ===")
    try:
        while True:
            audio = stt.record_until_enter()
            text = stt.transcribe(audio)
            if not text:
                print("(не расслышал)")
                continue
            print(f"Вы: {text}")
            messages.append({"role": "user", "content": text})
            print("Гоги: ", end="", flush=True)
            run_turn(messages, speaker)
            tts.wait_until_done()
    except KeyboardInterrupt:
        print("\nПока.")
        sys.exit(0)


if __name__ == "__main__":
    main()
