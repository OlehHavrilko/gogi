"""Гоги: локальный голосовой ассистент.
STT (faster-whisper) -> LLM (gpt-oss:20b через Ollama, streaming + tool calling)
-> TTS (F5-TTS-Russian, GPU, потоково по предложениям)."""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import ollama

from config import LLM_MODEL, MAX_TOOL_ITERATIONS, SYSTEM_PROMPT
from stt_engine import STTEngine
from tools import TOOL_SCHEMA, execute_tool_call
from tts_engine import StreamingSpeaker, TTSEngine


def run_turn(messages: list, speaker: StreamingSpeaker, _depth: int = 0) -> str:
    if _depth >= MAX_TOOL_ITERATIONS:
        print("\n[warn] достигнут лимит цепочки вызовов инструментов, останавливаюсь")
        return ""

    try:
        stream = ollama.chat(model=LLM_MODEL, messages=messages, tools=TOOL_SCHEMA, stream=True)
    except Exception as e:
        print(f"\n[ошибка] не удалось связаться с Ollama: {e}")
        return ""

    full_content = ""
    tool_calls = []

    try:
        for chunk in stream:
            msg = chunk.get("message", {})
            content = msg.get("content", "")
            if content:
                print(content, end="", flush=True)
                full_content += content
                speaker.feed(content)
            if msg.get("tool_calls"):
                tool_calls.extend(msg["tool_calls"])
    except Exception as e:
        print(f"\n[ошибка] обрыв потока ответа: {e}")

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
        return run_turn(messages, speaker, _depth + 1)

    messages.append({"role": "assistant", "content": full_content})
    return full_content


def main():
    print("Загружаю STT (faster-whisper)...")
    try:
        stt = STTEngine()
    except Exception as e:
        print(f"[ошибка] не удалось загрузить STT: {e}")
        sys.exit(1)

    print("Загружаю TTS (F5-TTS-Russian)...")
    try:
        tts = TTSEngine()
    except Exception as e:
        print(f"[ошибка] не удалось загрузить TTS: {e}")
        sys.exit(1)

    speaker = StreamingSpeaker(tts)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n=== Гоги готов. Ctrl+C для выхода. ===")
    try:
        while True:
            try:
                audio = stt.record_until_enter()
            except Exception as e:
                print(f"[ошибка] запись с микрофона не удалась: {e}")
                continue

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
