"""Гоги: локальный голосовой ассистент. CLI-обёртка вокруг Assistant
(STT -> LLM с tool calling -> TTS, см. assistant.py)."""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from assistant import Assistant


def main():
    print("Загружаю STT (faster-whisper)...")
    try:
        assistant = Assistant()
    except Exception as e:
        print(f"[ошибка] не удалось загрузить ассистента: {e}")
        sys.exit(1)

    print("\n=== Гоги готов. Ctrl+C для выхода. ===")
    try:
        while True:
            try:
                audio = assistant.record()
            except Exception as e:
                print(f"[ошибка] запись с микрофона не удалась: {e}")
                continue

            text = assistant.transcribe(audio)
            if not text:
                print("(не расслышал)")
                continue

            print(f"Вы: {text}")
            print("Гоги: ", end="", flush=True)
            assistant.respond(text)
    except KeyboardInterrupt:
        print("\nПока.")
        sys.exit(0)


if __name__ == "__main__":
    main()
