"""Гоги: локальный голосовой ассистент. CLI-обёртка вокруг Assistant
(STT -> LLM с tool calling -> TTS, см. assistant.py).

Ядро больше не печатает само — оно испускает события. Консоль подписывается
на них здесь и выводит поток токенов, вызовы инструментов и ошибки в stdout."""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from assistant import Assistant
from events import Event


def _wire_console_output(assistant: Assistant) -> None:
    assistant.on(Event.LLM_TOKEN, lambda token: print(token, end="", flush=True))
    assistant.on(Event.LLM_DONE, lambda text: print())
    assistant.on(
        Event.TOOL_STARTED,
        lambda name, args: print(f"\n[инструмент] {name}({args})...", flush=True),
    )
    assistant.on(Event.TOOL_DONE, lambda name, result: print(f"[инструмент] {name} -> {result}"))
    assistant.on(Event.ERROR, lambda message: print(f"\n[ошибка] {message}"))


def main():
    print("Загружаю STT (faster-whisper)...")
    try:
        assistant = Assistant()
    except Exception as e:
        print(f"[ошибка] не удалось загрузить ассистента: {e}")
        sys.exit(1)

    _wire_console_output(assistant)

    print("\n=== Гоги готов. Ctrl+C для выхода. ===")
    try:
        while True:
            try:
                audio = assistant.record()
            except EOFError:
                # stdin не интерактивен (нет реального терминала) — Enter
                # никогда не придёт, ретраить дальше бессмысленно и опасно:
                # быстрый цикл открытия/закрытия InputStream может уронить
                # аудио-подсистему.
                print("\n[ошибка] stdin не интерактивен, выхожу.")
                sys.exit(1)
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
