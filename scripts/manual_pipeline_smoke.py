"""Проверка полного пайплайна без микрофона: текст -> LLM -> TTS (с tool calling).
Ручной прогон на живом железе (Ollama + GPU) — не автотест.

Показывает события ядра (стриминг токенов, вызовы инструментов, смена
состояний, ошибки) — то, что раньше ядро печатало само, а теперь отдаёт
наружу через events.py."""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from assistant import Assistant
from events import Event

print("Загружаю ассистента (STT + TTS)...")
assistant = Assistant()

assistant.on(Event.STATE_CHANGED, lambda old, new: print(f"\n  [{old} -> {new}]"))
assistant.on(Event.LLM_TOKEN, lambda token: print(token, end="", flush=True))
assistant.on(Event.LLM_DONE, lambda text: print())
assistant.on(Event.TOOL_STARTED, lambda name, args: print(f"  >> {name}({args})"))
assistant.on(Event.TOOL_DONE, lambda name, result: print(f"  << {result}"))
assistant.on(Event.ERROR, lambda message: print(f"  [ошибка] {message}"))

print("\nВы: Открой, пожалуйста, калькулятор.")
print("Гоги: ", end="", flush=True)
assistant.respond("Открой, пожалуйста, калькулятор.")
print("\n=== Готово ===")
