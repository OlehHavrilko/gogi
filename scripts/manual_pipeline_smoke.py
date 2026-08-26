"""Проверка полного пайплайна без микрофона: текст -> LLM -> TTS (с tool calling).
Ручной прогон на живом железе (Ollama + GPU) — не автотест."""

from assistant import Assistant

print("Загружаю ассистента (STT + TTS)...")
assistant = Assistant()

print("Гоги: ", end="", flush=True)
assistant.respond("Открой, пожалуйста, калькулятор.")
print("\n=== Готово ===")
