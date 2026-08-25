"""Проверка полного пайплайна без микрофона: текст -> LLM -> TTS (с tool calling)."""
import ollama
from tts_engine import TTSEngine, StreamingSpeaker
from tools import TOOL_SCHEMA, execute_tool_call

MODEL = "gpt-oss:20b"
SYSTEM_PROMPT = (
    "Ты — голосовой ассистент по имени Домовой. Отвечай на русском языке, "
    "кратко и естественно, как в живом разговоре — без списков и markdown-разметки, "
    "это будет озвучено вслух. Если пользователь просит открыть приложение, "
    "используй инструмент open_app."
)


def run_turn(messages, speaker):
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
        return run_turn(messages, speaker)
    messages.append({"role": "assistant", "content": full_content})
    return full_content


print("Загружаю TTS (F5-TTS-Russian)...")
tts = TTSEngine()
speaker = StreamingSpeaker(tts)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
messages.append({"role": "user", "content": "Открой, пожалуйста, калькулятор."})

print("Домовой: ", end="", flush=True)
run_turn(messages, speaker)
tts.wait_until_done()
print("\n=== Готово ===")
