"""Оркестрация STT -> LLM (tool calling) -> TTS в одном объекте.

Единая точка правды для диалоговой логики — используется и CLI (main.py),
и GUI (gui.py), чтобы не дублировать run_turn и переключение голоса/модели
в двух местах."""

import ollama

from config import LLM_AVAILABLE_MODELS, LLM_MODEL, MAX_TOOL_ITERATIONS, SYSTEM_PROMPT
from stt_engine import STTEngine
from tools import TOOL_SCHEMA, execute_tool_call
from tts_engine import StreamingSpeaker, TTSEngine


class Assistant:
    def __init__(self):
        self.stt = STTEngine()
        self.tts = TTSEngine()
        self.speaker = StreamingSpeaker(self.tts)
        self.model = LLM_MODEL
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # --- голос и модель (для Settings-экрана GUI) ---------------------

    def available_models(self) -> list[str]:
        return LLM_AVAILABLE_MODELS

    def switch_model(self, model: str) -> None:
        self.model = model

    def available_voices(self) -> dict[str, str]:
        return self.tts.available_voices()

    def switch_voice(self, voice_id: str) -> None:
        self.tts.set_voice(voice_id)

    # --- диалог ---------------------------------------------------------

    def record(self):
        """Блокирующая запись с микрофона (push-to-talk, Enter/Enter)."""
        return self.stt.record_until_enter()

    def transcribe(self, audio) -> str:
        return self.stt.transcribe(audio)

    def respond(self, user_text: str) -> str:
        """Полный ход диалога: добавить реплику пользователя, прогнать через
        LLM (с возможными вызовами инструментов), озвучить, дождаться конца
        воспроизведения. Возвращает финальный текстовый ответ."""
        self.messages.append({"role": "user", "content": user_text})
        reply = self._run_turn()
        self.tts.wait_until_done()
        return reply

    def _run_turn(self, _depth: int = 0) -> str:
        if _depth >= MAX_TOOL_ITERATIONS:
            print("\n[warn] достигнут лимит цепочки вызовов инструментов, останавливаюсь")
            return ""

        try:
            stream = ollama.chat(
                model=self.model, messages=self.messages, tools=TOOL_SCHEMA, stream=True
            )
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
                    self.speaker.feed(content)
                if msg.get("tool_calls"):
                    tool_calls.extend(msg["tool_calls"])
        except Exception as e:
            print(f"\n[ошибка] обрыв потока ответа: {e}")

        self.speaker.flush()
        print()

        if tool_calls:
            self.messages.append(
                {"role": "assistant", "content": full_content, "tool_calls": tool_calls}
            )
            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                result = execute_tool_call(name, args)
                print(f"[tool] {name}({args}) -> {result}")
                self.messages.append({"role": "tool", "content": result, "name": name})
            # второй проход — модель озвучивает результат выполнения
            return self._run_turn(_depth + 1)

        self.messages.append({"role": "assistant", "content": full_content})
        return full_content
