"""Оркестрация STT -> LLM (tool calling) -> TTS в одном объекте.

Единая точка правды для диалоговой логики — используется и CLI (main.py),
и TUI (tui.py), и GUI (gui.py), чтобы не дублировать run_turn и переключение
голоса/модели.

Ядро не печатает в stdout: оно испускает события (events.py) и ведёт явную
машину состояний (state.py). Интерфейс подписывается через `assistant.on(...)`
и рендерит по-своему. Публичный API (`record`/`transcribe`/`respond`,
`switch_model`/`switch_voice`/`set_tts_params`) не изменился.
"""

import ollama

from config import LLM_AVAILABLE_MODELS, LLM_MODEL, MAX_TOOL_ITERATIONS, SYSTEM_PROMPT
from events import Event, EventEmitter
from state import State, StateMachine
from stt_engine import STTEngine
from tools import TOOL_SCHEMA, execute_tool_call
from tts_engine import StreamingSpeaker, TTSEngine


class Assistant(EventEmitter):
    def __init__(self):
        super().__init__()
        self.stt = STTEngine()
        self.tts = TTSEngine()
        self.speaker = StreamingSpeaker(self.tts)
        self.model = LLM_MODEL
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.fsm = StateMachine(
            on_change=lambda old, new: self.emit(Event.STATE_CHANGED, old=old, new=new)
        )

    @property
    def state(self) -> State:
        return self.fsm.state

    # --- голос и модель (для Settings-экрана GUI/TUI) ----------------------

    def available_models(self) -> list[str]:
        return LLM_AVAILABLE_MODELS

    def switch_model(self, model: str) -> None:
        self.model = model

    def available_voices(self) -> dict[str, str]:
        return self.tts.available_voices()

    def switch_voice(self, voice_id: str) -> None:
        self.tts.set_voice(voice_id)

    def set_tts_params(
        self, nfe_step: int | None = None, cfg_strength: float | None = None
    ) -> None:
        self.tts.set_synthesis_params(nfe_step, cfg_strength)

    # --- диалог ---------------------------------------------------------

    def interrupt(self) -> None:
        """Barge-in: оборвать то, что ассистент договаривает/обдумывает, и
        приготовиться слушать заново. Дёргается интерфейсом перед новой
        записью (в т.ч. из TUI/GUI, которые работают с self.stt напрямую)."""
        self.tts.interrupt()
        self.speaker.reset()
        self.emit(Event.INTERRUPTED)
        self.fsm.interrupt(to=State.LISTENING)

    def record(self):
        """Блокирующая запись с микрофона (push-to-talk, Enter/Enter) — для
        консольного main.py. Перед стартом barge-in."""
        self.interrupt()
        self.emit(Event.STT_STARTED)
        return self.stt.record_until_enter()

    def transcribe(self, audio) -> str:
        self.fsm.to(State.TRANSCRIBING)
        try:
            text = self.stt.transcribe(audio)
        except Exception as e:  # noqa: BLE001 — граница движка STT
            self.emit(Event.ERROR, message=f"распознавание не удалось: {e}")
            self.fsm.to(State.IDLE)
            return ""
        self.emit(Event.STT_DONE, text=text)
        if not text:
            self.fsm.to(State.IDLE)
        return text

    def respond(self, user_text: str) -> str:
        """Полный ход диалога: добавить реплику пользователя, прогнать через
        LLM (с возможными вызовами инструментов), озвучить, дождаться конца
        воспроизведения. Возвращает финальный текстовый ответ."""
        self.messages.append({"role": "user", "content": user_text})
        self.fsm.to(State.THINKING)
        self.emit(Event.LLM_STARTED)
        try:
            reply = self._run_turn()
            if reply:
                self.fsm.to(State.SPEAKING)
                self.emit(Event.TTS_STARTED)
            self.tts.wait_until_done()
            self.emit(Event.TTS_DONE)
            self.emit(Event.LLM_DONE, text=reply)
        finally:
            self.fsm.to(State.IDLE)
        return reply

    def _run_turn(self, _depth: int = 0) -> str:
        if _depth >= MAX_TOOL_ITERATIONS:
            self.emit(
                Event.ERROR,
                message="достигнут лимит цепочки вызовов инструментов, останавливаюсь",
            )
            return ""

        try:
            stream = ollama.chat(
                model=self.model, messages=self.messages, tools=TOOL_SCHEMA, stream=True
            )
        except Exception as e:  # noqa: BLE001 — граница Ollama
            self.emit(Event.ERROR, message=f"не удалось связаться с Ollama: {e}")
            return ""

        full_content = ""
        tool_calls = []

        try:
            for chunk in stream:
                msg = chunk.get("message", {})
                content = msg.get("content", "")
                if content:
                    self.emit(Event.LLM_TOKEN, token=content)
                    full_content += content
                    self.speaker.feed(content)
                if msg.get("tool_calls"):
                    tool_calls.extend(msg["tool_calls"])
        except Exception as e:  # noqa: BLE001 — обрыв потока на середине
            self.emit(Event.ERROR, message=f"обрыв потока ответа: {e}")

        self.speaker.flush()

        if tool_calls:
            self.messages.append(
                {"role": "assistant", "content": full_content, "tool_calls": tool_calls}
            )
            self.fsm.to(State.EXECUTING_TOOL)
            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                self.emit(Event.TOOL_STARTED, name=name, args=args)
                result = execute_tool_call(name, args)
                self.emit(Event.TOOL_DONE, name=name, result=result)
                self.messages.append({"role": "tool", "content": result, "name": name})
            self.fsm.to(State.THINKING)
            # второй проход — модель озвучивает результат выполнения
            return self._run_turn(_depth + 1)

        self.messages.append({"role": "assistant", "content": full_content})
        return full_content
