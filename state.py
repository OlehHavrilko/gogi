"""Явная машина состояний диалогового цикла.

Зачем: раньше состояние было размазано по флагам (`recording`, занятость
воркера, играет ли TTS) в трёх интерфейсах вразнобой. Теперь одна сущность:
`IDLE -> LISTENING -> TRANSCRIBING -> THINKING -> [EXECUTING_TOOL] -> SPEAKING
-> IDLE`, плюс `ERROR` и прерывание из любого состояния обратно в `LISTENING`
или `IDLE`. Переходы проверяются — нелегальный поднимает `InvalidTransition`,
что в тестах ловит рассинхрон логики.

Машина не знает про события: при смене состояния дёргает переданный колбэк
`on_change(old, new)`, а уже `Assistant` превращает это в `Event.STATE_CHANGED`.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum


class State(StrEnum):
    IDLE = "idle"                     # ждём — ни записи, ни обработки
    LISTENING = "listening"           # идёт запись с микрофона
    TRANSCRIBING = "transcribing"     # аудио есть, гоним через STT
    THINKING = "thinking"             # ждём/стримим ответ LLM
    EXECUTING_TOOL = "executing_tool" # выполняется вызов инструмента
    SPEAKING = "speaking"             # синтез/воспроизведение ответа
    ERROR = "error"                   # последний ход упал


# Куда можно перейти из каждого состояния. INTERRUPTED-переход (barge-in)
# разрешён отдельно методом `interrupt()`, а не через этот словарь.
_ALLOWED: dict[State, set[State]] = {
    State.IDLE: {State.LISTENING, State.TRANSCRIBING, State.THINKING, State.ERROR},
    State.LISTENING: {State.TRANSCRIBING, State.IDLE, State.ERROR},
    State.TRANSCRIBING: {State.THINKING, State.IDLE, State.ERROR},
    State.THINKING: {State.EXECUTING_TOOL, State.SPEAKING, State.IDLE, State.ERROR},
    State.EXECUTING_TOOL: {State.THINKING, State.SPEAKING, State.IDLE, State.ERROR},
    State.SPEAKING: {State.IDLE, State.LISTENING, State.ERROR},
    State.ERROR: {State.IDLE, State.LISTENING},
}


class InvalidTransition(RuntimeError):
    def __init__(self, current: State, target: State):
        super().__init__(f"переход {current.value} -> {target.value} недопустим")
        self.current = current
        self.target = target


class StateMachine:
    def __init__(self, on_change: Callable[[State, State], None] | None = None):
        self._state = State.IDLE
        self._on_change = on_change

    @property
    def state(self) -> State:
        return self._state

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StateMachine):
            return self._state == other._state
        if isinstance(other, State):
            return self._state == other
        return NotImplemented

    __hash__ = None  # изменяемый объект — не хэшируем

    def to(self, target: State) -> None:
        """Сменить состояние с проверкой. Повтор того же состояния — no-op."""
        if target == self._state:
            return
        if target not in _ALLOWED[self._state]:
            raise InvalidTransition(self._state, target)
        self._switch(target)

    def interrupt(self, *, to: State = State.LISTENING) -> None:
        """Barge-in: из любого рабочего состояния прыгнуть в LISTENING (по
        умолчанию) или IDLE, минуя таблицу переходов."""
        if to not in (State.LISTENING, State.IDLE):
            raise ValueError("interrupt() ведёт только в LISTENING или IDLE")
        if target := (to if to != self._state else None):
            self._switch(target)

    def reset(self) -> None:
        if self._state != State.IDLE:
            self._switch(State.IDLE)

    def _switch(self, target: State) -> None:
        old, self._state = self._state, target
        if self._on_change is not None:
            self._on_change(old, target)
