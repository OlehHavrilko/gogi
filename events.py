"""Событийная шина ассистента.

Зачем: раньше `Assistant` на всё реагировал `print()` — в консоли видно, в
TUI Textual перехватывает экран, в GUI вывод уходит в никуда. Теперь ядро не
печатает, а испускает события; каждый интерфейс подписывается и рендерит их
по-своему (консоль — `print`, TUI — в `RichLog`, GUI — в транскрипт). Этим же
разблокируется стриминг текста: токены долетают до экрана по мере генерации,
а не одним куском в конце.

Модель — простые колбэки, не очередь: у всех трёх потребителей уже есть свой
способ увести вызов в UI-поток (`call_from_thread` в Textual, evaluate_js в
pywebview), навязывать сверху ещё одну очередь незачем. `emit()` вызывает
слушателей синхронно в том потоке, где произошло событие; слушатель сам
отвечает за маршалинг. Исключение в одном слушателе не мешает остальным и не
роняет ядро.
"""

from __future__ import annotations

import sys
import traceback
from collections import defaultdict
from collections.abc import Callable
from enum import StrEnum


class Event(StrEnum):
    """Что произошло. Полезная нагрузка — в kwargs `emit()`, ключи указаны рядом."""

    STATE_CHANGED = "state_changed"   # old: State, new: State
    STT_STARTED = "stt_started"       # —
    STT_DONE = "stt_done"             # text: str
    LLM_STARTED = "llm_started"       # —
    LLM_TOKEN = "llm_token"           # token: str
    LLM_DONE = "llm_done"             # text: str
    TOOL_STARTED = "tool_started"     # name: str, args: dict
    TOOL_DONE = "tool_done"           # name: str, result: str
    TTS_STARTED = "tts_started"       # —
    TTS_DONE = "tts_done"             # —
    INTERRUPTED = "interrupted"       # —
    ERROR = "error"                   # message: str


Listener = Callable[..., None]


class EventEmitter:
    """Примесь: подписка (`on`) и рассылка (`emit`). Наследуется `Assistant`."""

    def __init__(self) -> None:
        self._listeners: dict[Event, list[Listener]] = defaultdict(list)

    def on(self, event: Event, callback: Listener) -> Callable[[], None]:
        """Подписаться. Возвращает функцию-отписку."""
        self._listeners[event].append(callback)
        return lambda: self._listeners[event].remove(callback)

    def emit(self, event: Event, **payload) -> None:
        for callback in list(self._listeners.get(event, ())):
            try:
                callback(**payload)
            except Exception:  # noqa: BLE001 — сломанный слушатель не должен ронять ядро
                print(f"[events] слушатель {event.value} упал:", file=sys.stderr)
                traceback.print_exc()
