"""Событийная шина."""

from events import Event, EventEmitter


def test_emit_calls_listener_with_payload():
    bus = EventEmitter()
    seen = []
    bus.on(Event.LLM_TOKEN, lambda token: seen.append(token))
    bus.emit(Event.LLM_TOKEN, token="привет")
    assert seen == ["привет"]


def test_multiple_listeners_all_fire_in_order():
    bus = EventEmitter()
    order = []
    bus.on(Event.TTS_DONE, lambda: order.append("a"))
    bus.on(Event.TTS_DONE, lambda: order.append("b"))
    bus.emit(Event.TTS_DONE)
    assert order == ["a", "b"]


def test_emit_without_listeners_is_silent():
    EventEmitter().emit(Event.ERROR, message="никто не слушает")


def test_broken_listener_does_not_block_others(capsys):
    bus = EventEmitter()
    hits = []
    bus.on(Event.ERROR, lambda message: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.on(Event.ERROR, lambda message: hits.append(message))
    bus.emit(Event.ERROR, message="x")
    assert hits == ["x"]
    assert "boom" in capsys.readouterr().err


def test_unsubscribe():
    bus = EventEmitter()
    seen = []
    off = bus.on(Event.LLM_TOKEN, lambda token: seen.append(token))
    bus.emit(Event.LLM_TOKEN, token="1")
    off()
    bus.emit(Event.LLM_TOKEN, token="2")
    assert seen == ["1"]
