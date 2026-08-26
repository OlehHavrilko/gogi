"""Проверка TTSEngine: barge-in (interrupt) и рантайм-параметры синтеза —
без реальной загрузки F5TTS/torch (конструктор не вызывается)."""

import queue

import tts_engine


def _bare_engine():
    """TTSEngine без __init__ (не грузит F5TTS/GPU), но с реальной очередью."""
    engine = tts_engine.TTSEngine.__new__(tts_engine.TTSEngine)
    engine._queue = queue.Queue()
    engine.nfe_step = 32
    engine.cfg_strength = 3.0
    return engine


def test_set_synthesis_params_updates_only_given_values():
    engine = _bare_engine()

    engine.set_synthesis_params(nfe_step=16)
    assert engine.nfe_step == 16
    assert engine.cfg_strength == 3.0

    engine.set_synthesis_params(cfg_strength=2.0)
    assert engine.nfe_step == 16
    assert engine.cfg_strength == 2.0


def test_interrupt_drains_pending_queue(mocker):
    mocker.patch.object(tts_engine.sd, "stop")
    engine = _bare_engine()
    engine._queue.put("фраза 1")
    engine._queue.put("фраза 2")

    engine.interrupt()

    assert engine._queue.empty()
    assert engine._queue.unfinished_tasks == 0
    tts_engine.sd.stop.assert_called_once()


def test_interrupt_on_empty_queue_still_stops_playback(mocker):
    mocker.patch.object(tts_engine.sd, "stop")
    engine = _bare_engine()

    engine.interrupt()  # не должно бросить исключение на пустой очереди

    tts_engine.sd.stop.assert_called_once()
