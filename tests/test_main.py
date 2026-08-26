"""Проверка main.main(): загрузка ассистента и обработка EOF на stdin —
без реального микрофона/STT/TTS."""

import pytest

import main


def test_main_exits_cleanly_when_assistant_fails_to_load(mocker):
    mocker.patch("main.Assistant", side_effect=RuntimeError("нет GPU"))

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 1


def test_main_exits_on_eof_instead_of_retry_looping(mocker):
    fake_assistant = mocker.Mock()
    fake_assistant.record.side_effect = EOFError()
    mocker.patch("main.Assistant", return_value=fake_assistant)

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 1
    # именно одна попытка — не бесконечный цикл retry на EOF
    assert fake_assistant.record.call_count == 1


def test_main_handles_keyboard_interrupt_and_exits_zero(mocker):
    fake_assistant = mocker.Mock()
    fake_assistant.record.side_effect = KeyboardInterrupt()
    mocker.patch("main.Assistant", return_value=fake_assistant)

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 0
