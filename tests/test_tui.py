"""Headless-проверка GogiApp через Textual Pilot (pilot API запускает
приложение без реального терминала). Assistant мокается — без загрузки
STT/TTS/GPU."""

import pytest

import tui


@pytest.mark.asyncio
async def test_app_loads_and_shows_ready_message(mocker):
    fake_assistant = mocker.Mock()
    fake_assistant.model = "gpt-oss:20b"
    fake_assistant.tts.voice_id = "gogi"
    fake_assistant.available_voices.return_value = {"gogi": "Gogi Kavkaz"}
    fake_assistant.available_models.return_value = ["gpt-oss:20b"]
    mocker.patch("tui.Assistant", return_value=fake_assistant)

    app = tui.GogiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.assistant is fake_assistant
        assert "модель: gpt-oss:20b" in app.sub_title


@pytest.mark.asyncio
async def test_app_list_is_populated_from_config_apps():
    app = tui.GogiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        from textual.widgets import ListView

        view = app.query_one("#app-list", ListView)
        assert len(view.children) == len(tui._APP_NAMES)


@pytest.mark.asyncio
async def test_text_input_submits_and_calls_assistant_respond(mocker):
    fake_assistant = mocker.Mock()
    fake_assistant.model = "gpt-oss:20b"
    fake_assistant.tts.voice_id = "gogi"
    fake_assistant.available_voices.return_value = {}
    fake_assistant.available_models.return_value = []
    fake_assistant.respond.return_value = "привет тебе"
    mocker.patch("tui.Assistant", return_value=fake_assistant)

    app = tui.GogiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        input_widget = app.query_one("#text-input")
        input_widget.focus()
        input_widget.value = "привет"
        input_widget.post_message(input_widget.Submitted(input_widget, "привет"))
        await pilot.pause(0.2)

        fake_assistant.respond.assert_called_once_with("привет")


@pytest.mark.asyncio
async def test_toggle_recording_starts_and_interrupts_tts(mocker):
    fake_assistant = mocker.Mock()
    fake_assistant.model = "gpt-oss:20b"
    fake_assistant.tts.voice_id = "gogi"
    fake_assistant.available_voices.return_value = {}
    fake_assistant.available_models.return_value = []
    mocker.patch("tui.Assistant", return_value=fake_assistant)

    app = tui.GogiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_toggle_recording()

        fake_assistant.tts.interrupt.assert_called_once()
        fake_assistant.stt.start_recording.assert_called_once()
        assert app.recording is True


@pytest.mark.asyncio
async def test_sys_lock_action_requires_confirmation(mocker):
    lock = mocker.patch("tui.lock_screen")
    fake_assistant = mocker.Mock()
    fake_assistant.model = "gpt-oss:20b"
    fake_assistant.tts.voice_id = "gogi"
    fake_assistant.available_voices.return_value = {}
    fake_assistant.available_models.return_value = []
    mocker.patch("tui.Assistant", return_value=fake_assistant)

    app = tui.GogiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # эмулируем прямой вызов обработчика подтверждения с отказом —
        # экран подтверждения не должен исполнить действие
        app._on_lock_confirmed(False)
        lock.assert_not_called()

        app._on_lock_confirmed(True)
        lock.assert_called_once()
