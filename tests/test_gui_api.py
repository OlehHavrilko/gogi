"""Проверка gui.Api — моста между фронтендом (pywebview) и Assistant/tools,
без реального окна/браузера и без загрузки моделей."""

import gui


def _api_with_fake_assistant(mocker):
    api = gui.Api()
    fake_assistant = mocker.Mock()
    fake_assistant.available_voices.return_value = {"gogi": "Gogi Kavkaz"}
    fake_assistant.available_models.return_value = ["gpt-oss:20b"]
    fake_assistant.tts.voice_id = "gogi"
    fake_assistant.model = "gpt-oss:20b"
    fake_assistant.stt.stop_recording.return_value = "AUDIO"
    fake_assistant.transcribe.return_value = "привет"
    fake_assistant.respond.return_value = "привет тебе"
    api.assistant = fake_assistant
    return api, fake_assistant


def test_init_builds_assistant_once(mocker):
    mock_assistant_cls = mocker.patch("gui.Assistant")
    mock_assistant_cls.return_value.available_voices.return_value = {}
    mock_assistant_cls.return_value.available_models.return_value = []
    mock_assistant_cls.return_value.tts.voice_id = None
    mock_assistant_cls.return_value.model = "gpt-oss:20b"

    api = gui.Api()
    api.init()
    api.init()

    mock_assistant_cls.assert_called_once()


def test_start_recording_interrupts_tts_first(mocker):
    api, fake = _api_with_fake_assistant(mocker)

    api.start_recording()

    fake.tts.interrupt.assert_called_once()
    fake.stt.start_recording.assert_called_once()


def test_set_tts_params_delegates_to_assistant(mocker):
    api, fake = _api_with_fake_assistant(mocker)

    api.set_tts_params(16, 2.0)

    fake.set_tts_params.assert_called_once_with(16, 2.0)


def test_stop_recording_and_respond_returns_reply(mocker):
    api, fake = _api_with_fake_assistant(mocker)

    result = api.stop_recording_and_respond()

    assert result == {"user_text": "привет", "reply": "привет тебе"}
    fake.respond.assert_called_once_with("привет")


def test_stop_recording_and_respond_handles_empty_transcript(mocker):
    api, fake = _api_with_fake_assistant(mocker)
    fake.transcribe.return_value = ""

    result = api.stop_recording_and_respond()

    assert result == {"user_text": "", "reply": ""}
    fake.respond.assert_not_called()


def test_send_text_delegates_to_assistant_respond(mocker):
    api, fake = _api_with_fake_assistant(mocker)

    result = api.send_text("тест")

    assert result == {"user_text": "тест", "reply": "привет тебе"}
    fake.respond.assert_called_once_with("тест")


def test_switch_voice_and_model_delegate(mocker):
    api, fake = _api_with_fake_assistant(mocker)

    api.switch_voice("vova")
    api.switch_model("qwen3.5:9b")

    fake.switch_voice.assert_called_once_with("vova")
    fake.switch_model.assert_called_once_with("qwen3.5:9b")


def test_list_apps_returns_whitelist_keys():
    api = gui.Api()
    apps = api.list_apps()
    assert "chrome" in apps


def test_open_app_uses_real_whitelist(mocker):
    popen = mocker.patch("tools.subprocess.Popen")
    api = gui.Api()
    result = api.open_app("chrome")
    assert "Открыл" in result
    popen.assert_called_once()


def test_file_tools_are_whitelisted(mocker):
    api = gui.Api()
    result = api.list_dir("")
    assert isinstance(result, list)

    content = api.read_file("README.md")
    assert "Гоги" in content

    startfile = mocker.patch("os.startfile")
    open_result = api.open_file("../outside.txt")
    assert "Доступ запрещён" in open_result
    startfile.assert_not_called()
