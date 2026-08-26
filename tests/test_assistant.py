"""Проверка Assistant._run_turn: лимит рекурсии, обработка ошибок Ollama,
переключение модели/голоса — без реального обращения к Ollama/TTS/микрофону."""

import assistant as assistant_module


class _FakeSpeaker:
    def __init__(self):
        self.flushed = False

    def feed(self, token):
        pass

    def flush(self):
        self.flushed = True


def _chunk(content="", tool_calls=None):
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"message": msg}


def _make_assistant(mocker):
    mocker.patch.object(assistant_module, "STTEngine")
    mocker.patch.object(assistant_module, "TTSEngine")
    a = assistant_module.Assistant()
    a.speaker = _FakeSpeaker()
    return a


def test_run_turn_returns_plain_text(mocker):
    mocker.patch("assistant.ollama.chat", return_value=iter([_chunk("привет"), _chunk(" мир")]))
    a = _make_assistant(mocker)

    result = a._run_turn()

    assert result == "привет мир"
    assert a.speaker.flushed
    assert a.messages[-1] == {"role": "assistant", "content": "привет мир"}


def test_run_turn_handles_ollama_connection_error(mocker):
    mocker.patch("assistant.ollama.chat", side_effect=ConnectionError("нет связи"))
    a = _make_assistant(mocker)

    assert a._run_turn() == ""


def test_run_turn_stops_at_max_tool_iterations(mocker):
    call = {"function": {"name": "open_app", "arguments": {"app": "chrome"}}}
    mocker.patch("assistant.ollama.chat", side_effect=lambda **kw: iter([_chunk("", [call])]))
    mocker.patch("assistant.execute_tool_call", return_value="Открыл chrome.")
    a = _make_assistant(mocker)

    result = a._run_turn()

    # модель бесконечно вызывает инструмент — цикл должен остановиться по
    # MAX_TOOL_ITERATIONS, а не уйти в бесконечную рекурсию.
    assert result == ""
    assert assistant_module.ollama.chat.call_count == assistant_module.MAX_TOOL_ITERATIONS


def test_switch_model_changes_model_used_in_next_call(mocker):
    chat = mocker.patch("assistant.ollama.chat", return_value=iter([_chunk("ok")]))
    a = _make_assistant(mocker)

    a.switch_model("qwen3.5:9b")
    a._run_turn()

    assert chat.call_args.kwargs["model"] == "qwen3.5:9b"


def test_switch_voice_delegates_to_tts_engine(mocker):
    a = _make_assistant(mocker)

    a.switch_voice("vova")

    a.tts.set_voice.assert_called_once_with("vova")


def test_respond_appends_user_message_and_waits_for_tts(mocker):
    mocker.patch("assistant.ollama.chat", return_value=iter([_chunk("ок")]))
    a = _make_assistant(mocker)

    result = a.respond("привет")

    assert a.messages[1] == {"role": "user", "content": "привет"}
    assert result == "ок"
    a.tts.wait_until_done.assert_called_once()
