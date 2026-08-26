"""Проверка цикла run_turn из main.py: лимит рекурсии, обработка ошибок Ollama —
без реального обращения к Ollama/TTS/микрофону."""

import main


class _FakeSpeaker:
    def __init__(self):
        self.fed = []
        self.flushed = False

    def feed(self, token):
        self.fed.append(token)

    def flush(self):
        self.flushed = True


def _chunk(content="", tool_calls=None):
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"message": msg}


def test_run_turn_returns_plain_text(mocker):
    mocker.patch("main.ollama.chat", return_value=iter([_chunk("привет"), _chunk(" мир")]))
    messages = [{"role": "system", "content": "sys"}]
    speaker = _FakeSpeaker()

    result = main.run_turn(messages, speaker)

    assert result == "привет мир"
    assert speaker.flushed
    assert messages[-1] == {"role": "assistant", "content": "привет мир"}


def test_run_turn_handles_ollama_connection_error(mocker):
    mocker.patch("main.ollama.chat", side_effect=ConnectionError("нет связи с Ollama"))
    messages = [{"role": "system", "content": "sys"}]

    result = main.run_turn(messages, _FakeSpeaker())

    assert result == ""


def test_run_turn_stops_at_max_tool_iterations(mocker):
    call = {
        "function": {"name": "open_app", "arguments": {"app": "chrome"}},
    }
    mocker.patch("main.ollama.chat", side_effect=lambda **kwargs: iter([_chunk("", [call])]))
    mocker.patch("main.execute_tool_call", return_value="Открыл chrome.")

    messages = [{"role": "system", "content": "sys"}]
    result = main.run_turn(messages, _FakeSpeaker())

    # модель бесконечно вызывает инструмент — цикл должен остановиться
    # по MAX_TOOL_ITERATIONS, а не уйти в бесконечную рекурсию.
    assert result == ""
    assert main.ollama.chat.call_count == main.MAX_TOOL_ITERATIONS
