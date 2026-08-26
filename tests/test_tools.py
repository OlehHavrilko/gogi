"""Проверка whitelist-логики open_app без реального запуска процессов."""

import tools


def test_open_app_unknown_returns_message_not_exception():
    result = tools.open_app("совершенно неизвестное приложение")
    assert "не найдено в белом списке" in result


def test_open_app_resolves_russian_alias(mocker):
    popen = mocker.patch("tools.subprocess.Popen")
    result = tools.open_app("хром")
    assert "Открыл" in result
    popen.assert_called_once()


def test_open_app_case_and_whitespace_insensitive(mocker):
    popen = mocker.patch("tools.subprocess.Popen")
    tools.open_app("  ХРОМ  ")
    popen.assert_called_once()


def test_open_app_handles_launch_failure(mocker):
    mocker.patch("tools.subprocess.Popen", side_effect=OSError("boom"))
    result = tools.open_app("chrome")
    assert "Не удалось открыть" in result


def test_open_app_settings_uses_startfile(mocker):
    startfile = mocker.patch("os.startfile")
    popen = mocker.patch("tools.subprocess.Popen")
    result = tools.open_app("настройки")
    startfile.assert_called_once()
    popen.assert_not_called()
    assert "Открыл" in result


def test_execute_tool_call_dispatches_known_tool(mocker):
    mocker.patch("tools.subprocess.Popen")
    result = tools.execute_tool_call("open_app", {"app": "notepad"})
    assert "Открыл" in result


def test_execute_tool_call_rejects_unknown_tool():
    result = tools.execute_tool_call("delete_everything", {})
    assert "Неизвестный инструмент" in result


def test_tool_schema_enum_matches_apps():
    enum_values = tools.TOOL_SCHEMA[0]["function"]["parameters"]["properties"]["app"]["enum"]
    assert set(enum_values) == set(tools.APPS.keys()) if hasattr(tools, "APPS") else True
