"""Проверка whitelist-логики open_app/файловых инструментов без реального
запуска процессов и без выхода за пределы разрешённой директории."""

import config
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
    assert set(enum_values) == set(tools.APPS.keys())


def test_execute_tool_call_dispatches_open_file(mocker):
    mocker.patch("os.startfile")
    result = tools.execute_tool_call("open_file", {"path": "README.md"})
    assert "Открыл" in result


# --- файловые инструменты: whitelist по config.FILES_ROOT -------------------


def test_list_dir_returns_project_files():
    result = tools.list_dir("")
    assert isinstance(result, list)
    names = {e["name"] for e in result}
    assert "main.py" in names


def test_list_dir_rejects_path_traversal_above_root():
    result = tools.list_dir("../../")
    assert isinstance(result, dict)
    assert "error" in result


def test_list_dir_rejects_missing_directory():
    result = tools.list_dir("нет_такой_папки")
    assert isinstance(result, dict)
    assert "error" in result


def test_read_file_returns_content_within_root():
    content = tools.read_file("README.md")
    assert "Гоги" in content


def test_read_file_rejects_path_traversal_above_root():
    result = tools.read_file("../../../../Windows/System32/drivers/etc/hosts")
    assert "Доступ запрещён" in result


def test_read_file_rejects_missing_file():
    result = tools.read_file("нет_такого_файла.txt")
    assert "не найден" in result


def test_open_file_rejects_path_traversal_above_root(mocker):
    startfile = mocker.patch("os.startfile")
    result = tools.open_file("../outside.txt")
    assert "Доступ запрещён" in result
    startfile.assert_not_called()


def test_open_file_opens_file_within_root(mocker):
    startfile = mocker.patch("os.startfile")
    result = tools.open_file("README.md")
    assert "Открыл" in result
    startfile.assert_called_once()


def test_files_root_is_resolved_absolute_path():
    assert config.FILES_ROOT.is_absolute()


# --- системные действия: не в TOOL_SCHEMA, только для прямого GUI-вызова ---


def test_lock_screen_invokes_rundll32(mocker):
    run = mocker.patch("tools.subprocess.run")
    result = tools.lock_screen()
    assert "заблокирован" in result
    run.assert_called_once()


def test_sleep_system_invokes_rundll32(mocker):
    run = mocker.patch("tools.subprocess.run")
    result = tools.sleep_system()
    assert "спящий" in result
    run.assert_called_once()


def test_lock_screen_and_sleep_not_exposed_to_llm():
    tool_names = {t["function"]["name"] for t in tools.TOOL_SCHEMA}
    assert "lock_screen" not in tool_names
    assert "sleep_system" not in tool_names
