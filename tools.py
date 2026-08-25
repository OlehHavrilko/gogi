"""Реестр приложений, которые ассистент может открывать по команде.
Белый список — сознательно: LLM не должен уметь запускать произвольные команды."""

import os
import subprocess

APPS = {
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "browser": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "explorer": "explorer.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "vscode": r"C:\Users\olehh\AppData\Local\Programs\Microsoft VS Code\Code.exe",
    "telegram": os.path.expandvars(r"%APPDATA%\Telegram Desktop\Telegram.exe"),
    "task_manager": "taskmgr.exe",
    "settings": "ms-settings:",
}

# Русские алиасы -> канонический ключ в APPS.
# Модель иногда называет приложение по-русски, несмотря на enum с английскими ключами.
ALIASES = {
    "хром": "chrome",
    "браузер": "browser",
    "блокнот": "notepad",
    "калькулятор": "calculator",
    "проводник": "explorer",
    "командная строка": "cmd",
    "терминал": "powershell",
    "телеграм": "telegram",
    "телеграмм": "telegram",
    "диспетчер задач": "task_manager",
    "настройки": "settings",
}

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": (
                "Открыть приложение на компьютере пользователя по имени. "
                "Всегда используй значение строго из enum (английские ключи), "
                "даже если пользователь назвал приложение по-русски."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "app": {
                        "type": "string",
                        "enum": list(APPS.keys()),
                        "description": "Имя приложения из известного списка.",
                    }
                },
                "required": ["app"],
            },
        },
    }
]


def open_app(app: str) -> str:
    app = app.lower().strip()
    app = ALIASES.get(app, app)
    target = APPS.get(app)
    if not target:
        return f"Приложение '{app}' не найдено в белом списке."
    try:
        if target.startswith("ms-settings:"):
            os.startfile(target)
        else:
            subprocess.Popen(target)
        return f"Открыл {app}."
    except Exception as e:
        return f"Не удалось открыть {app}: {e}"


def execute_tool_call(name: str, arguments: dict) -> str:
    if name == "open_app":
        return open_app(arguments.get("app", ""))
    return f"Неизвестный инструмент: {name}"
