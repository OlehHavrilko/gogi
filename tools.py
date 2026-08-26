"""Реестр приложений, которые ассистент может открывать по команде.
Белый список — сознательно: LLM не должен уметь запускать произвольные команды.
Сами пути и алиасы задаются в config.yaml / config.example.yaml."""

import subprocess

from config import ALIASES, APPS

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
            import os
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
