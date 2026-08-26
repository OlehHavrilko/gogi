"""Инструменты, которые ассистент может вызывать — по командам голоса (через
LLM tool calling, TOOL_SCHEMA) и напрямую из GUI (список ниже, не всё
экспортируется в TOOL_SCHEMA).

Везде — сознательный whitelist: приложения из config.APPS, файлы только
внутри config.FILES_ROOT. LLM не может исполнить произвольную команду или
прочитать произвольный файл на диске."""

import os
import subprocess
from pathlib import Path

from config import ALIASES, APPS, FILES_MAX_READ_BYTES, FILES_ROOT

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
    },
    {
        "type": "function",
        "function": {
            "name": "open_file",
            "description": (
                "Открыть файл в его приложении по умолчанию. Путь — относительно "
                "корня проекта, выход за его пределы запрещён и вернёт ошибку."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Относительный путь к файлу, например 'main.py'.",
                    }
                },
                "required": ["path"],
            },
        },
    },
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


def _resolve_within_root(rel_path: str) -> Path | None:
    """Разрешает путь относительно FILES_ROOT и проверяет, что он не вышел
    за его пределы (защита от '../../secrets.txt' и абсолютных путей)."""
    try:
        candidate = (FILES_ROOT / rel_path).resolve()
    except (OSError, ValueError):
        return None
    if candidate != FILES_ROOT and FILES_ROOT not in candidate.parents:
        return None
    return candidate


def list_dir(rel_path: str = "") -> list[dict] | dict:
    """GUI-инструмент (не в TOOL_SCHEMA): список файлов/директорий внутри
    белого списка. Возвращает {'error': ...} при выходе за пределы root."""
    target = _resolve_within_root(rel_path)
    if target is None:
        return {"error": "Доступ запрещён: путь вне разрешённой директории."}
    if not target.is_dir():
        return {"error": f"'{rel_path}' — не директория или не найдена."}

    entries = []
    for p in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        entries.append({
            "name": p.name,
            "is_dir": p.is_dir(),
            "size": p.stat().st_size if p.is_file() else None,
        })
    return entries


def read_file(rel_path: str) -> str:
    """GUI-инструмент (не в TOOL_SCHEMA): текстовое содержимое файла внутри
    белого списка, обрезанное до FILES_MAX_READ_BYTES."""
    target = _resolve_within_root(rel_path)
    if target is None:
        return "Доступ запрещён: путь вне разрешённой директории."
    if not target.is_file():
        return f"Файл '{rel_path}' не найден."

    data = target.read_bytes()[:FILES_MAX_READ_BYTES]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return "<бинарный файл, предпросмотр недоступен>"


def open_file(path: str) -> str:
    target = _resolve_within_root(path)
    if target is None:
        return "Доступ запрещён: путь вне разрешённой директории."
    if not target.exists():
        return f"Файл '{path}' не найден."
    try:
        os.startfile(target)
        return f"Открыл {path}."
    except Exception as e:
        return f"Не удалось открыть {path}: {e}"


def lock_screen() -> str:
    """GUI-инструмент (не в TOOL_SCHEMA — риск случайного голосового триггера).
    Требует явного подтверждения на стороне вызывающего (GUI)."""
    try:
        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=True)
        return "Экран заблокирован."
    except Exception as e:
        return f"Не удалось заблокировать экран: {e}"


def sleep_system() -> str:
    """GUI-инструмент (не в TOOL_SCHEMA — риск случайного голосового триггера).
    Требует явного подтверждения на стороне вызывающего (GUI)."""
    try:
        subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=True)
        return "Ушёл в спящий режим."
    except Exception as e:
        return f"Не удалось перейти в спящий режим: {e}"


def execute_tool_call(name: str, arguments: dict) -> str:
    if name == "open_app":
        return open_app(arguments.get("app", ""))
    if name == "open_file":
        return open_file(arguments.get("path", ""))
    return f"Неизвестный инструмент: {name}"
