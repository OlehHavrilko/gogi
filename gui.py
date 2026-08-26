"""GUI на pywebview: тонкая обёртка вокруг Assistant + инструментов из
tools.py, доступная фронтенду (gui/index.html) через window.pywebview.api.

Тяжёлая загрузка моделей (STT/TTS) не происходит в конструкторе Api —
она отложена в init(), которую фронтенд вызывает уже после отрисовки окна
и показа спиннера, чтобы GUI не казался зависшим на 10-30 секунд."""

from pathlib import Path

import webview

from assistant import Assistant
from tools import list_dir, lock_screen, open_app, open_file, read_file, sleep_system


class Api:
    def __init__(self):
        self.assistant: Assistant | None = None

    def init(self) -> dict:
        if self.assistant is None:
            self.assistant = Assistant()
        return {
            "voices": self.assistant.available_voices(),
            "models": self.assistant.available_models(),
            "current_voice": self.assistant.tts.voice_id,
            "current_model": self.assistant.model,
            "nfe_step": self.assistant.tts.nfe_step,
            "cfg_strength": self.assistant.tts.cfg_strength,
        }

    # --- диалог -----------------------------------------------------------

    def start_recording(self) -> bool:
        # barge-in: клик по орбу означает "я хочу говорить сейчас", даже
        # если ассистент ещё не договорил предыдущий ответ.
        self.assistant.tts.interrupt()
        self.assistant.stt.start_recording()
        return True

    def stop_recording_and_respond(self) -> dict:
        audio = self.assistant.stt.stop_recording()
        text = self.assistant.transcribe(audio)
        if not text:
            return {"user_text": "", "reply": ""}
        reply = self.assistant.respond(text)
        return {"user_text": text, "reply": reply}

    def send_text(self, text: str) -> dict:
        """Текстовый ввод — резервный путь без микрофона (тестирование,
        шумное окружение)."""
        reply = self.assistant.respond(text)
        return {"user_text": text, "reply": reply}

    # --- настройки ----------------------------------------------------------

    def switch_voice(self, voice_id: str) -> bool:
        self.assistant.switch_voice(voice_id)
        return True

    def switch_model(self, model: str) -> bool:
        self.assistant.switch_model(model)
        return True

    def set_tts_params(self, nfe_step: int, cfg_strength: float) -> bool:
        self.assistant.set_tts_params(nfe_step, cfg_strength)
        return True

    # --- команды / файлы (панель "Команды") --------------------------------

    def list_apps(self) -> list[str]:
        from config import APPS

        return list(APPS.keys())

    def open_app(self, app: str) -> str:
        return open_app(app)

    def list_dir(self, rel_path: str = "") -> list | dict:
        return list_dir(rel_path)

    def read_file(self, rel_path: str) -> str:
        return read_file(rel_path)

    def open_file(self, rel_path: str) -> str:
        return open_file(rel_path)

    def lock_screen(self) -> str:
        return lock_screen()

    def sleep_system(self) -> str:
        return sleep_system()


def main():
    api = Api()
    index_html = Path(__file__).parent / "gui" / "index.html"
    webview.create_window(
        "Гоги",
        str(index_html),
        js_api=api,
        width=1280,
        height=820,
        min_size=(960, 620),
        background_color="#0d1117",
    )
    webview.start()


if __name__ == "__main__":
    main()
