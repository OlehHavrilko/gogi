"""TUI на Textual — терминальный интерфейс поверх Assistant (тот же
бэкенд, что у main.py и gui.py). Не desktop-приложение с инсталлятором,
а нечто в духе терминальных клиентов (как сам Claude Code): три вкладки
в одном окне терминала, без браузера и без отдельного процесса-обёртки.

Тяжёлая загрузка STT/TTS уходит в фоновый поток, чтобы не блокировать
event loop Textual во время старта."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    TabbedContent,
    TabPane,
)

from assistant import Assistant
from config import APPS
from tools import list_dir, lock_screen, open_app, open_file, sleep_system

_APP_NAMES = list(APPS.keys())


class ConfirmScreen(ModalScreen[bool]):
    """Модалка подтверждения — для 'Заблокировать экран'/'Спящий режим',
    случайно нажать которые в терминале не должно быть так же легко, как
    Enter по обычному пункту списка (та же логика, что и confirm() в GUI)."""

    CSS = """
    ConfirmScreen { align: center middle; }
    #dialog { width: 46; height: auto; border: round $warning; padding: 1 2; background: $surface; }
    #buttons { height: 3; align: center middle; }
    #buttons Button { margin: 0 1; }
    """

    def __init__(self, question: str):
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.question)
            with Horizontal(id="buttons"):
                yield Button("Да", id="yes", variant="error")
                yield Button("Отмена", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class GogiApp(App):
    TITLE = "Гоги"
    CSS = """
    Screen { background: $surface; }
    .section-label { color: $warning; text-style: bold; margin: 1 0 0 1; }
    #transcript { border: round $warning-darken-2; height: 1fr; margin: 1; }
    #text-input { margin: 0 1 1 1; }
    ListView { height: auto; max-height: 14; margin: 0 1 1 1; }
    #file-path { color: $text-muted; margin: 0 0 0 1; }
    """

    BINDINGS = [
        ("r", "toggle_recording", "Запись"),
        ("q", "quit", "Выход"),
    ]

    def __init__(self):
        super().__init__()
        self.assistant: Assistant | None = None
        self.recording = False
        self.file_path = ""
        self._file_entries: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="tab-main"):
            with TabPane("Главный", id="tab-main"):
                yield RichLog(id="transcript", wrap=True, markup=True)
                yield Input(
                    placeholder="Напиши сообщение (или R — запись с микрофона)...",
                    id="text-input",
                )
            with TabPane("Команды", id="tab-commands"):
                yield Label("ОТКРЫТЬ ПРИЛОЖЕНИЕ", classes="section-label")
                yield ListView(id="app-list")
                yield Label("СИСТЕМА", classes="section-label")
                yield ListView(id="sys-list")
                yield Label("ФАЙЛОВАЯ СИСТЕМА · ДОСТУП ПО БЕЛОМУ СПИСКУ", classes="section-label")
                yield Label("/", id="file-path")
                yield ListView(id="file-list")
            with TabPane("Настройки", id="tab-settings"):
                yield Label("ГОЛОСОВЫЕ ПРОФИЛИ", classes="section-label")
                yield ListView(id="voice-list")
                yield Label("МОДЕЛЬ LLM", classes="section-label")
                yield ListView(id="model-list")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#transcript", RichLog).write(
            "Загружаю STT и TTS (может занять до минуты)..."
        )
        self._populate_apps()
        self._populate_sys_actions()
        self._refresh_file_list()
        self.run_worker(self._load_assistant_blocking, thread=True)

    # --- загрузка ------------------------------------------------------------

    def _load_assistant_blocking(self) -> None:
        try:
            self.assistant = Assistant()
        except Exception as e:
            self.call_from_thread(self._on_load_error, str(e))
            return
        self.call_from_thread(self._on_loaded)

    def _status_line(self) -> str:
        return f"модель: {self.assistant.model} · голос: {self.assistant.tts.voice_id}"

    def _on_loaded(self) -> None:
        log = self.query_one("#transcript", RichLog)
        log.clear()
        log.write("[b]Гоги готов.[/b] Нажми R, чтобы говорить, или напиши текстом ниже.")
        self.sub_title = self._status_line()
        self._populate_voices()
        self._populate_models()

    def _on_load_error(self, message: str) -> None:
        self.query_one("#transcript", RichLog).write(f"[red]Ошибка загрузки: {message}[/red]")

    # --- диалог -----------------------------------------------------------

    def action_toggle_recording(self) -> None:
        if self.assistant is None:
            return
        log = self.query_one("#transcript", RichLog)
        if not self.recording:
            self.recording = True
            self.assistant.tts.interrupt()  # barge-in
            self.assistant.stt.start_recording()
            log.write("[i]Слушаю... нажми R ещё раз, чтобы остановить.[/i]")
        else:
            self.recording = False
            log.write("[i]Распознаю и думаю...[/i]")
            self.run_worker(self._respond_to_recording, thread=True)

    def _respond_to_recording(self) -> None:
        audio = self.assistant.stt.stop_recording()
        text = self.assistant.transcribe(audio)
        if not text:
            self.call_from_thread(
                lambda: self.query_one("#transcript", RichLog).write("[dim](не расслышал)[/dim]")
            )
            return
        reply = self.assistant.respond(text)
        self.call_from_thread(self._append_exchange, text, reply)

    def _append_exchange(self, user_text: str, reply: str) -> None:
        log = self.query_one("#transcript", RichLog)
        log.write(f"[bold]Вы:[/bold] {user_text}")
        log.write(f"[bold $warning]Гоги:[/bold $warning] {reply or '(нет ответа)'}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "text-input" or self.assistant is None:
            return
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self.query_one("#transcript", RichLog).write(f"[bold]Вы:[/bold] {text}")
        self.run_worker(lambda: self._respond_to_text(text), thread=True)

    def _respond_to_text(self, text: str) -> None:
        reply = self.assistant.respond(text)
        self.call_from_thread(
            lambda: self.query_one("#transcript", RichLog).write(
                f"[bold $warning]Гоги:[/bold $warning] {reply or '(нет ответа)'}"
            )
        )

    # --- команды: приложения -----------------------------------------------------

    def _populate_apps(self) -> None:
        view = self.query_one("#app-list", ListView)
        for name in _APP_NAMES:
            view.append(ListItem(Label(name), id=f"app-{name}"))

    def _populate_sys_actions(self) -> None:
        view = self.query_one("#sys-list", ListView)
        view.append(ListItem(Label("Диспетчер задач"), id="sys-task_manager"))
        view.append(ListItem(Label("Настройки Windows"), id="sys-settings"))
        view.append(ListItem(Label("⚠ Заблокировать экран"), id="sys-lock"))
        view.append(ListItem(Label("⚠ Спящий режим"), id="sys-sleep"))

    # --- команды: файлы -----------------------------------------------------------

    def _refresh_file_list(self) -> None:
        # ID виджетов Textual разрешают только буквы/цифры/подчёркивание/дефис —
        # имена файлов (точки, юникод, пробелы) под это не подходят, поэтому
        # ID строится по индексу в self._file_entries, а не по имени файла.
        self.query_one("#file-path", Label).update("/" + self.file_path)
        view = self.query_one("#file-list", ListView)
        view.clear()

        result = list_dir(self.file_path)
        if isinstance(result, dict):
            self._file_entries = []
            view.append(ListItem(Label(f"[red]{result.get('error')}[/red]")))
            return

        self._file_entries = result
        if self.file_path:
            view.append(ListItem(Label(".. (наверх)"), id="file-up"))
        for i, entry in enumerate(result):
            icon = "📁" if entry["is_dir"] else "📄"
            view.append(ListItem(Label(f"{icon} {entry['name']}"), id=f"file-{i}"))

    # --- настройки: голоса и модели -------------------------------------------------

    def _populate_voices(self) -> None:
        view = self.query_one("#voice-list", ListView)
        for vid, name in self.assistant.available_voices().items():
            mark = " [green](активен)[/green]" if vid == self.assistant.tts.voice_id else ""
            view.append(ListItem(Label(f"{name}{mark}"), id=f"voice-{vid}"))

    def _populate_models(self) -> None:
        view = self.query_one("#model-list", ListView)
        for model in self.assistant.available_models():
            mark = " [green](загружена)[/green]" if model == self.assistant.model else ""
            view.append(ListItem(Label(f"{model}{mark}"), id=f"model-{model.replace(':', '_')}"))

    # --- обработка выбора в списках -------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""

        if item_id.startswith("app-"):
            open_app(item_id.removeprefix("app-"))

        elif item_id == "sys-task_manager":
            open_app("task_manager")
        elif item_id == "sys-settings":
            open_app("settings")
        elif item_id == "sys-lock":
            self.push_screen(ConfirmScreen("Заблокировать экран?"), self._on_lock_confirmed)
        elif item_id == "sys-sleep":
            self.push_screen(ConfirmScreen("Уйти в спящий режим?"), self._on_sleep_confirmed)

        elif item_id == "file-up":
            parts = self.file_path.split("/")
            parts.pop()
            self.file_path = "/".join(p for p in parts if p)
            self._refresh_file_list()
        elif item_id.startswith("file-"):
            entry = self._file_entries[int(item_id.removeprefix("file-"))]
            name = entry["name"]
            next_path = f"{self.file_path}/{name}" if self.file_path else name
            if entry["is_dir"]:
                self.file_path = next_path
                self._refresh_file_list()
            else:
                open_file(next_path)

        elif item_id.startswith("voice-") and self.assistant is not None:
            self.assistant.switch_voice(item_id.removeprefix("voice-"))
            self.query_one("#voice-list", ListView).clear()
            self._populate_voices()
            self.sub_title = self._status_line()
        elif item_id.startswith("model-") and self.assistant is not None:
            wanted = item_id.removeprefix("model-")
            model = next(
                (m for m in self.assistant.available_models() if m.replace(":", "_") == wanted),
                None,
            )
            if model:
                self.assistant.switch_model(model)
                self.query_one("#model-list", ListView).clear()
                self._populate_models()
                self.sub_title = self._status_line()

    def _on_lock_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            lock_screen()

    def _on_sleep_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            sleep_system()


if __name__ == "__main__":
    GogiApp().run()
