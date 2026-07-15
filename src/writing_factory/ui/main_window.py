"""Quiet operational PyQt6 shell for project workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from writing_factory.ui.author_chat_page import AuthorChatPage
from writing_factory.ui.branding import APP_WINDOW_TITLE, application_icon
from writing_factory.ui.help_ui import TutorialPage
from writing_factory.ui.knowledge_page import KnowledgeBasePage
from writing_factory.ui.persona_editor import (
    PersonaLoader,
    PersonaSaver,
    PersonaVersionLoader,
    RuntimePersonaLoader,
)
from writing_factory.ui.persona_page import PersonaPage
from writing_factory.ui.project_page import ProjectPage
from writing_factory.ui.settings_dialogs import SettingsDialogBackend
from writing_factory.ui.settings_page import SettingsPage
from writing_factory.ui.workers import BackgroundTaskManager, TaskContext
from writing_factory.ui.writing_task_page import WritingTaskPage


class MainWindow(QMainWindow):
    """Application shell with non-blocking provider diagnostics."""

    def __init__(
        self,
        siliconflow_check: Callable[[], Any],
        *,
        ingest_document: Callable[[Path, TaskContext], Any] | None = None,
        list_documents: Callable[[], list[dict[str, object]]] | None = None,
        delete_documents: Callable[[set[str], TaskContext], Any] | None = None,
        distill_persona: Callable[[str, str, set[str], set[str], str, TaskContext], Any]
        | None = None,
        evaluate_persona: Callable[[str, TaskContext], Any] | None = None,
        list_personas: Callable[[], list[dict[str, object]]] | None = None,
        delete_personas: Callable[[set[str], TaskContext], Any] | None = None,
        load_persona: PersonaLoader | None = None,
        save_persona: PersonaSaver | None = None,
        load_runtime_persona: RuntimePersonaLoader | None = None,
        list_persona_versions: PersonaVersionLoader | None = None,
        get_siliconflow_concurrency: Callable[[], int] | None = None,
        set_siliconflow_concurrency: Callable[[int], None] | None = None,
        get_siliconflow_request_timeout: Callable[[], int] | None = None,
        set_siliconflow_request_timeout: Callable[[int], None] | None = None,
        get_retrieval_option: Callable[[str, bool], bool] | None = None,
        set_retrieval_option: Callable[[str, bool], None] | None = None,
        retrieve: Callable[..., Any] | None = None,
        run_writing_pipeline: Callable[..., Any] | None = None,
        evaluate_generation: Callable[..., Any] | None = None,
        list_projects: Callable[[], list[dict[str, object]]] | None = None,
        create_project: Callable[[str, str], str] | None = None,
        update_project: Callable[[str, str, str], None] | None = None,
        delete_projects: Callable[[set[str]], int] | None = None,
        create_writing_task: Callable[..., str] | None = None,
        list_writing_tasks: Callable[[str], list[dict[str, object]]] | None = None,
        load_writing_task: Callable[[str], dict[str, object] | None] | None = None,
        save_edited_draft: Callable[..., None] | None = None,
        delete_writing_tasks: Callable[[set[str]], int] | None = None,
        preview_source_selection: Callable[[str, set[str], set[str]], dict[str, int]]
        | None = None,
        list_chat_conversations: Callable[[], list[dict[str, object]]] | None = None,
        load_chat_conversation: Callable[[str], Any] | None = None,
        create_chat_conversation: Callable[..., str] | None = None,
        rename_chat_conversation: Callable[[str, str], None] | None = None,
        delete_chat_conversations: Callable[[set[str]], int] | None = None,
        list_chat_messages: Callable[[str], list[Any]] | None = None,
        send_chat_message: Callable[..., Any] | None = None,
        verify_chat_message: Callable[..., Any] | None = None,
        get_author_chat_recent_rounds: Callable[[], int] | None = None,
        set_author_chat_recent_rounds: Callable[[int], None] | None = None,
        settings_backend: SettingsDialogBackend | None = None,
    ) -> None:
        super().__init__()
        self._siliconflow_check = siliconflow_check
        self._ingest_document = ingest_document
        self._list_documents = list_documents or (lambda: [])
        self._delete_documents = delete_documents
        self._distill_persona = distill_persona
        self._evaluate_persona = evaluate_persona
        self._list_personas = list_personas or (lambda: [])
        self._delete_personas = delete_personas
        self._load_persona = load_persona
        self._save_persona = save_persona
        self._load_runtime_persona = load_runtime_persona
        self._list_persona_versions = list_persona_versions
        self._get_siliconflow_concurrency = get_siliconflow_concurrency or (lambda: 3)
        self._set_siliconflow_concurrency = set_siliconflow_concurrency
        self._get_siliconflow_request_timeout = get_siliconflow_request_timeout or (lambda: 900)
        self._set_siliconflow_request_timeout = set_siliconflow_request_timeout
        self._get_retrieval_option = get_retrieval_option or (lambda _k, d=True: d)
        self._set_retrieval_option = set_retrieval_option
        self._retrieve = retrieve
        self._run_writing_pipeline = run_writing_pipeline
        self._evaluate_generation = evaluate_generation
        self._list_projects = list_projects or (lambda: [])
        self._create_project = create_project
        self._update_project = update_project
        self._delete_projects = delete_projects
        self._create_writing_task = create_writing_task
        self._list_writing_tasks = list_writing_tasks or (lambda _project_id: [])
        self._load_writing_task = load_writing_task
        self._save_edited_draft = save_edited_draft
        self._delete_writing_tasks = delete_writing_tasks
        self._preview_source_selection = preview_source_selection
        self._list_chat_conversations = list_chat_conversations or (lambda: [])
        self._load_chat_conversation = load_chat_conversation or (lambda _identifier: None)
        self._create_chat_conversation = create_chat_conversation
        self._rename_chat_conversation = rename_chat_conversation
        self._delete_chat_conversations = delete_chat_conversations
        self._list_chat_messages = list_chat_messages or (lambda _identifier: [])
        self._send_chat_message = send_chat_message
        self._verify_chat_message = verify_chat_message
        self._get_author_chat_recent_rounds = get_author_chat_recent_rounds or (lambda: 6)
        self._set_author_chat_recent_rounds = set_author_chat_recent_rounds
        self._settings_backend = settings_backend
        self._tasks = BackgroundTaskManager(self)
        self._close_pending = False
        self._tasks.task_finished.connect(self._task_finished)

        self.setWindowTitle(APP_WINDOW_TITLE)
        self.setWindowIcon(application_icon())
        self.setMinimumSize(960, 640)
        self.resize(1120, 720)
        self.setStyleSheet(_STYLESHEET)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.setFixedWidth(184)
        self.navigation.setIconSize(self.navigation.iconSize())
        nav_items = (
            ("项目", QStyle.StandardPixmap.SP_DirIcon),
            ("知识库", QStyle.StandardPixmap.SP_FileDialogDetailedView),
            ("作者档案", QStyle.StandardPixmap.SP_FileIcon),
            ("作者对话", QStyle.StandardPixmap.SP_MessageBoxInformation),
            ("写作任务", QStyle.StandardPixmap.SP_CommandLink),
            ("设置", QStyle.StandardPixmap.SP_ComputerIcon),
            ("教程", QStyle.StandardPixmap.SP_DialogHelpButton),
        )
        for label, icon_type in nav_items:
            item = QListWidgetItem(self.style().standardIcon(icon_type), label)
            item.setSizeHint(QSize(160, 44))
            self.navigation.addItem(item)

        self.pages = QStackedWidget()
        self.project_page = ProjectPage(
            list_projects=self._list_projects,
            create_project=self._create_project or (lambda _title, _description: ""),
            update_project=self._update_project or (lambda _project_id, _title, _description: None),
            delete_projects=self._delete_projects or (lambda _identifiers: 0),
            show_message=self.statusBar().showMessage,
        )
        self.pages.addWidget(self.project_page)
        self.knowledge_page = KnowledgeBasePage(
            self._tasks,
            ingest_document=self._ingest_document,
            list_documents=self._list_documents,
            delete_documents=self._delete_documents,
            retrieve=self._retrieve,
            get_retrieval_option=self._get_retrieval_option,
            show_message=self.statusBar().showMessage,
        )
        self.pages.addWidget(self.knowledge_page)
        self.persona_page = PersonaPage(
            self._tasks,
            distill_persona=self._distill_persona,
            evaluate_persona=self._evaluate_persona,
            list_sources=self._list_documents,
            list_personas=self._list_personas,
            delete_personas=self._delete_personas,
            load_persona=self._load_persona,
            save_persona=self._save_persona,
            load_runtime_persona=self._load_runtime_persona,
            list_persona_versions=self._list_persona_versions,
            show_message=self.statusBar().showMessage,
        )
        self.pages.addWidget(self.persona_page)
        self.knowledge_page.documents_changed.connect(self.persona_page.refresh)
        self.author_chat_page = AuthorChatPage(
            self._tasks,
            list_personas=self._list_personas,
            list_documents=self._list_documents,
            list_conversations=self._list_chat_conversations,
            load_conversation=self._load_chat_conversation,
            create_conversation=self._create_chat_conversation,
            rename_conversation=self._rename_chat_conversation,
            delete_conversations=self._delete_chat_conversations,
            list_messages=self._list_chat_messages,
            send_message=self._send_chat_message,
            verify_message=self._verify_chat_message,
            show_message=self.statusBar().showMessage,
        )
        self.pages.addWidget(self.author_chat_page)
        self.knowledge_page.documents_changed.connect(self.author_chat_page.refresh)
        self.persona_page.personas_changed.connect(self.author_chat_page.refresh)
        self.writing_task_page = WritingTaskPage(
            self._tasks,
            list_personas=self._list_personas,
            list_projects=self._list_projects,
            list_documents=self._list_documents,
            run_writing_pipeline=self._run_writing_pipeline,
            evaluate_generation=self._evaluate_generation,
            create_writing_task=self._create_writing_task,
            list_writing_tasks=self._list_writing_tasks,
            load_writing_task=self._load_writing_task,
            save_edited_draft=self._save_edited_draft,
            delete_writing_tasks=self._delete_writing_tasks,
            preview_source_selection=self._preview_source_selection,
            show_message=self.statusBar().showMessage,
        )
        self.pages.addWidget(self.writing_task_page)
        self.project_page.projects_changed.connect(self.writing_task_page.refresh_projects)
        self.settings_page = SettingsPage(
            self._tasks,
            backend=self._settings_backend,
            siliconflow_check=self._siliconflow_check,
            get_concurrency=self._get_siliconflow_concurrency,
            set_concurrency=self._set_siliconflow_concurrency,
            get_timeout=self._get_siliconflow_request_timeout,
            set_timeout=self._set_siliconflow_request_timeout,
            get_chat_recent_rounds=self._get_author_chat_recent_rounds,
            set_chat_recent_rounds=self._set_author_chat_recent_rounds,
            show_message=self.statusBar().showMessage,
        )
        self.pages.addWidget(self.settings_page)
        self.tutorial_page = TutorialPage()
        self.pages.addWidget(self.tutorial_page)
        # Keep stable handles used by existing UI tests and integrations.
        self.siliconflow_status = self.settings_page.siliconflow_status
        self.check_button = self.settings_page.check_button
        self.concurrency_input = self.settings_page.concurrency_input
        self.siliconflow_timeout_input = self.settings_page.timeout_input
        self.navigation.currentRowChanged.connect(self._switch_page)
        self.navigation.setCurrentRow(1)

        root_layout.addWidget(self.navigation)
        root_layout.addWidget(self.pages, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("就绪")

    def _switch_page(self, index: int) -> None:
        """Switch pages and refresh views backed by mutable local storage."""

        self.pages.setCurrentIndex(index)
        if index == 0:
            self.project_page.refresh()
        elif index == 1:
            self.knowledge_page.refresh_documents()
        elif index == 2:
            self.persona_page.refresh()
        elif index == 3:
            self.author_chat_page.refresh()
        elif index == 4:
            self.writing_task_page.refresh()
        elif index == 5:
            self.settings_page.refresh()

    def _empty_page(self, title: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 28, 32, 28)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        layout.addStretch(1)
        return page

    def closeEvent(self, event: QCloseEvent) -> None:
        """Request cooperative task cancellation before closing."""

        if self._tasks.active_count:
            self._close_pending = True
            self._tasks.cancel_all()
            self.statusBar().showMessage("正在结束后台任务")
            event.ignore()
            return
        self._tasks.cancel_all()
        super().closeEvent(event)

    def _task_finished(self, _task_id: str) -> None:
        if self._close_pending and self._tasks.active_count == 0:
            self._close_pending = False
            self.close()


_STYLESHEET = """
QMainWindow, QWidget {
    background: #f7f8fa;
    color: #1f2933;
    font-size: 14px;
}
#navigation {
    background: #20262e;
    color: #dfe4ea;
    border: 0;
    padding: 18px 10px;
    outline: 0;
}
#navigation::item {
    min-height: 42px;
    padding: 0 12px;
    margin: 2px 0;
    border-radius: 5px;
}
#navigation::item:selected {
    background: #3a4653;
    color: #ffffff;
}
#navigation::item:hover:!selected {
    background: #2b343e;
}
#pageTitle {
    font-size: 24px;
    font-weight: 700;
    color: #101820;
}
#sectionTitle {
    font-size: 15px;
    font-weight: 600;
    color: #485564;
}
#serviceRow {
    background: #ffffff;
    border: 1px solid #dce2e8;
    border-radius: 6px;
    min-height: 58px;
}
#providerName {
    font-weight: 600;
    min-width: 112px;
}
#mutedText {
    color: #6b7785;
}
#statusNeutral, #statusBusy, #statusReady, #statusError {
    min-width: 72px;
    font-weight: 600;
}
#statusNeutral { color: #6b7785; }
#statusBusy { color: #a06400; }
#statusReady { color: #17705b; }
#statusError { color: #b42318; }
QPushButton {
    background: #ffffff;
    border: 1px solid #aeb8c2;
    border-radius: 5px;
    padding: 7px 13px;
    min-height: 20px;
}
QPushButton:hover { background: #eef2f5; }
QPushButton:pressed { background: #e2e8ed; }
QPushButton:disabled { color: #9ca5ae; background: #f2f4f6; }
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #dce2e8;
    color: #55616e;
}
QTableWidget {
    background: #ffffff;
    alternate-background-color: #f4f6f8;
    border: 1px solid #dce2e8;
    border-radius: 5px;
    gridline-color: #e7ebef;
}
QHeaderView::section {
    background: #edf1f4;
    color: #485564;
    border: 0;
    border-bottom: 1px solid #d4dbe2;
    padding: 8px 10px;
    font-weight: 600;
}
QTableWidget::item {
    padding: 8px;
}
QProgressBar {
    background: #e7ebef;
    border: 0;
    border-radius: 4px;
    color: #28323c;
    text-align: center;
}
QProgressBar::chunk {
    background: #27806b;
    border-radius: 4px;
}
#modeButton {
    min-height: 20px;
    padding: 7px 8px;
}
#modeButton:checked {
    background: #27806b;
    border-color: #1f6a59;
    color: #ffffff;
}
"""
