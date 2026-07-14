"""Quiet operational PyQt6 shell for project workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from writing_factory.llm.models import ChatResult
from writing_factory.ui.knowledge_page import KnowledgeBasePage
from writing_factory.ui.persona_editor import (
    PersonaLoader,
    PersonaSaver,
    PersonaVersionLoader,
    RuntimePersonaLoader,
)
from writing_factory.ui.persona_page import PersonaPage
from writing_factory.ui.project_page import ProjectPage
from writing_factory.ui.workers import BackgroundTaskManager, TaskContext
from writing_factory.ui.writing_task_page import WritingTaskPage


class MainWindow(QMainWindow):
    """Application shell with non-blocking provider diagnostics."""

    def __init__(
        self,
        siliconflow_check: Callable[[], ChatResult],
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
        get_framework_generation_timeout: Callable[[], int] | None = None,
        set_framework_generation_timeout: Callable[[int], None] | None = None,
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
        self._get_framework_generation_timeout = get_framework_generation_timeout or (
            lambda: 900
        )
        self._set_framework_generation_timeout = set_framework_generation_timeout
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
        self._tasks = BackgroundTaskManager(self)
        self._check_task_id: str | None = None
        self._close_pending = False
        self._tasks.task_finished.connect(self._task_finished)

        self.setWindowTitle("写作工厂")
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
            ("写作任务", QStyle.StandardPixmap.SP_CommandLink),
            ("设置", QStyle.StandardPixmap.SP_ComputerIcon),
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
        self.pages.addWidget(self._settings_page())
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
            self.writing_task_page.refresh()

    def _empty_page(self, title: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 28, 32, 28)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        layout.addStretch(1)
        return page

    def _settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(20)
        heading = QLabel("设置")
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)

        section_title = QLabel("外部服务")
        section_title.setObjectName("sectionTitle")
        layout.addWidget(section_title)

        siliconflow_row = QFrame()
        siliconflow_row.setObjectName("serviceRow")
        row_layout = QHBoxLayout(siliconflow_row)
        row_layout.setContentsMargins(18, 14, 14, 14)
        row_layout.setSpacing(16)
        provider = QLabel("SiliconFlow")
        provider.setObjectName("providerName")
        model = QLabel("DeepSeek-V4-Flash")
        model.setObjectName("mutedText")
        self.siliconflow_status = QLabel("未检测")
        self.siliconflow_status.setObjectName("statusNeutral")
        self.check_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "检测"
        )
        self.check_button.setToolTip("检测 SiliconFlow 连接")
        self.check_button.clicked.connect(self._start_siliconflow_check)
        row_layout.addWidget(provider)
        row_layout.addWidget(model)
        row_layout.addStretch(1)
        row_layout.addWidget(self.siliconflow_status)
        row_layout.addWidget(self.check_button)
        layout.addWidget(siliconflow_row)

        concurrency_row = QFrame()
        concurrency_row.setObjectName("serviceRow")
        concurrency_layout = QHBoxLayout(concurrency_row)
        concurrency_layout.setContentsMargins(18, 14, 14, 14)
        concurrency_layout.setSpacing(16)
        concurrency_label = QLabel("最大并发数")
        concurrency_label.setObjectName("providerName")
        concurrency_provider = QLabel("SiliconFlow 全部请求")
        concurrency_provider.setObjectName("mutedText")
        self.concurrency_input = QSpinBox()
        self.concurrency_input.setRange(1, 8)
        self.concurrency_input.setValue(self._get_siliconflow_concurrency())
        self.concurrency_input.setToolTip("限制整个程序同时进行的 SiliconFlow 请求数")
        self.concurrency_input.valueChanged.connect(self._concurrency_changed)
        concurrency_layout.addWidget(concurrency_label)
        concurrency_layout.addWidget(concurrency_provider)
        concurrency_layout.addStretch(1)
        concurrency_layout.addWidget(self.concurrency_input)
        layout.addWidget(concurrency_row)

        timeout_row = QFrame()
        timeout_row.setObjectName("serviceRow")
        timeout_layout = QHBoxLayout(timeout_row)
        timeout_layout.setContentsMargins(18, 14, 14, 14)
        timeout_layout.setSpacing(16)
        timeout_label = QLabel("框架生成超时上限")
        timeout_label.setObjectName("providerName")
        timeout_description = QLabel("SiliconFlow 请求总等待窗口（含重试）")
        timeout_description.setObjectName("mutedText")
        self.framework_timeout_input = QSpinBox()
        self.framework_timeout_input.setRange(60, 3600)
        self.framework_timeout_input.setSingleStep(60)
        self.framework_timeout_input.setSuffix(" 秒")
        self.framework_timeout_input.setValue(self._get_framework_generation_timeout())
        self.framework_timeout_input.setToolTip(
            "仅用于论文框架生成；超时后任务停在框架节点，可从断点继续"
        )
        self.framework_timeout_input.valueChanged.connect(self._framework_timeout_changed)
        timeout_layout.addWidget(timeout_label)
        timeout_layout.addWidget(timeout_description)
        timeout_layout.addStretch(1)
        timeout_layout.addWidget(self.framework_timeout_input)
        layout.addWidget(timeout_row)

        mineru_row = QFrame()
        mineru_row.setObjectName("serviceRow")
        mineru_layout = QHBoxLayout(mineru_row)
        mineru_layout.setContentsMargins(18, 14, 14, 14)
        mineru_layout.setSpacing(16)
        mineru_name = QLabel("MinerU")
        mineru_name.setObjectName("providerName")
        mineru_endpoint = QLabel("API v4")
        mineru_endpoint.setObjectName("mutedText")
        mineru_status = QLabel("凭据已加载")
        mineru_status.setObjectName("statusReady")
        mineru_layout.addWidget(mineru_name)
        mineru_layout.addWidget(mineru_endpoint)
        mineru_layout.addStretch(1)
        mineru_layout.addWidget(mineru_status)
        layout.addWidget(mineru_row)

        retrieval_title = QLabel("检索增强")
        retrieval_title.setObjectName("sectionTitle")
        layout.addWidget(retrieval_title)

        self.hyde_checkbox = QCheckBox("HyDE 检索")
        self.hyde_checkbox.setChecked(self._get_retrieval_option("use_hyde", True))
        self.hyde_checkbox.setToolTip(
            "先让模型写一段假设性答案，用其向量检索，通常显著提升学术查询召回"
        )
        self.hyde_checkbox.stateChanged.connect(
            lambda state: self._retrieval_option_changed("use_hyde", state)
        )
        layout.addWidget(self.hyde_checkbox)

        self.rewrite_checkbox = QCheckBox("查询改写")
        self.rewrite_checkbox.setChecked(self._get_retrieval_option("use_rewrite", True))
        self.rewrite_checkbox.setToolTip("把一个抽象问题扩展为 3-5 个具体子查询分别检索后融合")
        self.rewrite_checkbox.stateChanged.connect(
            lambda state: self._retrieval_option_changed("use_rewrite", state)
        )
        layout.addWidget(self.rewrite_checkbox)

        layout.addStretch(1)
        return page

    def _retrieval_option_changed(self, key: str, state: int) -> None:
        """持久化检索增强开关。"""

        if self._set_retrieval_option is None:
            return
        enabled = bool(state)
        self._set_retrieval_option(key, enabled)
        self.statusBar().showMessage(f"检索选项「{key}」已{'开启' if enabled else '关闭'}", 4000)

    def _concurrency_changed(self, value: int) -> None:
        """运行时应用并持久化统一并发上限。"""

        if self._set_siliconflow_concurrency is None:
            return
        try:
            self._set_siliconflow_concurrency(value)
        except ValueError as exc:
            self.statusBar().showMessage(str(exc), 5000)
            return
        self.statusBar().showMessage(f"SiliconFlow 最大并发数已设为 {value}", 4000)

    def _framework_timeout_changed(self, value: int) -> None:
        """持久化框架生成请求的总超时上限。"""

        if self._set_framework_generation_timeout is None:
            return
        try:
            self._set_framework_generation_timeout(value)
        except ValueError as exc:
            self.statusBar().showMessage(str(exc), 5000)
            return
        self.statusBar().showMessage(f"框架生成超时上限已设为 {value} 秒", 4000)

    def _start_siliconflow_check(self) -> None:
        if self._check_task_id is not None:
            return
        self.check_button.setEnabled(False)
        self.siliconflow_status.setText("检测中")
        self.siliconflow_status.setObjectName("statusBusy")
        self.siliconflow_status.style().unpolish(self.siliconflow_status)
        self.siliconflow_status.style().polish(self.siliconflow_status)
        self.statusBar().showMessage("正在检测 SiliconFlow")

        def task(context: TaskContext) -> ChatResult:
            context.report_progress(20, "正在连接")
            result = self._siliconflow_check()
            context.report_progress(100, "完成")
            return result

        self._check_task_id = self._tasks.start(
            task,
            on_success=self._check_succeeded,
            on_error=self._check_failed,
            on_progress=self._check_progress,
        )

    def _check_succeeded(self, result: Any) -> None:
        chat = result if isinstance(result, ChatResult) else None
        tokens = chat.usage.total_tokens if chat is not None else 0
        self.siliconflow_status.setText("可用")
        self.siliconflow_status.setObjectName("statusReady")
        self.statusBar().showMessage(f"SiliconFlow 可用 · {tokens} tokens", 5000)
        self._finish_check()

    def _check_failed(self, message: str) -> None:
        self.siliconflow_status.setText("不可用")
        self.siliconflow_status.setObjectName("statusError")
        self.statusBar().showMessage(message, 8000)
        self._finish_check()

    def _check_progress(self, percent: int, message: str) -> None:
        if message:
            self.statusBar().showMessage(f"{message} · {percent}%")

    def _finish_check(self) -> None:
        self.siliconflow_status.style().unpolish(self.siliconflow_status)
        self.siliconflow_status.style().polish(self.siliconflow_status)
        self.check_button.setEnabled(True)
        self._check_task_id = None

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
    font-weight: 600;
    color: #18212b;
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
