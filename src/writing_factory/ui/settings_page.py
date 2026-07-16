"""Scrollable application settings surface with task-specific dialogs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from writing_factory.llm.configuration import (
    STEP_DEFINITIONS,
    ChatStepDefinition,
)
from writing_factory.llm.models import ChatResult
from writing_factory.llm.settings_service import ModelKind, ProviderName
from writing_factory.ui.help_ui import create_help_button
from writing_factory.ui.settings_dialogs import (
    ProviderSettingsDialog,
    SettingsDialogBackend,
    StepSettingsDialog,
)
from writing_factory.ui.settings_memory_backend import InMemorySettingsBackend
from writing_factory.ui.settings_model_dialog import ModelSettingsDialog
from writing_factory.ui.workers import BackgroundTaskManager, TaskContext

_GROUPS = (
    ("distill", "蒸馏"),
    ("retrieval", "检索"),
    ("chat", "作者对话"),
    ("writing", "写作"),
    ("evaluation", "评估"),
)


class SettingsPage(QWidget):
    """Display resolved settings and open one focused editor per concern."""

    def __init__(
        self,
        tasks: BackgroundTaskManager,
        *,
        backend: SettingsDialogBackend | None,
        siliconflow_check: Callable[[], Any],
        get_concurrency: Callable[[], int],
        set_concurrency: Callable[[int], None] | None,
        get_timeout: Callable[[], int],
        set_timeout: Callable[[int], None] | None,
        get_total_timeout: Callable[[], int],
        set_total_timeout: Callable[[int], None] | None,
        get_stream_idle_timeout: Callable[[], int],
        set_stream_idle_timeout: Callable[[int], None] | None,
        get_chat_recent_rounds: Callable[[], int],
        set_chat_recent_rounds: Callable[[int], None] | None,
        get_web_search_result_count: Callable[[], int],
        set_web_search_result_count: Callable[[int], None] | None,
        show_message: Callable[[str, int], None],
    ) -> None:
        super().__init__()
        self.tasks = tasks
        self.backend = backend or InMemorySettingsBackend()
        self.siliconflow_check = siliconflow_check
        self.get_concurrency = get_concurrency
        self.set_concurrency = set_concurrency
        self.get_timeout = get_timeout
        self.set_timeout = set_timeout
        self.get_total_timeout = get_total_timeout
        self.set_total_timeout = set_total_timeout
        self.get_stream_idle_timeout = get_stream_idle_timeout
        self.set_stream_idle_timeout = set_stream_idle_timeout
        self.get_chat_recent_rounds = get_chat_recent_rounds
        self.set_chat_recent_rounds = set_chat_recent_rounds
        self.get_web_search_result_count = get_web_search_result_count
        self.set_web_search_result_count = set_web_search_result_count
        self.show_message = show_message
        self._check_task_id: str | None = None
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        heading = QLabel("设置")
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        provider_title_row = QHBoxLayout()
        self.provider_title = QLabel("外部服务凭据")
        self.provider_title.setObjectName("sectionTitle")
        provider_title_row.addWidget(self.provider_title)
        self.credentials_help_button = create_help_button("credentials", self)
        provider_title_row.addWidget(self.credentials_help_button)
        provider_title_row.addStretch(1)
        layout.addLayout(provider_title_row)
        self.siliconflow_status, self.check_button = self._add_provider_row(
            layout, "siliconflow", "SiliconFlow"
        )
        self.mineru_status, _mineru_button = self._add_provider_row(layout, "mineru", "MinerU")
        self.bocha_status, _bocha_button = self._add_provider_row(layout, "bocha", "博查搜索")

        models_title = QLabel("基础模型")
        models_title.setObjectName("sectionTitle")
        layout.addWidget(models_title)
        self.model_value_labels: dict[ModelKind, QLabel] = {}
        for kind, label in (
            ("embedding", "Embedding"),
            ("reranker", "Rerank"),
            ("chat", "文字生成"),
        ):
            self._add_model_row(layout, kind, label)

        runtime_title = QLabel("运行规则")
        runtime_title.setObjectName("sectionTitle")
        layout.addWidget(runtime_title)
        runtime_row = QFrame()
        runtime_row.setObjectName("serviceRow")
        runtime_layout = QGridLayout(runtime_row)
        runtime_layout.setContentsMargins(18, 12, 18, 12)
        runtime_layout.setHorizontalSpacing(10)
        runtime_layout.setVerticalSpacing(8)
        runtime_layout.addWidget(QLabel("最大并发数"), 0, 0)
        self.concurrency_input = QSpinBox()
        self.concurrency_input.setRange(1, 8)
        self.concurrency_input.setValue(self.get_concurrency())
        self.concurrency_input.valueChanged.connect(self._concurrency_changed)
        runtime_layout.addWidget(self.concurrency_input, 0, 1)
        runtime_layout.addWidget(QLabel("默认单次尝试上限"), 0, 2)
        self.timeout_input = QSpinBox()
        self.timeout_input.setRange(60, 3600)
        self.timeout_input.setSingleStep(60)
        self.timeout_input.setSuffix(" 秒")
        self.timeout_input.setValue(self.get_timeout())
        self.timeout_input.valueChanged.connect(self._timeout_changed)
        runtime_layout.addWidget(self.timeout_input, 0, 3)
        runtime_layout.addWidget(QLabel("默认整项调用总预算"), 1, 0)
        self.total_timeout_input = QSpinBox()
        self.total_timeout_input.setRange(60, 21600)
        self.total_timeout_input.setSingleStep(60)
        self.total_timeout_input.setSuffix(" 秒")
        self.total_timeout_input.setToolTip("只决定能否启动下一次尝试，不截断已经开始的请求")
        self.total_timeout_input.setValue(self.get_total_timeout())
        self.total_timeout_input.valueChanged.connect(self._total_timeout_changed)
        runtime_layout.addWidget(self.total_timeout_input, 1, 1)
        runtime_layout.addWidget(QLabel("流式空闲超时"), 1, 2)
        self.stream_idle_timeout_input = QSpinBox()
        self.stream_idle_timeout_input.setRange(30, 1800)
        self.stream_idle_timeout_input.setSingleStep(30)
        self.stream_idle_timeout_input.setSuffix(" 秒")
        self.stream_idle_timeout_input.setToolTip("连续这么久没有收到任何有效流式数据才视为停滞")
        self.stream_idle_timeout_input.setValue(self.get_stream_idle_timeout())
        self.stream_idle_timeout_input.valueChanged.connect(self._stream_idle_timeout_changed)
        runtime_layout.addWidget(self.stream_idle_timeout_input, 1, 3)
        runtime_layout.addWidget(QLabel("作者对话最近轮数"), 2, 0)
        self.chat_recent_rounds_input = QSpinBox()
        self.chat_recent_rounds_input.setRange(1, 20)
        self.chat_recent_rounds_input.setSuffix(" 轮")
        self.chat_recent_rounds_input.setValue(self.get_chat_recent_rounds())
        self.chat_recent_rounds_input.valueChanged.connect(self._chat_rounds_changed)
        runtime_layout.addWidget(self.chat_recent_rounds_input, 2, 1)
        runtime_layout.addWidget(QLabel("每次联网检索条目数"), 2, 2)
        self.web_result_count_input = QSpinBox()
        self.web_result_count_input.setRange(1, 20)
        self.web_result_count_input.setSuffix(" 条")
        self.web_result_count_input.setToolTip("作者对话和写作任务中的每一次博查请求共用此设置")
        self.web_result_count_input.setValue(self.get_web_search_result_count())
        self.web_result_count_input.valueChanged.connect(self._web_result_count_changed)
        runtime_layout.addWidget(self.web_result_count_input, 2, 3)
        runtime_layout.setColumnStretch(4, 1)
        layout.addWidget(runtime_row)

        steps_title = QLabel("文字生成步骤")
        steps_title.setObjectName("sectionTitle")
        layout.addWidget(steps_title)
        self.step_tabs = QTabWidget()
        self.step_tables: dict[str, QTableWidget] = {}
        for group, label in _GROUPS:
            table = self._step_table(group)
            self.step_tables[group] = table
            self.step_tabs.addTab(table, label)
        self.step_tabs.setMinimumHeight(330)
        layout.addWidget(self.step_tabs)
        layout.addStretch(1)

    def _add_provider_row(
        self,
        layout: QVBoxLayout,
        provider: ProviderName,
        display_name: str,
    ) -> tuple[QLabel, QPushButton]:
        row = QFrame()
        row.setObjectName("serviceRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(18, 12, 14, 12)
        name = QLabel(display_name)
        name.setObjectName("providerName")
        status = QLabel()
        status.setObjectName("mutedText")
        status.setWordWrap(True)
        status.setMinimumWidth(0)
        row_layout.addWidget(name)
        row_layout.addWidget(status)
        row_layout.addStretch(1)
        if provider == "siliconflow":
            check = QPushButton(
                self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "检测"
            )
            check.clicked.connect(self._start_siliconflow_check)
            row_layout.addWidget(check)
        else:
            check = QPushButton()
            check.hide()
        edit = QPushButton("编辑")
        edit.clicked.connect(lambda: self._edit_provider(provider))
        row_layout.addWidget(edit)
        layout.addWidget(row)
        return status, check

    def _add_model_row(
        self,
        layout: QVBoxLayout,
        kind: ModelKind,
        display_name: str,
    ) -> None:
        row = QFrame()
        row.setObjectName("serviceRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(18, 12, 14, 12)
        name = QLabel(display_name)
        name.setObjectName("providerName")
        value = QLabel()
        value.setObjectName("mutedText")
        value.setWordWrap(True)
        value.setMinimumWidth(0)
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.model_value_labels[kind] = value
        edit = QPushButton("配置")
        edit.clicked.connect(lambda: self._edit_model(kind))
        row_layout.addWidget(name)
        row_layout.addWidget(value, 1)
        row_layout.addWidget(edit)
        layout.addWidget(row)

    def _step_table(self, group: str) -> QTableWidget:
        definitions = [item for item in STEP_DEFINITIONS if item.group == group]
        table = QTableWidget(len(definitions), 3)
        table.setHorizontalHeaderLabels(
            (
                "步骤",
                "设置摘要\n温度｜思考｜强度｜上限｜请求｜重试｜单次｜总预算",
                "",
            )
        )
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setProperty("definitions", definitions)
        header = table.horizontalHeader()
        header.setMinimumHeight(48)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(2, 42)
        for row, definition in enumerate(definitions):
            button = QToolButton()
            button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
            )
            button.setToolTip(f"配置{definition.name}")
            button.clicked.connect(lambda _checked=False, item=definition: self._edit_step(item))
            table.setCellWidget(row, 2, button)
        return table

    def refresh(self) -> None:
        siliconflow = self.backend.provider_snapshot("siliconflow")
        mineru = self.backend.provider_snapshot("mineru")
        bocha = self.backend.provider_snapshot("bocha")
        self._set_provider_status(self.siliconflow_status, siliconflow)
        self._set_provider_status(self.mineru_status, mineru)
        self._set_provider_status(self.bocha_status, bocha)
        models = self.backend.get_model_selections()
        self.model_value_labels["chat"].setText(models.chat_model)
        self.model_value_labels["reranker"].setText(models.rerank_model)
        embedding_text = models.embedding_model
        if models.pending_embedding_model:
            embedding_text += f" · 待重建后切换为 {models.pending_embedding_model}"
        self.model_value_labels["embedding"].setText(embedding_text)
        for group, table in self.step_tables.items():
            definitions = [item for item in STEP_DEFINITIONS if item.group == group]
            for row, definition in enumerate(definitions):
                config = self.backend.get_step_config(definition.step_id)
                thinking = (
                    "推荐" if config.thinking is None else ("开启" if config.thinking else "关闭")
                )
                effort = "" if config.reasoning_effort == "auto" else config.reasoning_effort
                transfer = (
                    f"{'流式' if config.stream else '非流式'} · 重试{config.retry_count} · "
                    f"单次 {config.timeout_seconds or self.get_timeout()}秒 · "
                    f"总预算 {config.total_timeout_seconds or self.get_total_timeout()}秒"
                )
                summary = (
                    f"T {config.temperature:.1f} · {thinking}"
                    f"{f' / {effort}' if effort else ''} · {config.max_tokens} tokens · "
                    f"{transfer}"
                )
                for column, value in enumerate((definition.name, summary)):
                    table.setItem(row, column, QTableWidgetItem(value))

    def _edit_provider(self, provider: ProviderName) -> None:
        dialog = ProviderSettingsDialog(self.backend, provider, self)
        if dialog.exec():
            self.refresh()
            self.show_message("API 设置已保存", 4000)

    def _edit_model(self, kind: ModelKind) -> None:
        dialog = ModelSettingsDialog(
            self.backend,
            self.tasks,
            kind,
            show_message=self.show_message,
            on_changed=self.refresh,
            parent=self,
        )
        dialog.exec()
        self.refresh()

    def _edit_step(self, definition: ChatStepDefinition) -> None:
        StepSettingsDialog(
            self.backend,
            definition,
            on_changed=self.refresh,
            parent=self,
        ).exec()

    def _concurrency_changed(self, value: int) -> None:
        if self.set_concurrency is None:
            return
        try:
            self.set_concurrency(value)
        except ValueError as exc:
            self.show_message(str(exc), 5000)
            return
        self.show_message(f"SiliconFlow 最大并发数已设为 {value}", 4000)

    def _timeout_changed(self, value: int) -> None:
        if self.set_timeout is None:
            return
        try:
            self.set_timeout(value)
        except ValueError as exc:
            self.show_message(str(exc), 5000)
            return
        self.refresh()
        self.show_message(f"默认单次尝试上限已设为 {value} 秒", 4000)

    def _total_timeout_changed(self, value: int) -> None:
        if self.set_total_timeout is None:
            return
        try:
            self.set_total_timeout(value)
        except ValueError as exc:
            self.show_message(str(exc), 5000)
            return
        self.refresh()
        self.show_message(f"默认整项调用总预算已设为 {value} 秒", 4000)

    def _stream_idle_timeout_changed(self, value: int) -> None:
        if self.set_stream_idle_timeout is None:
            return
        try:
            self.set_stream_idle_timeout(value)
        except ValueError as exc:
            self.show_message(str(exc), 5000)
            return
        self.show_message(f"流式空闲超时已设为 {value} 秒", 4000)

    def _chat_rounds_changed(self, value: int) -> None:
        if self.set_chat_recent_rounds is None:
            return
        try:
            self.set_chat_recent_rounds(value)
        except ValueError as exc:
            self.show_message(str(exc), 5000)
            return
        self.show_message(f"作者对话将原样保留最近 {value} 轮", 4000)

    def _web_result_count_changed(self, value: int) -> None:
        if self.set_web_search_result_count is None:
            return
        try:
            self.set_web_search_result_count(value)
        except ValueError as exc:
            self.show_message(str(exc), 5000)
            return
        self.show_message(f"每次博查联网检索将获取 {value} 条结果", 4000)

    def _start_siliconflow_check(self) -> None:
        if self._check_task_id is not None:
            return
        self.check_button.setEnabled(False)
        self.siliconflow_status.setText("检测中")

        def task(context: TaskContext):
            context.report_progress(20, "正在连接")
            result = self.siliconflow_check()
            context.report_progress(100, "完成")
            return result

        self._check_task_id = self.tasks.start(
            task,
            on_success=self._check_succeeded,
            on_error=self._check_failed,
            on_progress=lambda percent, message: self.show_message(f"{message} · {percent}%", 0),
        )

    def _check_succeeded(self, result: Any) -> None:
        if isinstance(result, ChatResult):
            detail = f" · {result.usage.total_tokens} tokens"
        elif isinstance(result, list):
            detail = f" · 可见 {len(result)} 个文字模型"
        else:
            detail = ""
        self.siliconflow_status.setText("可用")
        self.siliconflow_status.setObjectName("statusReady")
        self.show_message(f"SiliconFlow 可用{detail}", 5000)
        self._finish_check()

    def _check_failed(self, message: str) -> None:
        self.siliconflow_status.setText("不可用")
        self.siliconflow_status.setObjectName("statusError")
        self.show_message(message, 8000)
        self._finish_check()

    def _finish_check(self) -> None:
        self.siliconflow_status.style().unpolish(self.siliconflow_status)
        self.siliconflow_status.style().polish(self.siliconflow_status)
        self.check_button.setEnabled(True)
        self._check_task_id = None

    @staticmethod
    def _set_provider_status(label: QLabel, snapshot: dict[str, object]) -> None:
        configured = "已配置" if snapshot.get("configured") else "未配置"
        source = {
            "credential_store": "Windows 凭据库",
            "environment": "环境变量",
            "key_test": "key_test.txt",
            "missing": "未设置",
        }.get(str(snapshot.get("source")), "未知来源")
        label.setText(f"{configured} · {source} · {snapshot.get('base_url', '')}")
