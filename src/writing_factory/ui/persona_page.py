"""Source selection, mode control, and background PersonaSpec distillation UI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QElapsedTimer, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from writing_factory.distill.fidelity_models import (
    FidelityStageProgress,
    parse_fidelity_progress,
)
from writing_factory.distill.models import PersonaMode
from writing_factory.distill.options import DistillationOptions
from writing_factory.ui.distillation_dialogs import (
    DistillationQualityPanel,
    PersonaUpgradeDialog,
)
from writing_factory.ui.help_ui import create_help_button
from writing_factory.ui.persona_editor import (
    PersonaEditorWindow,
    PersonaLoader,
    PersonaSaver,
    PersonaVersionLoader,
    RuntimePersonaLoader,
)
from writing_factory.ui.stream_output_panel import StreamOutputPanel
from writing_factory.ui.workers import BackgroundTaskManager, TaskContext


class PersonaPage(QWidget):
    """Distill selected ready documents and list persisted profiles."""

    personas_changed = pyqtSignal()

    def __init__(
        self,
        tasks: BackgroundTaskManager,
        *,
        distill_persona: Callable[
            [str, PersonaMode, set[str], set[str], str, DistillationOptions, TaskContext], Any
        ]
        | None,
        resume_persona: Callable[[str, TaskContext], Any] | None = None,
        upgrade_persona: Callable[
            [str, set[str], set[str], str, DistillationOptions, TaskContext], Any
        ]
        | None = None,
        evaluate_persona: Callable[[str, TaskContext], Any] | None,
        list_sources: Callable[[], list[dict[str, object]]],
        list_personas: Callable[[], list[dict[str, object]]],
        delete_personas: Callable[[set[str], TaskContext], Any] | None,
        load_persona: PersonaLoader | None,
        save_persona: PersonaSaver | None,
        show_message: Callable[[str, int], None],
        load_runtime_persona: RuntimePersonaLoader | None = None,
        list_persona_versions: PersonaVersionLoader | None = None,
        load_distillation_context: Callable[[str], Any] | None = None,
        get_siliconflow_concurrency: Callable[[], int] | None = None,
    ) -> None:
        super().__init__()
        self._tasks = tasks
        self._distill_persona = distill_persona
        self._resume_persona = resume_persona
        self._upgrade_persona = upgrade_persona
        self._evaluate_persona = evaluate_persona
        self._list_sources = list_sources
        self._list_personas = list_personas
        self._delete_personas = delete_personas
        self._load_persona = load_persona
        self._save_persona = save_persona
        self._load_runtime_persona = load_runtime_persona
        self._list_persona_versions = list_persona_versions
        self._load_distillation_context = load_distillation_context
        self._get_siliconflow_concurrency = get_siliconflow_concurrency or (lambda: 3)
        self._show_message = show_message
        self._task_id: str | None = None
        self._active_task_kind: str | None = None
        self._active_fidelity_stage: str | None = None
        self._fidelity_stage_states: dict[str, tuple[str, int]] = {}
        self._fidelity_elapsed = QElapsedTimer()
        self._fidelity_timer = QTimer(self)
        self._fidelity_timer.setInterval(1000)
        self._fidelity_timer.timeout.connect(self._render_fidelity_timing)
        self._editor_windows: dict[str, PersonaEditorWindow] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(14)

        header_row = QHBoxLayout()
        heading = QLabel("作者档案")
        heading.setObjectName("pageTitle")
        self.help_button = create_help_button("persona", self)
        header_row.addWidget(heading)
        header_row.addWidget(self.help_button)
        header_row.addStretch(1)

        config_row = QHBoxLayout()
        name_label = QLabel("名称")
        self.name_input = QLineEdit()
        self.name_input.setMinimumWidth(160)
        self.name_input.setMaximumWidth(280)
        self.name_input.setMaxLength(80)
        self.name_input.textChanged.connect(self._update_button)
        self.domain_input = QLineEdit()
        self.domain_input.setMinimumWidth(140)
        self.domain_input.setMaximumWidth(160)
        self.domain_input.setMaxLength(80)
        self.domain_input.setPlaceholderText("内容领域（对照用）")
        self.domain_input.textChanged.connect(self._update_button)
        config_row.addWidget(name_label)
        config_row.addWidget(self.name_input, 1)
        config_row.addWidget(self.domain_input)
        config_row.addStretch(1)

        self.mode_group = QButtonGroup(self)
        self.person_button = self._mode_button("人物", "person")
        self.topic_button = self._mode_button("主题", "topic")
        self.person_button.setChecked(True)
        self.person_button.toggled.connect(lambda _checked: self._mode_changed())
        self.topic_button.toggled.connect(lambda _checked: self._mode_changed())
        config_row.addWidget(self.person_button)
        config_row.addWidget(self.topic_button)

        self.distill_button = QPushButton("蒸馏")
        self.distill_button.clicked.connect(self.start_distillation)
        self.distill_button.setEnabled(False)
        header_row.addWidget(self.distill_button)
        self.stop_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop),
            "停止",
        )
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_current_task)
        header_row.addWidget(self.stop_button)
        self.evaluate_button = QPushButton("自检")
        self.evaluate_button.setToolTip("运行独立的 Nüwa 保真度自检")
        self.evaluate_button.clicked.connect(self.start_evaluation)
        self.evaluate_button.setEnabled(False)
        header_row.addWidget(self.evaluate_button)
        layout.addLayout(header_row)
        layout.addLayout(config_row)

        self.quality_panel = DistillationQualityPanel(
            counts_provider=lambda: (
                len(self._selected_doc_ids()),
                len(self._selected_control_doc_ids()),
            ),
            concurrency_provider=self._get_siliconflow_concurrency,
            parent=self,
        )
        self.quality_panel.options_changed.connect(self._update_button)
        layout.addWidget(self.quality_panel)
        self.corpus_recommendation_label = QLabel(
            "语料建议：目标 6–12 篇 · 对照 4–8 篇同领域、同文体文本"
        )
        self.corpus_recommendation_label.setObjectName("mutedText")
        self.corpus_recommendation_label.setWordWrap(True)
        layout.addWidget(self.corpus_recommendation_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedHeight(18)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("mutedText")
        self.progress_label.hide()
        layout.addWidget(self.progress_label)
        self.fidelity_timing_label = QLabel("")
        self.fidelity_timing_label.setObjectName("mutedText")
        self.fidelity_timing_label.setWordWrap(True)
        self.fidelity_timing_label.hide()
        layout.addWidget(self.fidelity_timing_label)

        source_label = QLabel("语料")
        source_label.setObjectName("sectionTitle")
        self.source_table = QTableWidget(0, 4)
        self.source_table.setHorizontalHeaderLabels(["目标", "对照", "文件", "切片"])
        self.source_table.setMinimumHeight(130)
        self.source_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.source_table.verticalHeader().setVisible(False)
        source_header = self.source_table.horizontalHeader()
        source_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        source_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        source_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        source_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.source_table.itemChanged.connect(self._source_role_changed)

        profile_header_layout = QHBoxLayout()
        profile_label = QLabel("档案")
        profile_label.setObjectName("sectionTitle")
        profile_header_layout.addWidget(profile_label)
        profile_header_layout.addStretch(1)
        self.delete_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon),
            "删除",
        )
        self.delete_button.setToolTip("删除选中的作者档案")
        self.delete_button.clicked.connect(self.start_deletion)
        self.delete_button.setEnabled(False)
        self.upgrade_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload),
            "升级",
        )
        self.upgrade_button.setToolTip("以选中的已完成档案为基础创建新版本")
        self.upgrade_button.clicked.connect(self.start_upgrade)
        self.upgrade_button.setEnabled(False)
        profile_header_layout.addWidget(self.upgrade_button)
        self.continue_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
            "继续",
        )
        self.continue_button.setToolTip("从选中未完成档案的精确断点继续")
        self.continue_button.clicked.connect(self.start_resume)
        self.continue_button.setEnabled(False)
        profile_header_layout.addWidget(self.continue_button)
        profile_header_layout.addWidget(self.delete_button)
        self.profile_table = QTableWidget(0, 9)
        self.profile_table.setHorizontalHeaderLabels(
            ["选择", "名称", "模式", "版本", "状态", "质量", "心智模型", "自检", "调研日期"]
        )
        self.profile_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.profile_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.profile_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.profile_table.setAlternatingRowColors(True)
        self.profile_table.itemSelectionChanged.connect(self._update_button)
        self.profile_table.itemChanged.connect(self._update_button)
        self.profile_table.cellDoubleClicked.connect(self._open_profile)
        self.profile_table.verticalHeader().setVisible(False)
        profile_header = self.profile_table.horizontalHeader()
        profile_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        profile_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (2, 3, 4, 5, 6, 7, 8):
            profile_header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

        source_section = QWidget()
        source_layout = QVBoxLayout(source_section)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(6)
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_table, 1)

        profile_section = QWidget()
        profile_layout = QVBoxLayout(profile_section)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.setSpacing(6)
        profile_layout.addLayout(profile_header_layout)
        profile_layout.addWidget(self.profile_table, 1)

        self.tables_splitter = QSplitter(Qt.Orientation.Vertical)
        self.tables_splitter.setChildrenCollapsible(False)
        self.tables_splitter.addWidget(source_section)
        self.tables_splitter.addWidget(profile_section)
        self.tables_splitter.setStretchFactor(0, 2)
        self.tables_splitter.setStretchFactor(1, 3)
        self.tables_splitter.setSizes([210, 300])

        self.output_panel = StreamOutputPanel()
        self.output_panel.setMinimumWidth(320)
        self.output_panel.hide()
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.addWidget(self.tables_splitter)
        self.content_splitter.addWidget(self.output_panel)
        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 2)
        self.content_splitter.setSizes([620, 380])
        layout.addWidget(self.content_splitter, 1)
        self.refresh()

    def _mode_button(self, label: str, mode: PersonaMode) -> QPushButton:
        button = QPushButton(label)
        button.setCheckable(True)
        button.setProperty("personaMode", mode)
        button.setObjectName("modeButton")
        button.setFixedWidth(62)
        self.mode_group.addButton(button)
        return button

    def refresh(self) -> None:
        """Reload ready sources and all profiles from SQLite."""

        self.source_table.blockSignals(True)
        sources = [item for item in self._list_sources() if item.get("status") == "ready"]
        self.source_table.setRowCount(len(sources))
        for row, source in enumerate(sources):
            target = self._source_checkbox(source.get("doc_id"), checked=True)
            control = self._source_checkbox(source.get("doc_id"), checked=False)
            self.source_table.setItem(row, 0, target)
            self.source_table.setItem(row, 1, control)
            self.source_table.setItem(row, 2, QTableWidgetItem(str(source.get("filename", ""))))
            self.source_table.setItem(row, 3, QTableWidgetItem(str(source.get("chunk_count", 0))))
        self.source_table.blockSignals(False)

        profiles = self._list_personas()
        self.profile_table.blockSignals(True)
        self.profile_table.setRowCount(len(profiles))
        for row, profile in enumerate(profiles):
            select = QTableWidgetItem()
            select.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            select.setCheckState(Qt.CheckState.Unchecked)
            select.setData(Qt.ItemDataRole.UserRole, profile.get("persona_id"))
            select.setData(Qt.ItemDataRole.UserRole + 1, profile.get("status"))
            select.setData(Qt.ItemDataRole.UserRole + 2, profile)
            self.profile_table.setItem(row, 0, select)
            values = (
                str(profile.get("name", "")),
                "人物" if profile.get("mode") == "person" else "主题",
                f"v{profile.get('version_number', 1)} / {profile.get('version_count', 1)}",
                self._status_label(
                    str(profile.get("status", "")),
                    str(profile.get("error_type", "")),
                ),
                str(profile.get("quality_label", "历史完整模式")),
                str(profile.get("model_count", 0)),
                self._score_label(
                    profile.get("fidelity_score"),
                    profile.get("fidelity_checkpoint_count"),
                ),
                str(profile.get("research_date", ""))[:10],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                self.profile_table.setItem(row, column + 1, item)
        self.profile_table.blockSignals(False)
        self._update_quality_context()
        self._update_button()

    def start_distillation(self) -> None:
        """Start one distillation using only checked source documents."""

        if self._task_id is not None or self._distill_persona is None:
            return
        doc_ids = self._selected_doc_ids()
        name = self.name_input.text().strip()
        if not name or not doc_ids:
            return
        mode = self._selected_mode()
        control_doc_ids = self._selected_control_doc_ids() if mode == "person" else set()
        domain = self.domain_input.text().strip() if mode == "person" else ""
        options = self.quality_panel.options_for_context(
            mode=mode,
            has_control=bool(control_doc_ids),
        )
        self._active_task_kind = "distillation"
        self.fidelity_timing_label.hide()
        self._set_running(True)
        self.output_panel.clear()
        self.output_panel.show()
        self.content_splitter.setSizes([620, 380])
        self._show_message("准备蒸馏", 0)

        def task(context: TaskContext):
            return self._distill_persona(
                name,
                mode,
                doc_ids,
                control_doc_ids,
                domain,
                options,
                context,
            )

        self._task_id = self._tasks.start(
            task,
            on_success=self._succeeded,
            on_error=self._failed,
            on_progress=self._progressed,
            on_stream=self.output_panel.append_stream,
        )

    def start_resume(self) -> None:
        """Resume exactly one selected interrupted persona version."""

        if self._task_id is not None or self._resume_persona is None:
            return
        profile = self._selected_profile()
        if profile is None or profile.get("status") == "ready":
            return
        persona_id = str(profile["persona_id"])
        self._active_task_kind = "distillation"
        self.fidelity_timing_label.hide()
        self._set_running(True)
        self.output_panel.clear()
        self.output_panel.show()
        self.content_splitter.setSizes([620, 380])
        self._show_message("读取精确蒸馏断点", 0)

        def task(context: TaskContext):
            return self._resume_persona(persona_id, context)

        self._task_id = self._tasks.start(
            task,
            on_success=self._succeeded,
            on_error=self._failed,
            on_progress=self._progressed,
            on_stream=self.output_panel.append_stream,
        )

    def start_upgrade(self) -> None:
        """Create a new version from one selected ready profile and chosen sources."""

        if (
            self._task_id is not None
            or self._upgrade_persona is None
            or self._load_distillation_context is None
        ):
            return
        profile = self._selected_profile()
        if profile is None or profile.get("status") != "ready":
            return
        persona_id = str(profile["persona_id"])
        context = self._load_distillation_context(persona_id)
        if context is None:
            self._show_message("无法读取所选档案的原始蒸馏配置", 6000)
            return
        dialog = PersonaUpgradeDialog(
            sources=self._list_sources(),
            target_doc_ids=context.target_doc_ids,
            control_doc_ids=context.control_doc_ids,
            domain=context.domain,
            parent=self,
        )
        if dialog.exec() != PersonaUpgradeDialog.DialogCode.Accepted:
            return
        selection = dialog.selection()
        profile_mode: PersonaMode = "topic" if profile.get("mode") == "topic" else "person"
        options = self.quality_panel.options_for_context(
            mode=profile_mode,
            has_control=bool(selection.control_doc_ids),
        )
        self._active_task_kind = "distillation"
        self.fidelity_timing_label.hide()
        self._set_running(True)
        self.output_panel.clear()
        self.output_panel.show()
        self.content_splitter.setSizes([620, 380])
        self._show_message("准备升级作者档案", 0)

        def task(task_context: TaskContext):
            return self._upgrade_persona(
                persona_id,
                set(selection.target_doc_ids),
                set(selection.control_doc_ids),
                selection.domain,
                options,
                task_context,
            )

        self._task_id = self._tasks.start(
            task,
            on_success=self._succeeded,
            on_error=self._failed,
            on_progress=self._progressed,
            on_stream=self.output_panel.append_stream,
        )

    def start_evaluation(self) -> None:
        """Run the selected ready profile through the paid independent self-check."""

        if self._task_id is not None or self._evaluate_persona is None:
            return
        persona_id = self._selected_persona_id()
        if persona_id is None:
            return
        self._active_task_kind = "fidelity"
        self._prepare_fidelity_timing()
        self._set_running(True)
        self.output_panel.clear()
        self.output_panel.show()
        self.content_splitter.setSizes([620, 380])
        self._show_message("准备独立自检", 0)

        def task(context: TaskContext):
            return self._evaluate_persona(persona_id, context)

        self._task_id = self._tasks.start(
            task,
            on_success=self._evaluation_succeeded,
            on_error=self._failed,
            on_progress=self._progressed,
            on_stream=self.output_panel.append_stream,
        )

    def stop_current_task(self) -> None:
        """Request cooperative cancellation of the active distillation or self-check."""

        if self._task_id is None:
            return
        self._tasks.cancel(self._task_id)
        self.stop_button.setEnabled(False)
        self.progress_label.setText("正在安全停止，已完成的断点会保留")
        self._show_message("正在安全停止作者档案任务", 5000)

    def start_deletion(self) -> None:
        """在后台删除单个或多个选中档案，不弹确认框。"""

        if self._task_id is not None or self._delete_personas is None:
            return
        persona_ids = self._selected_persona_ids()
        if not persona_ids:
            return
        self._active_task_kind = "deletion"
        self._set_running(True)
        self._show_message(f"准备删除 · {len(persona_ids)} 个档案", 0)

        def task(context: TaskContext):
            return self._delete_personas(persona_ids, context)

        self._task_id = self._tasks.start(
            task,
            on_success=lambda result: self._deletion_succeeded(persona_ids, result),
            on_error=self._failed,
            on_progress=self._progressed,
        )

    def _selected_doc_ids(self) -> set[str]:
        """返回目标作者语料。"""

        return self._checked_source_ids(0)

    def _selected_control_doc_ids(self) -> set[str]:
        """返回本次可选的同领域对照语料。"""

        return self._checked_source_ids(1)

    def _checked_source_ids(self, column: int) -> set[str]:
        selected: set[str] = set()
        for row in range(self.source_table.rowCount()):
            item = self.source_table.item(row, column)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                value = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(value, str):
                    selected.add(value)
        return selected

    @staticmethod
    def _source_checkbox(doc_id: object, *, checked: bool) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        item.setData(Qt.ItemDataRole.UserRole, doc_id)
        return item

    def _source_role_changed(self, item: QTableWidgetItem) -> None:
        """目标和对照角色互斥；主题模式会忽略对照列。"""

        if item.column() in (0, 1) and item.checkState() == Qt.CheckState.Checked:
            other = self.source_table.item(item.row(), 1 - item.column())
            if other is not None and other.checkState() == Qt.CheckState.Checked:
                self.source_table.blockSignals(True)
                other.setCheckState(Qt.CheckState.Unchecked)
                self.source_table.blockSignals(False)
        self._update_quality_context()
        self._update_button()

    def _selected_mode(self) -> PersonaMode:
        return "topic" if self.topic_button.isChecked() else "person"

    def _mode_changed(self) -> None:
        if self._selected_mode() == "topic":
            self.corpus_recommendation_label.setText(
                "主题语料建议：目标 6–12 篇 · 至少 4 篇才能运行留出生成力验证"
            )
        else:
            self.corpus_recommendation_label.setText(
                "语料建议：目标 6–12 篇 · 对照 4–8 篇同领域、同文体文本"
            )
        self._update_quality_context()
        self._update_button()

    def _selected_persona_id(self) -> str | None:
        profile = self._selected_profile()
        if profile is None or profile.get("status") != "ready":
            return None
        value = profile.get("persona_id")
        return value if isinstance(value, str) else None

    def _selected_persona_rows(self) -> list[int]:
        selection = self.profile_table.selectionModel()
        if selection is None:
            return []
        return sorted(index.row() for index in selection.selectedRows(1))

    def _selected_persona_ids(self) -> set[str]:
        checked = self._checked_persona_ids()
        if checked:
            return checked
        identifiers: set[str] = set()
        for row in self._selected_persona_rows():
            item = self.profile_table.item(row, 0)
            value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if isinstance(value, str):
                identifiers.add(value)
        return identifiers

    def _selected_profile(self) -> dict[str, object] | None:
        identifiers = self._selected_persona_ids()
        if len(identifiers) != 1:
            return None
        identifier = next(iter(identifiers))
        for row in range(self.profile_table.rowCount()):
            item = self.profile_table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == identifier:
                payload = item.data(Qt.ItemDataRole.UserRole + 2)
                return payload if isinstance(payload, dict) else None
        return None

    def _update_quality_context(self) -> None:
        self.quality_panel.set_context(
            mode=self._selected_mode(),
            has_control=bool(self._selected_control_doc_ids()),
        )

    def _checked_persona_ids(self) -> set[str]:
        identifiers: set[str] = set()
        for row in range(self.profile_table.rowCount()):
            item = self.profile_table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                value = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(value, str):
                    identifiers.add(value)
        return identifiers

    def _update_button(self) -> None:
        profile = self._selected_profile()
        enabled = (
            self._distill_persona is not None
            and self._task_id is None
            and bool(self.name_input.text().strip())
            and bool(self._selected_doc_ids())
            and (
                self._selected_mode() == "topic"
                or not self._selected_control_doc_ids()
                or bool(self.domain_input.text().strip())
            )
        )
        self.distill_button.setEnabled(enabled)
        checkpoint_count = (
            profile.get("fidelity_checkpoint_count", 0) if profile is not None else 0
        )
        resumable_fidelity = (
            profile is not None
            and profile.get("fidelity_score") is None
            and isinstance(checkpoint_count, int)
            and checkpoint_count > 0
        )
        self.evaluate_button.setText("继续自检" if resumable_fidelity else "自检")
        self.evaluate_button.setToolTip(
            "从已完成阶段继续档案自检"
            if resumable_fidelity
            else "运行独立的 Nüwa 保真度自检"
        )
        self.evaluate_button.setEnabled(
            self._evaluate_persona is not None
            and self._task_id is None
            and self._selected_persona_id() is not None
        )
        self.delete_button.setEnabled(
            self._delete_personas is not None
            and self._task_id is None
            and bool(self._selected_persona_ids())
        )
        self.continue_button.setEnabled(
            self._resume_persona is not None
            and self._task_id is None
            and profile is not None
            and profile.get("status") != "ready"
        )
        self.upgrade_button.setEnabled(
            self._upgrade_persona is not None
            and self._load_distillation_context is not None
            and self._task_id is None
            and profile is not None
            and profile.get("status") == "ready"
        )

    def _open_profile(self, row: int, _column: int) -> None:
        """双击可用档案时打开或激活对应的独立编辑窗口。"""

        if self._load_persona is None or self._save_persona is None:
            return
        item = self.profile_table.item(row, 0)
        if item is None or item.data(Qt.ItemDataRole.UserRole + 1) != "ready":
            self._show_message("只能查看已经完成的档案", 5000)
            return
        persona_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(persona_id, str):
            return
        existing = self._editor_windows.get(persona_id)
        if existing is not None:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return
        editor = PersonaEditorWindow(
            persona_id,
            load_persona=self._load_persona,
            save_persona=self._save_persona,
            load_runtime_persona=self._load_runtime_persona,
            list_persona_versions=self._list_persona_versions,
            parent=self,
        )
        editor.saved.connect(lambda _persona_id: self.refresh())
        editor.destroyed.connect(lambda: self._editor_windows.pop(persona_id, None))
        self._editor_windows[persona_id] = editor
        editor.show()

    def _deletion_succeeded(self, persona_ids: set[str], result: Any) -> None:
        removed = int(result) if isinstance(result, int) else 0
        for persona_id in persona_ids:
            editor = self._editor_windows.pop(persona_id, None)
            if editor is not None:
                editor.close()
        self._show_message(f"已删除 · {removed} 个档案", 6000)
        self._set_running(False)
        self.refresh()
        self.personas_changed.emit()

    def _progressed(self, percent: int, message: str) -> None:
        self.progress.setValue(percent)
        if message:
            fidelity_event = parse_fidelity_progress(message)
            if fidelity_event is not None:
                self._update_fidelity_timing(fidelity_event)
                message = self._fidelity_event_text(fidelity_event)
            self.progress_label.setText(message)
            self._show_message(f"{message} · {percent}%", 0)

    def _succeeded(self, result: Any) -> None:
        count = len(getattr(getattr(result, "persona", None), "mental_models", []))
        self._show_message(f"蒸馏完成 · {count} 个心智模型", 6000)
        self._set_running(False)
        self.refresh()
        self.personas_changed.emit()

    def _failed(self, message: str) -> None:
        task_kind = self._active_task_kind
        self.output_panel.discard_incomplete_attempts()
        if message == "任务已取消":
            label = "档案自检" if task_kind == "fidelity" else "作者蒸馏"
            self.output_panel.append_stream(
                f"status::{label}",
                "任务已停止；当前不完整输出未保存，已完成断点仍保留",
            )
            self._show_message(f"{label}已停止 · 已完成断点仍保留", 8000)
        else:
            if task_kind in {"fidelity", "distillation"}:
                label = "档案自检" if task_kind == "fidelity" else "作者蒸馏"
                self.output_panel.append_stream(f"error::{label}", message)
                self.output_panel.show()
            self._show_message(message, 8000)
        if task_kind == "fidelity":
            self._finish_active_fidelity_stage()
        self._set_running(False)
        self.refresh()

    def _evaluation_succeeded(self, result: Any) -> None:
        self._fidelity_timer.stop()
        self._active_fidelity_stage = None
        self._render_fidelity_timing()
        score = getattr(result, "total", None)
        label = str(score) if isinstance(score, int) else "完成"
        self._show_message(f"独立自检完成 · {label}/100", 6000)
        self._set_running(False)
        self.refresh()

    def _set_running(self, running: bool) -> None:
        self.progress.setValue(0)
        self.progress.setVisible(running)
        self.progress_label.setVisible(running)
        self.name_input.setEnabled(not running)
        self.domain_input.setEnabled(not running)
        self.person_button.setEnabled(not running)
        self.topic_button.setEnabled(not running)
        self.source_table.setEnabled(not running)
        self.profile_table.setEnabled(not running)
        self.quality_panel.setEnabled(not running)
        self.distill_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.evaluate_button.setEnabled(not running)
        self.delete_button.setEnabled(False)
        self.continue_button.setEnabled(False)
        self.upgrade_button.setEnabled(False)
        if not running:
            self._task_id = None
            self._active_task_kind = None
            self._update_button()

    @staticmethod
    def _status_label(status: str, error_type: str = "") -> str:
        if status == "failed" and error_type == "TaskCancelled":
            return "未完成"
        return {
            "ready": "可用",
            "mapping": "提取中",
            "reducing": "归并中",
            "validating": "校验中",
            "failed": "失败",
        }.get(status, status)

    @staticmethod
    def _score_label(score: object, checkpoint_count: object = 0) -> str:
        if score is not None:
            return f"{score}/100"
        count = checkpoint_count if isinstance(checkpoint_count, int) else 0
        return f"未检 · 可续 {count}/3" if count else "未检"

    def _prepare_fidelity_timing(self) -> None:
        self._fidelity_timer.stop()
        self._active_fidelity_stage = None
        self._fidelity_stage_states = {
            "design": ("等待", 0),
            "answer": ("等待", 0),
            "judge": ("等待", 0),
        }
        self.fidelity_timing_label.show()
        self._render_fidelity_timing()

    def _update_fidelity_timing(self, event: FidelityStageProgress) -> None:
        if not self._fidelity_stage_states:
            self._prepare_fidelity_timing()
        if event.state == "started":
            self._active_fidelity_stage = event.stage
            self._fidelity_elapsed.start()
            self._fidelity_stage_states[event.stage] = ("运行中", 0)
            self._fidelity_timer.start()
        else:
            if self._active_fidelity_stage == event.stage:
                self._fidelity_timer.stop()
                self._active_fidelity_stage = None
            state = {
                "restored": "已恢复",
                "completed": "完成",
                "failed": "失败",
            }[event.state]
            self._fidelity_stage_states[event.stage] = (state, event.duration_ms)
        self._render_fidelity_timing()

    def _finish_active_fidelity_stage(self) -> None:
        stage = self._active_fidelity_stage
        if stage is None:
            return
        duration_ms = max(0, self._fidelity_elapsed.elapsed())
        self._fidelity_stage_states[stage] = ("失败", duration_ms)
        self._active_fidelity_stage = None
        self._fidelity_timer.stop()
        self._render_fidelity_timing()

    def _render_fidelity_timing(self) -> None:
        if not self._fidelity_stage_states:
            return
        labels = {"design": "设计问题", "answer": "盲测回答", "judge": "中性评判"}
        parts = []
        for stage in ("design", "answer", "judge"):
            state, duration_ms = self._fidelity_stage_states[stage]
            if stage == self._active_fidelity_stage and self._fidelity_elapsed.isValid():
                duration_ms = max(0, self._fidelity_elapsed.elapsed())
            duration = self._format_duration(duration_ms)
            suffix = "" if state == "等待" else f" {duration}"
            parts.append(f"{labels[stage]}：{state}{suffix}")
        self.fidelity_timing_label.setText("自检阶段 · " + "  |  ".join(parts))

    @staticmethod
    def _fidelity_event_text(event: FidelityStageProgress) -> str:
        labels = {"design": "设计自检问题", "answer": "档案盲测回答", "judge": "中性独立评判"}
        states = {
            "started": "运行中",
            "restored": "已从断点恢复",
            "completed": "已完成",
            "failed": "失败",
        }
        text = f"{labels[event.stage]} · {states[event.state]}"
        if event.state != "started":
            text += f" · {PersonaPage._format_duration(event.duration_ms)}"
        return text

    @staticmethod
    def _format_duration(duration_ms: int) -> str:
        total_seconds = max(0, duration_ms // 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
