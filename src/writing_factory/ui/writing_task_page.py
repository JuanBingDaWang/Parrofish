"""Writing task page — run the full generation pipeline (Stages 4-6) and
display evaluation results (Stage 7) in a single UI surface."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QTextCursor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from writing_factory.generate.models import GenerationOptions
from writing_factory.orchestration.state import (
    PIPELINE_STATUS_ASSEMBLING,
    PIPELINE_STATUS_DONE,
    PIPELINE_STATUS_DRAFTING,
    PIPELINE_STATUS_ERROR,
    PIPELINE_STATUS_EVIDENCE_PREFETCH,
    PIPELINE_STATUS_FRAMEWORK,
    PIPELINE_STATUS_GLOBAL_POLISH,
    PIPELINE_STATUS_STRUCTURE_REVIEW,
    PIPELINE_STATUS_TERM_REVIEW,
    PIPELINE_STATUS_TOPIC,
    PIPELINE_STATUS_VERIFYING,
)
from writing_factory.ui.live_output_window import LiveOutputWindow
from writing_factory.ui.time_format import format_china_datetime
from writing_factory.ui.workers import BackgroundTaskManager, TaskContext

logger = logging.getLogger(__name__)

# ── Status display helpers ─────────────────────────────────────

_PIPELINE_LABELS: dict[str, str] = {
    PIPELINE_STATUS_TOPIC: "选题中",
    PIPELINE_STATUS_FRAMEWORK: "构建框架",
    PIPELINE_STATUS_EVIDENCE_PREFETCH: "证据已冻结",
    PIPELINE_STATUS_DRAFTING: "起草中",
    PIPELINE_STATUS_VERIFYING: "核对中",
    PIPELINE_STATUS_TERM_REVIEW: "术语审查",
    PIPELINE_STATUS_STRUCTURE_REVIEW: "结构审查",
    PIPELINE_STATUS_GLOBAL_POLISH: "全局打磨",
    PIPELINE_STATUS_ASSEMBLING: "组装参考文献",
    PIPELINE_STATUS_DONE: "完成",
    PIPELINE_STATUS_ERROR: "出错",
}

_SECTION_LABELS: dict[str, str] = {
    "pending": "⏳ 等待",
    "drafting": "▶ 起草",
    "drafted": "✓ 草稿",
    "verifying": "▶ 核对",
    "verified": "✓ 已核",
    "revising": "▶ 修订",
    "polishing": "▶ 打磨",
    "polished": "✓ 完成",
    "error": "✗ 错误",
}

_TASK_STATUS_LABELS: dict[str, str] = {
    "pending": "待开始",
    "running": "运行中",
    "cancelled": "已取消",
    PIPELINE_STATUS_DONE: "已完成",
    PIPELINE_STATUS_ERROR: "失败",
}


class WritingTaskPage(QWidget):
    """Run the writing pipeline and display results.

    Layout:
        ┌─ 任务配置 ──────────────────────────┐
        │ Persona / 主题 / 领域 / [开始] [停止] │
        ├─ 写入进度 ──────────────────────────┤
        │ [████████░░] 60%                     │
        │ 状态: 起草中 (第 3/5 节)              │
        │ 节状态列表                           │
        ├─ 写入结果 ──────────────────────────┤
        │ [论文] [提纲] [参考文献] [评估]        │
        └──────────────────────────────────────┘
    """

    def __init__(
        self,
        tasks: BackgroundTaskManager,
        *,
        list_personas: Callable[[], list[dict[str, object]]],
        list_projects: Callable[[], list[dict[str, object]]],
        list_documents: Callable[[], list[dict[str, object]]],
        run_writing_pipeline: Callable[..., Any] | None,
        evaluate_generation: Callable[..., Any] | None,
        create_writing_task: Callable[..., str] | None,
        list_writing_tasks: Callable[[str], list[dict[str, object]]],
        load_writing_task: Callable[[str], dict[str, object] | None] | None,
        save_edited_draft: Callable[..., None] | None,
        delete_writing_tasks: Callable[[set[str]], int] | None,
        preview_source_selection: Callable[[str, set[str], set[str]], dict[str, int]]
        | None = None,
        show_message: Callable[[str, int], None],
    ) -> None:
        super().__init__()
        self._tasks = tasks
        self._list_personas = list_personas
        self._list_projects = list_projects
        self._list_documents = list_documents
        self._run_writing_pipeline = run_writing_pipeline
        self._evaluate_generation = evaluate_generation
        self._create_writing_task = create_writing_task
        self._list_writing_tasks = list_writing_tasks
        self._load_writing_task = load_writing_task
        self._save_edited_draft = save_edited_draft
        self._delete_writing_tasks = delete_writing_tasks
        self._preview_source_selection = preview_source_selection
        self._show_message = show_message
        self._task_id: str | None = None
        self._writing_task_id: str | None = None
        self._last_result: dict[str, Any] | None = None
        self._current_section_index: int | None = None
        self._personas: list[dict[str, object]] = []
        self._task_records: list[dict[str, object]] = []
        self._history_refreshed_for_run = False
        self._run_started_at: float | None = None
        self._step_started_at: float | None = None
        self._last_stream_at: float | None = None
        self._current_step = ""
        self._stream_stage = ""
        self._diagnostic_stream_stage = ""
        self._live_attempt_start = 0
        self._diagnostic_attempt_start = 0
        self._stream_auto_switched = False
        self._updating_quality_controls = False
        self._live_output_window: LiveOutputWindow | None = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed_display)
        self._build_ui()

    # ── UI construction ────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        # ── Header ──
        header = QHBoxLayout()
        heading = QLabel("写作任务")
        heading.setObjectName("pageTitle")
        header.addWidget(heading)
        header.addStretch(1)
        layout.addLayout(header)

        self.source_summary_label = QLabel("已选 0 篇 · 隔离 0 篇 · 实际可用 0 篇")
        self.source_summary_label.setObjectName("mutedText")
        self.source_summary_label.setWordWrap(True)
        layout.addWidget(self.source_summary_label)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        layout.addWidget(self.main_splitter, 1)

        # ── Config panel ──
        config_group = QGroupBox("任务配置")
        config_group.setObjectName("configGroup")
        config_layout = QFormLayout(config_group)
        config_layout.setSpacing(10)

        self.project_combo = QComboBox()
        self.project_combo.currentIndexChanged.connect(self._reload_task_history)
        config_layout.addRow("所属项目:", self.project_combo)

        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("任务标题")
        config_layout.addRow("任务标题:", self.title_input)

        persona_row = QHBoxLayout()
        self.persona_combo = QComboBox()
        self.persona_combo.setMinimumWidth(300)
        self.persona_combo.setToolTip("选择一个已蒸馏的作者档案作为写作风格来源")
        self.persona_combo.currentIndexChanged.connect(self._update_source_summary)
        self.refresh_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload),
            "",
        )
        self.refresh_button.setFixedWidth(36)
        self.refresh_button.setToolTip("刷新 Persona 列表")
        self.refresh_button.clicked.connect(self._reload_personas)
        persona_row.addWidget(self.persona_combo, 1)
        persona_row.addWidget(self.refresh_button)
        config_layout.addRow("写作风格:", persona_row)

        self.task_input = QTextEdit()
        self.task_input.setPlaceholderText(
            "描述写作任务，例如：「请写一篇关于数字人文在出版领域应用研究的论文」"
        )
        self.task_input.setMaximumHeight(72)
        config_layout.addRow("主题/要求:", self.task_input)

        domain_row = QHBoxLayout()
        self.domain_input = QTextEdit()
        self.domain_input.setPlaceholderText("研究领域（可选），如「出版学」")
        self.domain_input.setMaximumHeight(48)
        self.domain_input.setMaximumWidth(300)
        domain_row.addWidget(self.domain_input, 1)
        domain_row.addStretch(1)
        config_layout.addRow("研究领域:", domain_row)

        self.target_length_spin = QSpinBox()
        self.target_length_spin.setRange(0, 100000)
        self.target_length_spin.setSingleStep(500)
        self.target_length_spin.setSpecialValueText("自动识别")
        self.target_length_spin.setSuffix(" 字")
        self.target_length_spin.setToolTip("设为自动时，从主题/要求中的“1500字”等表达识别")
        config_layout.addRow("目标篇幅:", self.target_length_spin)

        self.quality_preset_combo = QComboBox()
        self.quality_preset_combo.addItem("严谨定稿", "strict")
        self.quality_preset_combo.addItem("平衡模式", "balanced")
        self.quality_preset_combo.addItem("快速草稿", "fast_draft")
        self.quality_preset_combo.addItem("自定义", "custom")
        self.quality_preset_combo.currentIndexChanged.connect(self._apply_quality_preset)
        config_layout.addRow("质量模式:", self.quality_preset_combo)

        quality_widget = QWidget()
        quality_grid = QGridLayout(quality_widget)
        quality_grid.setContentsMargins(0, 0, 0, 0)
        quality_grid.setHorizontalSpacing(16)
        quality_grid.setVerticalSpacing(6)
        self.fact_verification_checkbox = QCheckBox("事实语义核验")
        self.section_polish_checkbox = QCheckBox("章节文风打磨")
        self.section_drift_checkbox = QCheckBox("章节防漂移")
        self.term_review_checkbox = QCheckBox("术语审查")
        self.structure_review_checkbox = QCheckBox("结构审查")
        self.global_polish_checkbox = QCheckBox("全局打磨")
        self.global_drift_checkbox = QCheckBox("全局防漂移")
        self._quality_checkboxes = (
            self.fact_verification_checkbox,
            self.section_polish_checkbox,
            self.section_drift_checkbox,
            self.term_review_checkbox,
            self.structure_review_checkbox,
            self.global_polish_checkbox,
            self.global_drift_checkbox,
        )
        for index, checkbox in enumerate(self._quality_checkboxes):
            quality_grid.addWidget(checkbox, index // 2, index % 2)
            checkbox.toggled.connect(self._mark_quality_custom)
        config_layout.addRow("质量步骤:", quality_widget)
        self._apply_quality_preset()

        self.document_list = QListWidget()
        self.document_list.setMinimumHeight(180)
        self.document_list.setToolTip("勾选本任务允许作为事实与引用来源的文档")
        self.document_list.itemChanged.connect(self._update_source_summary)
        config_layout.addRow("事实语料:", self.document_list)
        self.allow_persona_sources = QCheckBox("明确允许复用所选作者蒸馏语料作为事实来源")
        self.allow_persona_sources.setToolTip(
            "默认隔离作者旧论文；只有本任务确实需要引用它们时才开启"
        )
        self.allow_persona_sources.stateChanged.connect(self._update_source_summary)
        config_layout.addRow("来源隔离:", self.allow_persona_sources)

        button_row = QHBoxLayout()
        self.start_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
            "开始写作",
        )
        self.start_button.setEnabled(self._run_writing_pipeline is not None)
        self.start_button.setToolTip("启动全流水线写作（阶段 4-6）")
        self.start_button.clicked.connect(self._start_writing)
        self.stop_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop),
            "停止",
        )
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("取消正在进行的写作任务")
        self.stop_button.clicked.connect(self._stop_writing)
        self.eval_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView),
            "评估结果",
        )
        self.eval_button.setEnabled(False)
        self.eval_button.setToolTip("对已完成稿件运行阶段 7 评估")
        self.eval_button.clicked.connect(self._run_evaluation)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addStretch(1)
        button_row.addWidget(self.eval_button)
        config_layout.addRow("", button_row)

        self.config_scroll = QScrollArea()
        self.config_scroll.setWidgetResizable(True)
        self.config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.config_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.config_scroll.setWidget(config_group)
        self.config_scroll.setMinimumHeight(190)

        history_group = QGroupBox("项目任务")
        history_layout = QVBoxLayout(history_group)
        history_buttons = QHBoxLayout()
        self.resume_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "继续所选"
        )
        self.resume_button.clicked.connect(self._resume_selected_task)
        self.delete_task_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon), "删除所选"
        )
        self.delete_task_button.clicked.connect(self._delete_selected_tasks)
        history_buttons.addWidget(self.resume_button)
        history_buttons.addWidget(self.delete_task_button)
        history_buttons.addStretch(1)
        history_layout.addLayout(history_buttons)
        self.task_table = QTableWidget(0, 4)
        self.task_table.setHorizontalHeaderLabels(["", "任务", "状态", "更新时间"])
        self.task_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.task_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.task_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.task_table.setMinimumHeight(92)
        self.task_table.verticalHeader().setVisible(False)
        task_header = self.task_table.horizontalHeader()
        task_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        task_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        task_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        task_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.task_table.itemDoubleClicked.connect(lambda _item: self._load_selected_task())
        self.task_table.itemSelectionChanged.connect(self._update_task_error_banner)
        history_layout.addWidget(self.task_table)
        self.task_error_label = QPlainTextEdit()
        self.task_error_label.setObjectName("errorText")
        self.task_error_label.setReadOnly(True)
        self.task_error_label.setMaximumHeight(96)
        self.task_error_label.setToolTip("可滚动查看并复制完整失败详情")
        self.task_error_label.hide()
        history_layout.addWidget(self.task_error_label)
        history_group.setMinimumHeight(150)
        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.history_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.history_scroll.setWidget(history_group)
        self.history_scroll.setMinimumHeight(90)
        self.history_scroll.setMinimumWidth(250)
        self.main_splitter.addWidget(self.history_scroll)

        # ── Progress panel ──
        self.progress_group = QGroupBox("写入进度")
        self.progress_group.setObjectName("progressGroup")
        progress_layout = QVBoxLayout(self.progress_group)
        progress_layout.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusLabel")
        progress_layout.addWidget(self.status_label)

        self.elapsed_label = QLabel("步骤耗时 00:00 · 本次运行 00:00")
        self.elapsed_label.setObjectName("mutedText")
        progress_layout.addWidget(self.elapsed_label)

        self.activity_label = QLabel("尚未收到模型流式输出")
        self.activity_label.setObjectName("mutedText")
        progress_layout.addWidget(self.activity_label)

        self.section_table = QTableWidget(0, 5)
        self.section_table.setHorizontalHeaderLabels(["节", "标题", "状态", "修订", "耗时"])
        self.section_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.section_table.setAlternatingRowColors(True)
        self.section_table.setMinimumHeight(84)
        self.section_table.verticalHeader().setVisible(False)
        sh = self.section_table.horizontalHeader()
        sh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        sh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        sh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        sh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        sh.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        section_tab = QWidget()
        section_layout = QVBoxLayout(section_tab)
        section_layout.setContentsMargins(0, 6, 0, 0)
        section_layout.addWidget(self.section_table)

        self.live_output_view = QPlainTextEdit()
        self.live_output_view.setReadOnly(True)
        self.live_output_view.setPlaceholderText("模型的实时输出将在这里逐步显示")
        self.live_output_view.setMinimumHeight(110)
        self.live_output_view.document().setMaximumBlockCount(5000)

        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)
        live_layout.setContentsMargins(0, 6, 0, 0)
        live_toolbar = QHBoxLayout()
        self.auto_scroll_checkbox = QCheckBox("自动滚动")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.toggled.connect(self._sync_live_output_auto_scroll)
        live_toolbar.addWidget(self.auto_scroll_checkbox)
        live_toolbar.addStretch(1)

        self.copy_live_output_button = QToolButton()
        self.copy_live_output_button.setIcon(
            self._theme_icon("edit-copy", QStyle.StandardPixmap.SP_FileIcon)
        )
        self.copy_live_output_button.setToolTip("复制全部实时输出")
        self.copy_live_output_button.clicked.connect(self._copy_live_output)
        live_toolbar.addWidget(self.copy_live_output_button)

        self.popout_live_output_button = QToolButton()
        self.popout_live_output_button.setIcon(
            self._theme_icon("window-new", QStyle.StandardPixmap.SP_TitleBarNormalButton)
        )
        self.popout_live_output_button.setToolTip("在独立窗口中查看实时输出")
        self.popout_live_output_button.clicked.connect(self._show_live_output_window)
        live_toolbar.addWidget(self.popout_live_output_button)
        live_layout.addLayout(live_toolbar)
        live_layout.addWidget(self.live_output_view, 1)

        self.diagnostic_output_view = QPlainTextEdit()
        self.diagnostic_output_view.setReadOnly(True)
        self.diagnostic_output_view.setPlaceholderText("HyDE、查询改写和检索辅助输出将在这里显示")
        self.diagnostic_output_view.document().setMaximumBlockCount(5000)

        self.progress_tabs = QTabWidget()
        self.progress_tabs.addTab(section_tab, "章节进度")
        self._live_output_tab_index = self.progress_tabs.addTab(live_tab, "实时输出")
        self.progress_tabs.addTab(self.diagnostic_output_view, "调用明细")
        progress_layout.addWidget(self.progress_tabs)

        self.progress_group.setMinimumHeight(240)
        self.progress_scroll = QScrollArea()
        self.progress_scroll.setWidgetResizable(True)
        self.progress_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.progress_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.progress_scroll.setWidget(self.progress_group)
        self.progress_scroll.setMinimumHeight(90)

        # ── Results panel ──
        self.results_group = QGroupBox("写作结果")
        self.results_group.setObjectName("resultsGroup")
        results_layout = QVBoxLayout(self.results_group)
        results_layout.setSpacing(8)

        self.result_tabs = QTabWidget()
        self._draft_view = QPlainTextEdit()
        self._draft_view.setPlaceholderText("最终稿将在此显示…")
        self.result_tabs.addTab(self._draft_view, "论文")

        self._outline_view = QPlainTextEdit()
        self._outline_view.setPlaceholderText("提纲将在此显示…")
        self.result_tabs.addTab(self._outline_view, "提纲")

        self._ref_view = QPlainTextEdit()
        self._ref_view.setReadOnly(True)
        self._ref_view.setPlaceholderText("参考文献将在此显示…")
        self.result_tabs.addTab(self._ref_view, "参考文献")

        self._eval_view = QPlainTextEdit()
        self._eval_view.setReadOnly(True)
        self._eval_view.setPlaceholderText("阶段 7 评估结果将在此显示…")
        self.result_tabs.addTab(self._eval_view, "评估")

        results_layout.addWidget(self.result_tabs, 1)
        save_row = QHBoxLayout()
        save_row.addStretch(1)
        self.save_draft_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "保存人工编辑稿",
        )
        self.save_draft_button.clicked.connect(self._save_draft)
        save_row.addWidget(self.save_draft_button)
        results_layout.addLayout(save_row)
        self.results_group.setMinimumHeight(150)

        self.workspace_tabs = QTabWidget()
        self._config_workspace_index = self.workspace_tabs.addTab(
            self.config_scroll,
            "配置",
        )
        self._progress_workspace_index = self.workspace_tabs.addTab(
            self.progress_scroll,
            "章节与输出",
        )
        self._results_workspace_index = self.workspace_tabs.addTab(
            self.results_group,
            "稿件",
        )
        self.main_splitter.addWidget(self.workspace_tabs)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 5)
        self.main_splitter.setSizes([280, 680])

        # ── Initial load ──
        self.refresh_projects()
        self._reload_personas()
        self._reload_documents()
        self._update_source_summary()

    # ── Local project/task loading ─────────────────────────────

    def _apply_quality_preset(self) -> None:
        preset = str(self.quality_preset_combo.currentData() or "strict")
        custom = preset == "custom"
        if not custom:
            options = GenerationOptions.from_preset(
                preset,
                target_length_chars=max(500, self._resolved_target_length()),
            )
            values = (
                options.fact_verification,
                options.section_polish,
                options.section_drift_check,
                options.term_review,
                options.structure_review,
                options.global_polish,
                options.global_drift_check,
            )
            self._updating_quality_controls = True
            try:
                for checkbox, checked in zip(self._quality_checkboxes, values, strict=True):
                    checkbox.setChecked(checked)
            finally:
                self._updating_quality_controls = False
        for checkbox in self._quality_checkboxes:
            checkbox.setEnabled(custom)

    def _mark_quality_custom(self) -> None:
        if self._updating_quality_controls:
            return
        index = self.quality_preset_combo.findData("custom")
        if index >= 0 and self.quality_preset_combo.currentIndex() != index:
            self.quality_preset_combo.setCurrentIndex(index)

    def _resolved_target_length(self) -> int:
        explicit = self.target_length_spin.value()
        if explicit:
            return explicit
        match = re.search(r"(?<!\d)(\d{3,6})\s*(?:字|字符)", self.task_input.toPlainText())
        return int(match.group(1)) if match else 5000

    def _generation_options(self) -> GenerationOptions:
        target = self._resolved_target_length()
        preset = str(self.quality_preset_combo.currentData() or "strict")
        if preset != "custom":
            return GenerationOptions.from_preset(
                preset,
                target_length_chars=target,
            )
        return GenerationOptions(
            preset="custom",
            target_length_chars=target,
            fact_verification=self.fact_verification_checkbox.isChecked(),
            section_polish=self.section_polish_checkbox.isChecked(),
            section_drift_check=self.section_drift_checkbox.isChecked(),
            term_review=self.term_review_checkbox.isChecked(),
            structure_review=self.structure_review_checkbox.isChecked(),
            global_polish=self.global_polish_checkbox.isChecked(),
            global_drift_check=self.global_drift_checkbox.isChecked(),
        )

    def _set_generation_options(self, raw: object) -> None:
        options = GenerationOptions.model_validate(raw or {})
        self.target_length_spin.setValue(options.target_length_chars)
        index = self.quality_preset_combo.findData(options.preset)
        self.quality_preset_combo.setCurrentIndex(max(0, index))
        if options.preset == "custom":
            values = (
                options.fact_verification,
                options.section_polish,
                options.section_drift_check,
                options.term_review,
                options.structure_review,
                options.global_polish,
                options.global_drift_check,
            )
            self._updating_quality_controls = True
            try:
                for checkbox, checked in zip(self._quality_checkboxes, values, strict=True):
                    checkbox.setChecked(checked)
            finally:
                self._updating_quality_controls = False
            for checkbox in self._quality_checkboxes:
                checkbox.setEnabled(True)

    def refresh_projects(self) -> None:
        current = self.project_combo.currentData()
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for project in self._list_projects() or []:
            self.project_combo.addItem(
                str(project.get("title", "未命名项目")),
                str(project.get("project_id", "")),
            )
        if current:
            index = self.project_combo.findData(current)
            if index >= 0:
                self.project_combo.setCurrentIndex(index)
        self.project_combo.blockSignals(False)
        self._reload_task_history()

    def _reload_documents(self) -> None:
        selected = self._selected_doc_ids()
        self.document_list.blockSignals(True)
        self.document_list.clear()
        for document in self._list_documents() or []:
            item = QListWidgetItem(
                str(document.get("title") or document.get("filename") or "未命名文档")
            )
            item.setData(Qt.ItemDataRole.UserRole, str(document.get("doc_id", "")))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if item.data(Qt.ItemDataRole.UserRole) in selected
                else Qt.CheckState.Unchecked
            )
            self.document_list.addItem(item)
        self.document_list.blockSignals(False)
        self._update_source_summary()

    def _selected_doc_ids(self) -> set[str]:
        return {
            str(self.document_list.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.document_list.count())
            if self.document_list.item(index).checkState() == Qt.CheckState.Checked
        }

    def _source_preview(self) -> dict[str, int]:
        """Return the same source counts used by the generation policy."""

        selected = self._selected_doc_ids()
        fallback = {
            "selected_count": len(selected),
            "isolated_count": 0,
            "usable_count": len(selected),
        }
        persona_id = self._selected_persona_id()
        if not persona_id or self._preview_source_selection is None:
            return fallback
        explicitly_allowed = selected if self.allow_persona_sources.isChecked() else set()
        return self._preview_source_selection(persona_id, selected, explicitly_allowed)

    def _update_source_summary(self, *_args: object) -> None:
        """Refresh preflight counts without starting a background task."""

        try:
            preview = self._source_preview()
        except Exception as exc:
            self.source_summary_label.setText(f"来源统计不可用：{exc}")
            self.source_summary_label.setObjectName("statusError")
            self.source_summary_label.style().unpolish(self.source_summary_label)
            self.source_summary_label.style().polish(self.source_summary_label)
            if self._preview_source_selection is not None and self._task_id is None:
                self.start_button.setEnabled(False)
            return
        self.source_summary_label.setText(
            f"已选 {preview['selected_count']} 篇 · "
            f"隔离 {preview['isolated_count']} 篇 · "
            f"实际可用 {preview['usable_count']} 篇"
        )
        object_name = (
            "statusError"
            if preview["selected_count"] and not preview["usable_count"]
            else "mutedText"
        )
        self.source_summary_label.setObjectName(object_name)
        self.source_summary_label.style().unpolish(self.source_summary_label)
        self.source_summary_label.style().polish(self.source_summary_label)
        if self._preview_source_selection is not None and self._task_id is None:
            self.start_button.setEnabled(
                self._run_writing_pipeline is not None and preview["usable_count"] > 0
            )

    def _reload_task_history(self) -> None:
        project_id = self.project_combo.currentData()
        records = self._list_writing_tasks(str(project_id)) if project_id else []
        self._task_records = records
        self.task_table.setRowCount(len(records))
        for row, record in enumerate(records):
            checkbox = QTableWidgetItem()
            checkbox.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            checkbox.setCheckState(Qt.CheckState.Unchecked)
            checkbox.setData(Qt.ItemDataRole.UserRole, record.get("task_id"))
            self.task_table.setItem(row, 0, checkbox)
            self.task_table.setItem(row, 1, QTableWidgetItem(str(record.get("title", ""))))
            status = str(record.get("status", ""))
            self.task_table.setItem(
                row,
                2,
                QTableWidgetItem(_TASK_STATUS_LABELS.get(status, status)),
            )
            self.task_table.setItem(
                row,
                3,
                QTableWidgetItem(format_china_datetime(record.get("updated_at"))),
            )
            error = str(record.get("error") or "")
            if error:
                for column in range(self.task_table.columnCount()):
                    self.task_table.item(row, column).setToolTip(error)
        self._update_task_error_banner()

    def _update_task_error_banner(self) -> None:
        task_id = self._selected_history_task_id()
        record = next(
            (item for item in self._task_records if str(item.get("task_id")) == task_id),
            None,
        )
        error = str(record.get("error") or "") if record else ""
        self.task_error_label.setPlainText(f"失败详情：{error}" if error else "")
        self.task_error_label.setVisible(bool(error))

    def _selected_history_task_id(self) -> str | None:
        rows = sorted({item.row() for item in self.task_table.selectedItems()})
        if not rows:
            rows = [
                row
                for row in range(self.task_table.rowCount())
                if self.task_table.item(row, 0).checkState() == Qt.CheckState.Checked
            ]
        if not rows:
            return None
        return str(self.task_table.item(rows[0], 0).data(Qt.ItemDataRole.UserRole))

    def _load_selected_task(self) -> dict[str, object] | None:
        task_id = self._selected_history_task_id()
        if not task_id or self._load_writing_task is None:
            return None
        record = self._load_writing_task(task_id)
        if record is None:
            self._show_message("任务不存在", 4000)
            return None
        self._writing_task_id = task_id
        self.title_input.setText(str(record.get("title", "")))
        self.task_input.setPlainText(str(record.get("task_description", "")))
        self.domain_input.setPlainText(str(record.get("domain", "")))
        self._set_generation_options(record.get("generation_options", {}))
        persona_index = self.persona_combo.findData(record.get("persona_id"))
        if persona_index >= 0:
            self.persona_combo.setCurrentIndex(persona_index)
        selected_docs = set(record.get("selected_doc_ids", set()))
        self.allow_persona_sources.setChecked(bool(record.get("allowed_persona_doc_ids", set())))
        for index in range(self.document_list.count()):
            item = self.document_list.item(index)
            item.setCheckState(
                Qt.CheckState.Checked
                if item.data(Qt.ItemDataRole.UserRole) in selected_docs
                else Qt.CheckState.Unchecked
            )
        state = record.get("state")
        if isinstance(state, dict):
            self._last_result = state
            self._display_results(state)
            self.workspace_tabs.setCurrentIndex(self._progress_workspace_index)
            edited = record.get("edited_draft_text")
            if edited:
                self._draft_view.setPlainText(str(edited))
            edited_outline = record.get("edited_outline_text")
            if edited_outline:
                self._outline_view.setPlainText(str(edited_outline))
            self.eval_button.setEnabled(
                state.get("status") == PIPELINE_STATUS_DONE
                and self._evaluate_generation is not None
            )
        return record

    def _resume_selected_task(self) -> None:
        if self._task_id is not None:
            return
        record = self._load_selected_task()
        if record is None:
            self._show_message("请先选择要继续的任务", 4000)
            return
        if record.get("status") == PIPELINE_STATUS_DONE:
            self._show_message("该任务已经完成，已载入稿件", 4000)
            return
        try:
            source_preview = self._source_preview()
        except Exception as exc:
            self._show_message(f"无法检查事实来源：{exc}", 6000)
            return
        if source_preview["usable_count"] == 0:
            self._show_message(
                "该任务没有实际可用的事实语料，请调整来源选择后新建任务",
                7000,
            )
            return
        self._launch_pipeline(record, resume=True)

    def _delete_selected_tasks(self) -> None:
        if self._delete_writing_tasks is None:
            return
        identifiers = {
            str(self.task_table.item(row, 0).data(Qt.ItemDataRole.UserRole))
            for row in range(self.task_table.rowCount())
            if self.task_table.item(row, 0).checkState() == Qt.CheckState.Checked
        }
        identifiers.update(
            str(self.task_table.item(row, 0).data(Qt.ItemDataRole.UserRole))
            for row in {item.row() for item in self.task_table.selectedItems()}
        )
        if not identifiers:
            self._show_message("请勾选或选择要删除的任务", 4000)
            return
        removed = self._delete_writing_tasks(identifiers)
        self._reload_task_history()
        self._show_message(f"已删除 {removed} 个任务", 4000)

    def _save_draft(self) -> None:
        if not self._writing_task_id or self._save_edited_draft is None:
            self._show_message("当前没有可保存的任务", 4000)
            return
        self._save_edited_draft(
            self._writing_task_id,
            self._draft_view.toPlainText(),
            self._outline_view.toPlainText(),
        )
        self._reload_task_history()
        self._show_message("人工编辑稿已保存", 4000)

    # ── Persona loading ────────────────────────────────────────

    def _reload_personas(self) -> None:
        """Refresh the persona dropdown from the database."""

        self._personas = self._list_personas() or []
        current = self.persona_combo.currentText()
        self.persona_combo.clear()
        self.persona_combo.addItem("— 请选择 —", None)
        for p in self._personas:
            name = str(p.get("name", p.get("persona_id", "?")))
            pid = str(p.get("persona_id", ""))
            mode = str(p.get("mode", "person"))
            label = f"{name} ({mode})" if mode else name
            self.persona_combo.addItem(label, pid)

        idx = self.persona_combo.findText(current)
        if idx >= 0:
            self.persona_combo.setCurrentIndex(idx)

    def _selected_persona_id(self) -> str | None:
        """Return the selected persona ID or None."""

        data = self.persona_combo.currentData()
        return str(data) if data else None

    # ── Writing pipeline control ───────────────────────────────

    def _start_writing(self) -> None:
        if self._task_id is not None:
            return

        persona_id = self._selected_persona_id()
        if not persona_id:
            self._show_message("请先选择一个作者档案", 5000)
            return

        task_text = self.task_input.toPlainText().strip()
        if not task_text:
            self._show_message("请输入写作任务描述", 5000)
            return
        project_id = self.project_combo.currentData()
        if not project_id:
            self._show_message("请先创建或选择一个项目", 5000)
            return
        selected_doc_ids = self._selected_doc_ids()
        if not selected_doc_ids:
            self._show_message("请至少勾选一篇事实语料", 5000)
            return
        try:
            source_preview = self._source_preview()
        except Exception as exc:
            self._show_message(f"无法检查事实来源：{exc}", 6000)
            return
        if source_preview["usable_count"] == 0:
            self._show_message(
                "所选文档均被来源隔离，请增加事实语料或明确允许复用目标语料",
                7000,
            )
            return
        if self._create_writing_task is None:
            self._show_message("任务持久化服务不可用", 5000)
            return
        domain = self.domain_input.toPlainText().strip()
        generation_options = self._generation_options()
        task_id = self._create_writing_task(
            project_id=str(project_id),
            persona_id=persona_id,
            title=self.title_input.text().strip() or task_text[:60],
            task_description=task_text,
            domain=domain,
            selected_doc_ids=selected_doc_ids,
            allowed_persona_doc_ids=(
                selected_doc_ids if self.allow_persona_sources.isChecked() else set()
            ),
            generation_options=generation_options.model_dump(mode="json"),
        )
        record = {
            "task_id": task_id,
            "persona_id": persona_id,
            "task_description": task_text,
            "domain": domain,
            "selected_doc_ids": selected_doc_ids,
            "allowed_persona_doc_ids": (
                selected_doc_ids if self.allow_persona_sources.isChecked() else set()
            ),
            "generation_options": generation_options.model_dump(mode="json"),
        }
        self._reload_task_history()
        self._launch_pipeline(record, resume=False)

    def _launch_pipeline(self, record: dict[str, object], *, resume: bool) -> None:
        if self._run_writing_pipeline is None:
            return
        self._writing_task_id = str(record["task_id"])
        self._history_refreshed_for_run = False
        self._set_running(True)
        if not resume:
            self._last_result = None
            self._draft_view.clear()
            self._outline_view.clear()
            self._ref_view.clear()
            self._eval_view.clear()
        self.workspace_tabs.setCurrentIndex(self._progress_workspace_index)
        resume_state = record.get("state") if resume else None
        self.progress_bar.setValue(
            self._resume_progress(resume_state) if isinstance(resume_state, dict) else 0
        )
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("")
        self.status_label.setText("正在准备流水线…")
        if not resume:
            self.section_table.setRowCount(0)
        self.live_output_view.clear()
        self.diagnostic_output_view.clear()
        self.progress_tabs.setCurrentIndex(0)
        self._start_run_clock()
        self.eval_button.setEnabled(False)

        def task(context: TaskContext) -> dict[str, Any]:
            return self._run_writing_pipeline(
                task_id=self._writing_task_id,
                persona_id=str(record["persona_id"]),
                task_description=str(record["task_description"]),
                domain=str(record.get("domain", "")),
                selected_doc_ids=set(record.get("selected_doc_ids", set())),
                explicitly_allowed_persona_doc_ids=set(
                    record.get("allowed_persona_doc_ids", set())
                ),
                generation_options=dict(record.get("generation_options", {})),
                resume=resume,
                context=context,
            )

        self._task_id = self._tasks.start(
            task,
            on_success=self._pipeline_succeeded,
            on_error=self._pipeline_failed,
            on_progress=self._pipeline_progressed,
            on_stream=self._pipeline_streamed,
        )

    @staticmethod
    def _resume_progress(state: dict[str, Any]) -> int:
        sections = state.get("sections", [])
        if not sections:
            return 2
        fractions = {
            "pending": 0.0,
            "drafted": 0.34,
            "verified": 0.67,
            "revising": 0.34,
            "polished": 1.0,
        }
        completed = sum(
            fractions.get(str(section.get("status", "pending")), 0.0)
            for section in sections
        )
        return 14 + round(68 * completed / len(sections))

    def _stop_writing(self) -> None:
        if self._task_id is not None:
            self._tasks.cancel(self._task_id)
            self.stop_button.setEnabled(False)
            self.status_label.setText("正在安全停止…")
            self._show_message("已请求停止，将在当前调用结束后保存断点", 5000)

    def _pipeline_progressed(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        if message:
            self._set_current_step(message)
            self.status_label.setText(message)
            self._show_current_section_activity(message)
        if not self._history_refreshed_for_run:
            self._reload_task_history()
            self._history_refreshed_for_run = True

    def _pipeline_streamed(self, kind: str, text: str) -> None:
        """Display public response deltas while using reasoning only as a heartbeat."""

        event_kind, separator, stream_label = kind.partition("::")
        if event_kind == "pipeline_state" and text:
            try:
                self._display_pipeline_state(json.loads(text))
            except (json.JSONDecodeError, TypeError):
                logger.warning("忽略无法解析的流水线状态事件")
            self._update_elapsed_display()
            return

        self._last_stream_at = time.monotonic()
        stage = stream_label if separator else (self._current_step or "模型输出")
        output_view = self._stream_view_for_stage(stage)
        if event_kind == "status" and text:
            if output_view is self.live_output_view:
                self._activate_live_output_once()
            self._ensure_stream_stage(stage, output_view)
            if "中断，正在重试" in text:
                self._discard_stream_attempt(output_view)
            self._append_stream_text(output_view, f"\n[系统] {text}\n")
            self._set_attempt_start(output_view)
            self._update_elapsed_display()
            return
        if event_kind != "content" or not text:
            self._update_elapsed_display()
            return
        if output_view is self.live_output_view:
            self._activate_live_output_once()
        self._ensure_stream_stage(stage, output_view)
        self._append_stream_text(output_view, text)
        self._update_elapsed_display()

    def _stream_view_for_stage(self, stage: str) -> QPlainTextEdit:
        diagnostic_markers = ("HyDE", "查询改写", "证据预取", "框架证据")
        return (
            self.diagnostic_output_view
            if any(marker in stage for marker in diagnostic_markers)
            else self.live_output_view
        )

    def _ensure_stream_stage(self, stage: str, view: QPlainTextEdit) -> None:
        current = (
            self._diagnostic_stream_stage
            if view is self.diagnostic_output_view
            else self._stream_stage
        )
        if stage == current:
            return
        if view.toPlainText():
            self._append_stream_text(view, "\n\n")
        self._append_stream_text(view, f"===== {stage} =====\n")
        if view is self.diagnostic_output_view:
            self._diagnostic_stream_stage = stage
        else:
            self._stream_stage = stage
        self._set_attempt_start(view)

    def _append_stream_text(self, view: QPlainTextEdit, text: str) -> None:
        view.moveCursor(QTextCursor.MoveOperation.End)
        view.insertPlainText(text)
        if view is self.live_output_view:
            self._scroll_live_output_if_enabled()
        else:
            view.ensureCursorVisible()

    def _set_attempt_start(self, view: QPlainTextEdit) -> None:
        position = max(0, view.document().characterCount() - 1)
        if view is self.diagnostic_output_view:
            self._diagnostic_attempt_start = position
        else:
            self._live_attempt_start = position

    def _discard_stream_attempt(self, view: QPlainTextEdit) -> None:
        start = (
            self._diagnostic_attempt_start
            if view is self.diagnostic_output_view
            else self._live_attempt_start
        )
        cursor = QTextCursor(view.document())
        cursor.setPosition(min(start, max(0, view.document().characterCount() - 1)))
        cursor.movePosition(
            QTextCursor.MoveOperation.End,
            QTextCursor.MoveMode.KeepAnchor,
        )
        cursor.removeSelectedText()
        view.setTextCursor(cursor)

    def _activate_live_output_once(self) -> None:
        if self._stream_auto_switched:
            return
        self.progress_tabs.setCurrentIndex(self._live_output_tab_index)
        self._stream_auto_switched = True
        QTimer.singleShot(0, self._reveal_live_output)

    def _reveal_live_output(self) -> None:
        scrollbar = self.progress_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _scroll_live_output_if_enabled(self) -> None:
        if self.auto_scroll_checkbox.isChecked():
            self.live_output_view.moveCursor(QTextCursor.MoveOperation.End)
            self.live_output_view.ensureCursorVisible()

    def _copy_live_output(self) -> None:
        QApplication.clipboard().setText(self.live_output_view.toPlainText())

    def _show_live_output_window(self) -> None:
        if self._live_output_window is None:
            self._live_output_window = LiveOutputWindow(
                self.live_output_view.document(),
                auto_scroll=self.auto_scroll_checkbox.isChecked(),
                parent=self,
            )
            self._live_output_window.auto_scroll_changed.connect(
                self.auto_scroll_checkbox.setChecked
            )
        self._live_output_window.show_output()

    def _sync_live_output_auto_scroll(self, enabled: bool) -> None:
        if self._live_output_window is not None:
            self._live_output_window.set_auto_scroll(enabled)
        if enabled:
            self._scroll_live_output_if_enabled()

    def _theme_icon(self, name: str, fallback: QStyle.StandardPixmap) -> QIcon:
        icon = QIcon.fromTheme(name)
        return icon if not icon.isNull() else self.style().standardIcon(fallback)

    def _display_pipeline_state(self, state: dict[str, Any]) -> None:
        sections = state.get("sections", [])
        if not isinstance(sections, list):
            return
        current_index = state.get("current_section_index")
        self._current_section_index = current_index if isinstance(current_index, int) else None
        self._set_section_rows(sections, current_index)
        completed_parts: list[str] = []
        for section in sections:
            text = str(section.get("polished_text") or "").strip()
            if not text:
                continue
            heading = str(section.get("heading") or "").strip()
            if heading:
                completed_parts.append(heading)
            completed_parts.extend([text, ""])
        if completed_parts:
            self._draft_view.setPlainText("\n".join(completed_parts).strip())

    def _show_current_section_activity(self, message: str) -> None:
        row = self._current_section_index
        if row is None or not 0 <= row < self.section_table.rowCount():
            return
        status = None
        if "修订" in message:
            status = "▶ 修订"
        elif "起草" in message:
            status = "▶ 起草"
        elif "核对" in message or "核验" in message:
            status = "▶ 核对"
        elif "打磨" in message:
            status = "▶ 打磨"
        if status:
            self.section_table.setItem(row, 2, QTableWidgetItem(status))

    def _set_section_rows(
        self,
        sections: list[dict[str, Any]],
        current_index: object = None,
    ) -> None:
        self.section_table.setRowCount(len(sections))
        for row, section in enumerate(sections):
            values = (
                str(section.get("section_id", "")),
                str(section.get("heading", "")),
                _SECTION_LABELS.get(
                    str(section.get("status", "")),
                    str(section.get("status", "")),
                ),
                str(section.get("revision_count", 0)),
                self._format_section_elapsed(section.get("elapsed_seconds")),
            )
            for column, value in enumerate(values):
                self.section_table.setItem(row, column, QTableWidgetItem(value))
        if isinstance(current_index, int) and 0 <= current_index < len(sections):
            self.section_table.selectRow(current_index)

    @staticmethod
    def _format_section_elapsed(value: object) -> str:
        if not isinstance(value, (int, float)):
            return "—"
        seconds = max(0, round(value))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _pipeline_succeeded(self, result: Any) -> None:
        if not isinstance(result, dict):
            self._show_message("写作流水线返回了意外的结果类型", 8000)
            self._finish_writing()
            return

        self._last_result = result
        status = result.get("status", "")
        error = result.get("error")

        if status == PIPELINE_STATUS_ERROR:
            self._show_message(f"写作流水线出错: {error}", 8000)
            self._show_pipeline_failure(str(error or "未知错误"))
            self._reload_task_history()
            self._finish_writing()
            return

        # Display results
        self._display_results(result)
        self.workspace_tabs.setCurrentIndex(self._results_workspace_index)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("")
        quality_status = self._result_quality_status(result)
        if quality_status == "unverified_draft":
            self.status_label.setText("快速草稿完成 · 尚未通过完整质量核验")
            self._show_message("未核验草稿已生成，可补做质量步骤后定稿", 7000)
        else:
            self.status_label.setText("写作完成 ✓")
            self._show_message("写作流水线已完成全篇稿件的生成", 6000)
        self.eval_button.setEnabled(self._evaluate_generation is not None)
        self._reload_task_history()
        self._finish_writing()

    def _pipeline_failed(self, message: str) -> None:
        self._show_message(f"写作流水线失败: {message}", 8000)
        self._show_pipeline_failure(message)
        self._reload_task_history()
        self._finish_writing()

    def _show_pipeline_failure(self, message: str) -> None:
        """Render a terminal failure without making it resemble 100% success."""

        if self.progress_bar.value() >= 100:
            self.progress_bar.setValue(99)
        self.progress_bar.setFormat("失败 · %p%")
        self.progress_bar.setStyleSheet(
            "QProgressBar::chunk { background: #b42318; border-radius: 4px; }"
        )
        self.status_label.setText(f"失败: {message}")

    def _finish_writing(self) -> None:
        self._task_id = None
        self._stop_run_clock()
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running and self._run_writing_pipeline is not None)
        self.stop_button.setEnabled(running)
        self.persona_combo.setEnabled(not running)
        self.project_combo.setEnabled(not running)
        self.title_input.setEnabled(not running)
        self.document_list.setEnabled(not running)
        self.allow_persona_sources.setEnabled(not running)
        self.task_input.setEnabled(not running)
        self.domain_input.setEnabled(not running)
        self.target_length_spin.setEnabled(not running)
        self.quality_preset_combo.setEnabled(not running)
        if not running:
            self._apply_quality_preset()
        elif self.quality_preset_combo.currentData() == "custom":
            for checkbox in self._quality_checkboxes:
                checkbox.setEnabled(False)
        self.refresh_button.setEnabled(not running)
        if not running:
            self._update_source_summary()

    # ── Results display ────────────────────────────────────────

    @staticmethod
    def _result_quality_status(state: dict[str, Any]) -> str:
        raw = state.get("final_draft_json")
        if not raw:
            return "verified_final"
        try:
            return str(json.loads(raw).get("quality_status", "verified_final"))
        except (json.JSONDecodeError, TypeError):
            return "verified_final"

    def _display_results(self, state: dict[str, Any]) -> None:
        """Populate the result tabs from the final WritingState."""

        sections_state = state.get("sections", [])
        if not isinstance(sections_state, list):
            sections_state = []
        current_index = state.get("current_section_index")
        self._current_section_index = (
            current_index if isinstance(current_index, int) else None
        )
        self._set_section_rows(
            sections_state,
            current_index,
        )

        # Draft
        final_draft_json = state.get("final_draft_json")
        if final_draft_json:
            try:
                draft_data = json.loads(final_draft_json)
                sections = draft_data.get("sections", [])
                lines: list[str] = []
                if draft_data.get("quality_status") == "unverified_draft":
                    lines.extend(["[未核验草稿]", ""])
                title = draft_data.get("title", "")
                if title:
                    lines.extend([str(title), ""])
                for sec in sections:
                    text = sec.get("polished_text", "")
                    if text:
                        heading = sec.get("heading", "")
                        if heading:
                            lines.append(str(heading))
                        lines.append(text)
                        lines.append("")
                self._draft_view.setPlainText("\n".join(lines).strip())
            except (json.JSONDecodeError, TypeError) as exc:
                self._draft_view.setPlainText(f"[解析稿件出错: {exc}]")

        thesis_json = state.get("thesis_json")
        outline_json = state.get("outline_json")

        # Outline
        outline_parts: list[str] = []
        if thesis_json:
            try:
                thesis = json.loads(thesis_json)
                t = thesis.get("thesis_text", "")
                angle = thesis.get("angle", "")
                outline_parts.append(f"论点: {t}")
                if angle:
                    outline_parts.append(f"角度: {angle}")
                outline_parts.append("")
            except (json.JSONDecodeError, TypeError):
                pass

        if outline_json:
            try:
                outline = json.loads(outline_json)
                root_nodes = outline.get("root_nodes", [])
                for node in root_nodes:
                    outline_parts.append(self._format_node(node, 0))
                term_registry = outline.get("term_registry", {})
                if term_registry:
                    outline_parts.append("")
                    outline_parts.append("术语登记表:")
                    for term, defn in term_registry.items():
                        outline_parts.append(f"  {term}: {defn}")
            except (json.JSONDecodeError, TypeError) as exc:
                outline_parts.append(f"[解析提纲出错: {exc}]")

        self._outline_view.setPlainText("\n".join(outline_parts))

        # References
        ref_json = state.get("reference_list_json")
        if ref_json:
            try:
                refs = json.loads(ref_json)
                items = refs.get("items", [])
                ref_lines = [f"参考文献 (样式: {refs.get('style', 'gb-t-7714')})", ""]
                for i, item in enumerate(items, 1):
                    ref_lines.append(f"[{i}] {item.get('citation_text', '')}")
                self._ref_view.setPlainText("\n".join(ref_lines))
            except (json.JSONDecodeError, TypeError) as exc:
                self._ref_view.setPlainText(f"[解析参考文献出错: {exc}]")

        self.results_group.show()
        self.result_tabs.setCurrentIndex(0)

    @staticmethod
    def _format_node(node: dict, depth: int) -> str:
        """Recursively format an outline node for display."""

        prefix = "  " * depth + "• "
        heading = node.get("heading", "?")
        purpose = node.get("rhetorical_purpose", "")
        line = f"{prefix}{heading}"
        if purpose:
            line += f"  — {purpose}"
        children = node.get("children", [])
        child_lines = [WritingTaskPage._format_node(c, depth + 1) for c in children]
        return "\n".join([line] + child_lines)

    # ── Evaluation (Stage 7) ───────────────────────────────────

    def _run_evaluation(self) -> None:
        if self._task_id is not None or self._evaluate_generation is None:
            return
        if self._last_result is None:
            self._show_message("没有可评估的稿件", 3000)
            return

        final_draft_json = self._last_result.get("final_draft_json")
        thesis_json = self._last_result.get("thesis_json")
        if not final_draft_json or not thesis_json:
            self._show_message("稿件或论点数据不完整，无法评估", 5000)
            return

        self._set_running(True)
        self._start_run_clock()
        self.eval_button.setEnabled(False)
        self.status_label.setText("正在运行阶段 7 评估…")
        self.progress_bar.setValue(0)
        self._eval_view.clear()

        def task(context: TaskContext) -> Any:
            return self._evaluate_generation(
                thesis_json=thesis_json,
                draft_json=final_draft_json,
                context=self._last_result,
                task_context=context,
            )

        self._task_id = self._tasks.start(
            task,
            on_success=self._eval_succeeded,
            on_error=self._eval_failed,
            on_progress=self._eval_progressed,
            on_stream=self._pipeline_streamed,
        )

    def _eval_progressed(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        if message:
            label = f"评估: {message}"
            self._set_current_step(label)
            self.status_label.setText(label)

    def _eval_succeeded(self, result: Any) -> None:
        if isinstance(result, dict) and result.get("error"):
            self._eval_view.setPlainText(f"评估失败: {result['error']}")
            self.status_label.setText("评估失败")
            self._finish_eval()
            return
        if result is None:
            self._eval_view.setPlainText("评估完成，未返回具体结果。")
        else:
            lines = ["== 阶段 7 — 评估结果 ==", ""]
            if isinstance(result, dict):
                traceability = result.get("traceability")
                if traceability is not None:
                    lines.append(f"引用可溯性: {traceability}")
                hallucination = result.get("hallucination_rate")
                if hallucination is not None:
                    lines.append(f"幻觉率: {hallucination}")
                faithfulness = result.get("faithfulness")
                if faithfulness is not None:
                    lines.append(f"忠实度 (Faithfulness): {faithfulness}")
                judge = result.get("judge")
                if judge is not None:
                    lines.append(f"裁判评分 (LLM-Judge): {judge}")
                rationale = result.get("judge_rationale")
                if rationale:
                    lines.append("")
                    lines.append(f"评语: {rationale}")
                injection = result.get("injection")
                if injection is not None:
                    lines.append("")
                    lines.append(f"注入检测: {injection}")
            else:
                lines.append(str(result))
            self._eval_view.setPlainText("\n".join(lines))

        self.status_label.setText("评估完成 ✓")
        self.progress_bar.setValue(100)
        self._show_message("阶段 7 评估完成", 6000)
        self.result_tabs.setCurrentIndex(3)
        self._finish_eval()

    def _eval_failed(self, message: str) -> None:
        self._eval_view.setPlainText(f"评估失败: {message}")
        self._finish_eval()

    def _finish_eval(self) -> None:
        self._task_id = None
        self._stop_run_clock()
        self._set_running(False)
        self.eval_button.setEnabled(self._last_result is not None)

    def _start_run_clock(self) -> None:
        now = time.monotonic()
        self._run_started_at = now
        self._step_started_at = now
        self._last_stream_at = None
        self._current_step = "正在准备流水线"
        self._stream_stage = ""
        self._diagnostic_stream_stage = ""
        self._live_attempt_start = 0
        self._diagnostic_attempt_start = 0
        self._stream_auto_switched = False
        self._elapsed_timer.start()
        self._update_elapsed_display()

    def _stop_run_clock(self) -> None:
        self._update_elapsed_display()
        self._elapsed_timer.stop()

    def _set_current_step(self, message: str) -> None:
        if message != self._current_step:
            self._current_step = message
            self._step_started_at = time.monotonic()
            self._update_elapsed_display()

    def _update_elapsed_display(self) -> None:
        now = time.monotonic()
        step_seconds = now - self._step_started_at if self._step_started_at is not None else 0
        run_seconds = now - self._run_started_at if self._run_started_at is not None else 0
        self.elapsed_label.setText(
            f"步骤耗时 {self._format_duration(step_seconds)} · "
            f"本次运行 {self._format_duration(run_seconds)}"
        )
        if self._last_stream_at is None:
            activity = "尚未收到模型流式输出"
        else:
            idle = max(0, round(now - self._last_stream_at))
            activity = f"模型最近活动：{idle} 秒前"
        self.activity_label.setText(activity)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    # ── Public refresh ─────────────────────────────────────────

    def refresh(self) -> None:
        """Called when the page becomes visible — reload persona list."""

        self.refresh_projects()
        self._reload_personas()
        self._reload_documents()
        self._reload_task_history()
