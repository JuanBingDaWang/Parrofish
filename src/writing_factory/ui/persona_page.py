"""Source selection, mode control, and background PersonaSpec distillation UI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from writing_factory.distill.models import PersonaMode
from writing_factory.ui.persona_editor import PersonaEditorWindow, PersonaLoader, PersonaSaver
from writing_factory.ui.workers import BackgroundTaskManager, TaskContext


class PersonaPage(QWidget):
    """Distill selected ready documents and list persisted profiles."""

    def __init__(
        self,
        tasks: BackgroundTaskManager,
        *,
        distill_persona: Callable[[str, PersonaMode, set[str], TaskContext], Any] | None,
        evaluate_persona: Callable[[str, TaskContext], Any] | None,
        list_sources: Callable[[], list[dict[str, object]]],
        list_personas: Callable[[], list[dict[str, object]]],
        delete_personas: Callable[[set[str], TaskContext], Any] | None,
        load_persona: PersonaLoader | None,
        save_persona: PersonaSaver | None,
        show_message: Callable[[str, int], None],
    ) -> None:
        super().__init__()
        self._tasks = tasks
        self._distill_persona = distill_persona
        self._evaluate_persona = evaluate_persona
        self._list_sources = list_sources
        self._list_personas = list_personas
        self._delete_personas = delete_personas
        self._load_persona = load_persona
        self._save_persona = save_persona
        self._show_message = show_message
        self._task_id: str | None = None
        self._editor_windows: dict[str, PersonaEditorWindow] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(14)

        toolbar = QHBoxLayout()
        heading = QLabel("作者档案")
        heading.setObjectName("pageTitle")
        name_label = QLabel("名称")
        self.name_input = QLineEdit()
        self.name_input.setMaximumWidth(240)
        self.name_input.setMaxLength(80)
        self.name_input.textChanged.connect(self._update_button)
        toolbar.addWidget(heading)
        toolbar.addStretch(1)
        toolbar.addWidget(name_label)
        toolbar.addWidget(self.name_input)

        self.mode_group = QButtonGroup(self)
        self.person_button = self._mode_button("人物", "person")
        self.topic_button = self._mode_button("主题", "topic")
        self.person_button.setChecked(True)
        toolbar.addWidget(self.person_button)
        toolbar.addWidget(self.topic_button)

        self.distill_button = QPushButton("蒸馏")
        self.distill_button.clicked.connect(self.start_distillation)
        self.distill_button.setEnabled(False)
        toolbar.addWidget(self.distill_button)
        self.evaluate_button = QPushButton("自检")
        self.evaluate_button.setToolTip("运行独立的 Nüwa 保真度自检")
        self.evaluate_button.clicked.connect(self.start_evaluation)
        self.evaluate_button.setEnabled(False)
        toolbar.addWidget(self.evaluate_button)
        layout.addLayout(toolbar)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedHeight(18)
        self.progress.hide()
        layout.addWidget(self.progress)

        source_label = QLabel("语料")
        source_label.setObjectName("sectionTitle")
        layout.addWidget(source_label)
        self.source_table = QTableWidget(0, 3)
        self.source_table.setHorizontalHeaderLabels(["使用", "文件", "切片"])
        self.source_table.setMaximumHeight(210)
        self.source_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.source_table.verticalHeader().setVisible(False)
        source_header = self.source_table.horizontalHeader()
        source_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        source_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        source_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.source_table.itemChanged.connect(self._update_button)
        layout.addWidget(self.source_table)

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
        profile_header_layout.addWidget(self.delete_button)
        layout.addLayout(profile_header_layout)
        self.profile_table = QTableWidget(0, 7)
        self.profile_table.setHorizontalHeaderLabels(
            ["选择", "名称", "模式", "状态", "心智模型", "自检", "调研日期"]
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
        for column in (2, 3, 4, 5, 6):
            profile_header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.profile_table, 1)
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
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(Qt.CheckState.Checked)
            use.setData(Qt.ItemDataRole.UserRole, source.get("doc_id"))
            self.source_table.setItem(row, 0, use)
            self.source_table.setItem(row, 1, QTableWidgetItem(str(source.get("filename", ""))))
            self.source_table.setItem(row, 2, QTableWidgetItem(str(source.get("chunk_count", 0))))
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
            self.profile_table.setItem(row, 0, select)
            values = (
                str(profile.get("name", "")),
                "人物" if profile.get("mode") == "person" else "主题",
                self._status_label(str(profile.get("status", ""))),
                str(profile.get("model_count", 0)),
                self._score_label(profile.get("fidelity_score")),
                str(profile.get("research_date", ""))[:10],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                self.profile_table.setItem(row, column + 1, item)
        self.profile_table.blockSignals(False)
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
        self._set_running(True)
        self._show_message("准备蒸馏", 0)

        def task(context: TaskContext):
            return self._distill_persona(name, mode, doc_ids, context)

        self._task_id = self._tasks.start(
            task,
            on_success=self._succeeded,
            on_error=self._failed,
            on_progress=self._progressed,
        )

    def start_evaluation(self) -> None:
        """Run the selected ready profile through the paid independent self-check."""

        if self._task_id is not None or self._evaluate_persona is None:
            return
        persona_id = self._selected_persona_id()
        if persona_id is None:
            return
        self._set_running(True)
        self._show_message("准备独立自检", 0)

        def task(context: TaskContext):
            return self._evaluate_persona(persona_id, context)

        self._task_id = self._tasks.start(
            task,
            on_success=self._evaluation_succeeded,
            on_error=self._failed,
            on_progress=self._progressed,
        )

    def start_deletion(self) -> None:
        """在后台删除单个或多个选中档案，不弹确认框。"""

        if self._task_id is not None or self._delete_personas is None:
            return
        persona_ids = self._selected_persona_ids()
        if not persona_ids:
            return
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
        selected: set[str] = set()
        for row in range(self.source_table.rowCount()):
            item = self.source_table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                value = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(value, str):
                    selected.add(value)
        return selected

    def _selected_mode(self) -> PersonaMode:
        return "topic" if self.topic_button.isChecked() else "person"

    def _selected_persona_id(self) -> str | None:
        selected = self._selected_persona_rows()
        if len(selected) != 1:
            return None
        item = self.profile_table.item(selected[0], 0)
        if item is None or item.data(Qt.ItemDataRole.UserRole + 1) != "ready":
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
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
        enabled = (
            self._distill_persona is not None
            and self._task_id is None
            and bool(self.name_input.text().strip())
            and bool(self._selected_doc_ids())
        )
        self.distill_button.setEnabled(enabled)
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

    def _progressed(self, percent: int, message: str) -> None:
        self.progress.setValue(percent)
        if message:
            self._show_message(f"{message} · {percent}%", 0)

    def _succeeded(self, result: Any) -> None:
        count = len(getattr(getattr(result, "persona", None), "mental_models", []))
        self._show_message(f"蒸馏完成 · {count} 个心智模型", 6000)
        self._set_running(False)
        self.refresh()

    def _failed(self, message: str) -> None:
        self._show_message(message, 8000)
        self._set_running(False)
        self.refresh()

    def _evaluation_succeeded(self, result: Any) -> None:
        score = getattr(result, "total", None)
        label = str(score) if isinstance(score, int) else "完成"
        self._show_message(f"独立自检完成 · {label}/100", 6000)
        self._set_running(False)
        self.refresh()

    def _set_running(self, running: bool) -> None:
        self.progress.setValue(0)
        self.progress.setVisible(running)
        self.name_input.setEnabled(not running)
        self.person_button.setEnabled(not running)
        self.topic_button.setEnabled(not running)
        self.source_table.setEnabled(not running)
        self.profile_table.setEnabled(not running)
        self.distill_button.setEnabled(not running)
        self.evaluate_button.setEnabled(not running)
        self.delete_button.setEnabled(False)
        if not running:
            self._task_id = None
            self._update_button()

    @staticmethod
    def _status_label(status: str) -> str:
        return {
            "ready": "可用",
            "mapping": "提取中",
            "reducing": "归并中",
            "validating": "校验中",
            "failed": "失败",
        }.get(status, status)

    @staticmethod
    def _score_label(score: object) -> str:
        return "未检" if score is None else f"{score}/100"
