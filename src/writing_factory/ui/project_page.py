"""Project management page for grouping persistent writing tasks."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from writing_factory.ui.time_format import format_china_datetime


class ProjectPage(QWidget):
    """Create, edit, select, and delete local writing projects."""

    projects_changed = pyqtSignal()

    def __init__(
        self,
        *,
        list_projects: Callable[[], list[dict[str, object]]],
        create_project: Callable[[str, str], str],
        update_project: Callable[[str, str, str], None],
        delete_projects: Callable[[set[str]], int],
        show_message: Callable[[str, int], None],
    ) -> None:
        super().__init__()
        self._list_projects = list_projects
        self._create_project = create_project
        self._update_project = update_project
        self._delete_projects = delete_projects
        self._show_message = show_message
        self._records: list[dict[str, object]] = []
        self._selected_project_id: str | None = None
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(14)

        header = QHBoxLayout()
        title = QLabel("项目")
        title.setObjectName("pageTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.new_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon), "新建"
        )
        self.new_button.clicked.connect(self._new_project)
        self.save_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "保存"
        )
        self.save_button.clicked.connect(self._save_project)
        self.delete_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon), "删除所选"
        )
        self.delete_button.clicked.connect(self._delete_selected)
        header.addWidget(self.new_button)
        header.addWidget(self.save_button)
        header.addWidget(self.delete_button)
        layout.addLayout(header)

        form = QFormLayout()
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("项目名称")
        self.description_input = QPlainTextEdit()
        self.description_input.setPlaceholderText("项目说明")
        self.description_input.setMaximumHeight(72)
        form.addRow("名称:", self.title_input)
        form.addRow("说明:", self.description_input)
        layout.addLayout(form)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "项目", "说明", "任务数", "更新时间"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemDoubleClicked.connect(lambda _item: self._selection_changed())
        layout.addWidget(self.table, 1)

    def refresh(self) -> None:
        self._records = self._list_projects() or []
        self.table.setRowCount(len(self._records))
        for row, record in enumerate(self._records):
            checkbox = QTableWidgetItem()
            checkbox.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            checkbox.setCheckState(Qt.CheckState.Unchecked)
            checkbox.setData(Qt.ItemDataRole.UserRole, record.get("project_id"))
            self.table.setItem(row, 0, checkbox)
            self.table.setItem(row, 1, QTableWidgetItem(str(record.get("title", ""))))
            self.table.setItem(row, 2, QTableWidgetItem(str(record.get("description", ""))))
            self.table.setItem(row, 3, QTableWidgetItem(str(record.get("task_count", 0))))
            self.table.setItem(
                row,
                4,
                QTableWidgetItem(format_china_datetime(record.get("updated_at"))),
            )

    def _selection_changed(self) -> None:
        rows = sorted({item.row() for item in self.table.selectedItems()})
        if not rows:
            return
        record = self._records[rows[0]]
        self._selected_project_id = str(record.get("project_id", ""))
        self.title_input.setText(str(record.get("title", "")))
        self.description_input.setPlainText(str(record.get("description", "")))

    def _new_project(self) -> None:
        title = self.title_input.text().strip()
        if not title:
            self._show_message("请输入项目名称", 4000)
            self.title_input.setFocus()
            return
        try:
            self._create_project(title, self.description_input.toPlainText())
        except Exception as exc:
            self._show_message(str(exc), 6000)
            return
        self._selected_project_id = None
        self.title_input.clear()
        self.description_input.clear()
        self.refresh()
        self.projects_changed.emit()
        self._show_message("项目已创建", 4000)

    def _save_project(self) -> None:
        if not self._selected_project_id:
            self._show_message("请先选择要编辑的项目", 4000)
            return
        try:
            self._update_project(
                self._selected_project_id,
                self.title_input.text(),
                self.description_input.toPlainText(),
            )
        except Exception as exc:
            self._show_message(str(exc), 6000)
            return
        self.refresh()
        self.projects_changed.emit()
        self._show_message("项目已保存", 4000)

    def _delete_selected(self) -> None:
        identifiers = {
            str(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
            for row in range(self.table.rowCount())
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked
        }
        identifiers.update(
            str(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
            for row in {item.row() for item in self.table.selectedItems()}
        )
        if not identifiers:
            self._show_message("请勾选或选择要删除的项目", 4000)
            return
        removed = self._delete_projects(identifiers)
        self._selected_project_id = None
        self.title_input.clear()
        self.description_input.clear()
        self.refresh()
        self.projects_changed.emit()
        self._show_message(f"已删除 {removed} 个项目", 4000)
