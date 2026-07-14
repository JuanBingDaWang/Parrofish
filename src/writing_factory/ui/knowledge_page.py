"""Knowledge-base document table and non-blocking ingestion controls."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from writing_factory.ui.retrieval_panel import RetrievalPanel
from writing_factory.ui.time_format import format_china_datetime
from writing_factory.ui.workers import (
    BackgroundTaskManager,
    TaskCancelled,
    TaskContext,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BatchIngestionResult:
    """Compact UI result for a sequential multi-file import."""

    imported_count: int
    child_chunk_count: int
    failed_files: tuple[str, ...] = ()


class KnowledgeBasePage(QWidget):
    """Import into the default KB and display persisted document state."""

    documents_changed = pyqtSignal()

    def __init__(
        self,
        tasks: BackgroundTaskManager,
        *,
        ingest_document: Callable[[Path, TaskContext], Any] | None,
        list_documents: Callable[[], list[dict[str, object]]],
        delete_documents: Callable[[set[str], TaskContext], Any] | None,
        retrieve: Callable[..., Any] | None,
        get_retrieval_option: Callable[[str, bool], bool] | None,
        show_message: Callable[[str, int], None],
    ) -> None:
        super().__init__()
        self._tasks = tasks
        self._ingest_document = ingest_document
        self._list_documents = list_documents
        self._delete_documents = delete_documents
        self._show_message = show_message
        self._ingest_task_id: str | None = None
        self.retrieval_panel = RetrievalPanel(
            tasks,
            retrieve=retrieve,
            get_option=get_retrieval_option,
            show_message=show_message,
        )
        # Preserve concise accessors used by keyboard workflows and UI automation.
        self.query_input = self.retrieval_panel.query_input
        self.retrieve_button = self.retrieval_panel.retrieve_button
        self.retrieve_progress = self.retrieval_panel.progress
        self.retrieval_table = self.retrieval_panel.result_table
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        toolbar = QHBoxLayout()
        heading = QLabel("知识库")
        heading.setObjectName("pageTitle")
        self.import_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            "批量导入",
        )
        self.import_button.setEnabled(self._ingest_document is not None)
        self.import_button.clicked.connect(self._select_documents)
        self.delete_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon),
            "删除",
        )
        self.delete_button.setToolTip("删除选中的知识库文档")
        self.delete_button.clicked.connect(self.start_deletion)
        self.delete_button.setEnabled(False)
        toolbar.addWidget(heading)
        toolbar.addStretch(1)
        toolbar.addWidget(self.delete_button)
        toolbar.addWidget(self.import_button)
        layout.addLayout(toolbar)

        self.ingest_progress = QProgressBar()
        self.ingest_progress.setRange(0, 100)
        self.ingest_progress.setTextVisible(True)
        self.ingest_progress.setFixedHeight(18)
        self.ingest_progress.hide()
        layout.addWidget(self.ingest_progress)

        self.document_table = QTableWidget(0, 5)
        self.document_table.setHorizontalHeaderLabels(["选择", "文件", "状态", "切片", "入库时间"])
        self.document_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.document_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.document_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.document_table.setAlternatingRowColors(True)
        self.document_table.itemSelectionChanged.connect(self._update_buttons)
        self.document_table.itemChanged.connect(self._update_buttons)
        self.document_table.verticalHeader().setVisible(False)
        header = self.document_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (2, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.document_table, 1)

        layout.addWidget(self.retrieval_panel)
        self.refresh_documents()

    def start_deletion(self) -> None:
        """在后台删除单个或多个选中文档，不弹确认框。"""

        if self._ingest_task_id is not None or self._delete_documents is None:
            return
        doc_ids = self._selected_document_ids()
        if not doc_ids:
            return
        self._set_running(True)
        self._show_message(f"准备删除 · {len(doc_ids)} 个文档", 0)

        def task(context: TaskContext):
            return self._delete_documents(doc_ids, context)

        self._ingest_task_id = self._tasks.start(
            task,
            on_success=self._deletion_succeeded,
            on_error=self._deletion_failed,
            on_progress=self._ingest_progressed,
        )

    def _selected_document_ids(self) -> set[str]:
        checked = self._checked_document_ids()
        if checked:
            return checked
        selected: set[str] = set()
        selection = self.document_table.selectionModel()
        if selection is None:
            return selected
        for index in selection.selectedRows(1):
            item = self.document_table.item(index.row(), 0)
            value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if isinstance(value, str):
                selected.add(value)
        return selected

    def _checked_document_ids(self) -> set[str]:
        checked: set[str] = set()
        for row in range(self.document_table.rowCount()):
            item = self.document_table.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                value = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(value, str):
                    checked.add(value)
        return checked

    def _deletion_succeeded(self, result: Any) -> None:
        removed = int(getattr(result, "removed_count", 0))
        failures = int(getattr(result, "cleanup_failures", 0))
        message = f"已删除 · {removed} 个文档"
        if failures:
            message += f" · {failures} 个派生文件待清理"
        self._show_message(message, 7000)
        self.refresh_documents()
        self.documents_changed.emit()
        self._finish_ingestion()

    def _deletion_failed(self, message: str) -> None:
        self._show_message(message, 8000)
        self.refresh_documents()
        self.documents_changed.emit()
        self._finish_ingestion()

    def _select_documents(self) -> None:
        filenames, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "批量导入文档",
            "",
            "支持的文档 (*.pdf *.doc *.docx *.ppt *.pptx *.txt);;所有文件 (*)",
        )
        if filenames:
            self.start_ingestions(Path(filename) for filename in filenames)

    def start_ingestion(self, source_path: Path) -> None:
        """Keep the single-file programmatic entry point used by callers and tests."""

        self.start_ingestions((source_path,))

    def start_ingestions(self, source_paths: Iterable[Path]) -> None:
        """Import selected files sequentially in one background task."""

        if self._ingest_task_id is not None or self._ingest_document is None:
            return
        paths = tuple(Path(path) for path in source_paths)
        if not paths:
            return
        self._set_running(True)
        self.ingest_progress.setValue(0)
        self._show_message(f"准备入库 · {len(paths)} 个文件", 0)

        def task(context: TaskContext) -> BatchIngestionResult:
            imported = 0
            child_chunks = 0
            failed: list[str] = []
            total = len(paths)
            for index, path in enumerate(paths):
                context.check_cancelled()
                start = round(index * 100 / total)
                end = round((index + 1) * 100 / total)
                child_context = context.scaled(
                    start,
                    end,
                    prefix=f"{index + 1}/{total} {path.name} · ",
                )
                try:
                    result = self._ingest_document(path, child_context)
                except TaskCancelled:
                    raise
                except Exception as exc:
                    logger.exception("Document import failed: %s", type(exc).__name__)
                    failed.append(path.name)
                else:
                    imported += 1
                    child_chunks += int(getattr(result, "child_chunk_count", 0))
            return BatchIngestionResult(imported, child_chunks, tuple(failed))

        self._ingest_task_id = self._tasks.start(
            task,
            on_success=self._ingest_succeeded,
            on_error=self._ingest_failed,
            on_progress=self._ingest_progressed,
        )

    def _ingest_succeeded(self, result: Any) -> None:
        batch = result if isinstance(result, BatchIngestionResult) else None
        if batch is None:
            self._show_message("入库完成", 6000)
        elif batch.failed_files:
            self._show_message(
                f"批量入库结束 · {batch.imported_count} 成功 · {len(batch.failed_files)} 失败",
                8000,
            )
        elif batch.imported_count == 1:
            self._show_message(f"入库完成 · {batch.child_chunk_count} 个切片", 6000)
        else:
            self._show_message(
                f"批量入库完成 · {batch.imported_count} 个文件 · {batch.child_chunk_count} 个切片",
                6000,
            )
        self.refresh_documents()
        if batch is None or batch.imported_count:
            self.documents_changed.emit()
        self._finish_ingestion()

    def _ingest_failed(self, message: str) -> None:
        self._show_message(message, 8000)
        self.refresh_documents()
        self.documents_changed.emit()
        self._finish_ingestion()

    def _ingest_progressed(self, percent: int, message: str) -> None:
        self.ingest_progress.setValue(percent)
        if message:
            self._show_message(f"{message} · {percent}%", 0)

    def _finish_ingestion(self) -> None:
        self._ingest_task_id = None
        self._set_running(False)

    def refresh_documents(self) -> None:
        """Reload the table from SQLite after every terminal task state."""

        documents = self._list_documents()
        self.retrieval_panel.set_document_names(
            {
                str(document.get("doc_id", "")): str(document.get("filename", ""))
                for document in documents
            }
        )
        self.document_table.blockSignals(True)
        self.document_table.setRowCount(len(documents))
        for row, document in enumerate(documents):
            select = QTableWidgetItem()
            select.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            select.setCheckState(Qt.CheckState.Unchecked)
            select.setData(Qt.ItemDataRole.UserRole, document.get("doc_id"))
            self.document_table.setItem(row, 0, select)
            values = (
                str(document.get("filename", "")),
                self._status_label(str(document.get("status", ""))),
                str(document.get("chunk_count", 0)),
                format_china_datetime(document.get("ingest_date")),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.document_table.setItem(row, column + 1, item)
        self.document_table.blockSignals(False)
        self._update_buttons()

    def _set_running(self, running: bool) -> None:
        self.import_button.setEnabled(not running and self._ingest_document is not None)
        self.document_table.setEnabled(not running)
        self.ingest_progress.setVisible(running)
        if not running:
            self._update_buttons()
        else:
            self.delete_button.setEnabled(False)

    def _update_buttons(self) -> None:
        self.delete_button.setEnabled(
            self._ingest_task_id is None
            and self._delete_documents is not None
            and bool(self._selected_document_ids())
        )

    @staticmethod
    def _status_label(status: str) -> str:
        return {
            "ready": "可检索",
            "indexing": "索引中",
            "failed": "失败",
        }.get(status, status)

    def start_retrieval(self) -> None:
        """Forward the page action to the dedicated retrieval component."""

        self.retrieval_panel.start_retrieval()
