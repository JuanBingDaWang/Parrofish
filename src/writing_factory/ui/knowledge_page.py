"""Knowledge-base document table and non-blocking ingestion controls."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
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
        show_message: Callable[[str, int], None],
    ) -> None:
        super().__init__()
        self._tasks = tasks
        self._ingest_document = ingest_document
        self._list_documents = list_documents
        self._show_message = show_message
        self._ingest_task_id: str | None = None
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
        toolbar.addWidget(heading)
        toolbar.addStretch(1)
        toolbar.addWidget(self.import_button)
        layout.addLayout(toolbar)

        self.ingest_progress = QProgressBar()
        self.ingest_progress.setRange(0, 100)
        self.ingest_progress.setTextVisible(True)
        self.ingest_progress.setFixedHeight(18)
        self.ingest_progress.hide()
        layout.addWidget(self.ingest_progress)

        self.document_table = QTableWidget(0, 4)
        self.document_table.setHorizontalHeaderLabels(["文件", "状态", "切片", "入库时间"])
        self.document_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.document_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.document_table.setAlternatingRowColors(True)
        self.document_table.verticalHeader().setVisible(False)
        header = self.document_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.document_table, 1)
        self.refresh_documents()

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
        self.import_button.setEnabled(False)
        self.ingest_progress.setValue(0)
        self.ingest_progress.show()
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
        self.import_button.setEnabled(self._ingest_document is not None)
        self.ingest_progress.hide()
        self._ingest_task_id = None

    def refresh_documents(self) -> None:
        """Reload the table from SQLite after every terminal task state."""

        documents = self._list_documents()
        self.document_table.setRowCount(len(documents))
        for row, document in enumerate(documents):
            values = (
                str(document.get("filename", "")),
                self._status_label(str(document.get("status", ""))),
                str(document.get("chunk_count", 0)),
                str(document.get("ingest_date", ""))[:19].replace("T", " "),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.document_table.setItem(row, column, item)

    @staticmethod
    def _status_label(status: str) -> str:
        return {
            "ready": "可检索",
            "indexing": "索引中",
            "failed": "失败",
        }.get(status, status)
