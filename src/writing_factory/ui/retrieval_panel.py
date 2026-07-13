"""Non-blocking stage-3 retrieval controls and traceable result table."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtWidgets import (
    QFrame,
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
)

from writing_factory.ui.workers import BackgroundTaskManager, TaskContext


class RetrievalPanel(QFrame):
    """Run hybrid retrieval on a worker and render source-readable results."""

    def __init__(
        self,
        tasks: BackgroundTaskManager,
        *,
        retrieve: Callable[..., Any] | None,
        get_option: Callable[[str, bool], bool] | None,
        show_message: Callable[[str, int], None],
    ) -> None:
        super().__init__()
        self._tasks = tasks
        self._retrieve = retrieve
        self._get_option = get_option or (lambda _key, default=True: default)
        self._show_message = show_message
        self._task_id: str | None = None
        self._document_names: dict[str, str] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        self.setObjectName("retrievalPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("检索测试")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.cancel_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton),
            "取消",
        )
        self.cancel_button.setToolTip("取消当前检索")
        self.cancel_button.clicked.connect(self.cancel_retrieval)
        self.cancel_button.hide()
        header.addWidget(self.cancel_button)
        self.retrieve_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView),
            "检索",
        )
        self.retrieve_button.setEnabled(self._retrieve is not None)
        self.retrieve_button.clicked.connect(self.start_retrieval)
        header.addWidget(self.retrieve_button)
        layout.addLayout(header)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("输入研究问题")
        self.query_input.setClearButtonEnabled(True)
        self.query_input.returnPressed.connect(self.start_retrieval)
        layout.addWidget(self.query_input)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setFixedHeight(16)
        self.progress.hide()
        layout.addWidget(self.progress)

        self.result_table = QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(
            ["排名", "来源", "文档", "章节/页码", "文本预览"]
        )
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setMaximumHeight(220)
        self.result_table.verticalHeader().setVisible(False)
        result_header = self.result_table.horizontalHeader()
        for column in range(4):
            result_header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        result_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.result_table)

    def set_document_names(self, names: dict[str, str]) -> None:
        """Refresh the display-name lookup without changing stable source IDs."""

        self._document_names = dict(names)

    def start_retrieval(self) -> None:
        """Start one cooperative retrieval task without blocking the GUI thread."""

        if self._task_id is not None or self._retrieve is None:
            return
        query = self.query_input.text().strip()
        if not query:
            self._show_message("请输入检索问题", 4000)
            return
        self.retrieve_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.progress.setValue(0)
        self.progress.show()
        self._show_message("正在检索", 0)

        def task(context: TaskContext):
            context.report_progress(10, "检索中")
            result = self._retrieve(
                query=query,
                use_rewrite=self._get_option("use_rewrite", True),
                use_hyde=self._get_option("use_hyde", True),
                context=context,
            )
            context.report_progress(100, "完成")
            return result

        self._task_id = self._tasks.start(
            task,
            on_success=self._succeeded,
            on_error=self._failed,
            on_progress=self._progressed,
        )

    def cancel_retrieval(self) -> None:
        """Request cooperative cancellation at the next safe pipeline boundary."""

        if self._task_id is None:
            return
        self._tasks.cancel(self._task_id)
        self.cancel_button.setEnabled(False)
        self._show_message("正在取消检索", 0)

    def _succeeded(self, result: Any) -> None:
        hits = tuple(getattr(result, "hits", ()))
        self.result_table.setRowCount(len(hits))
        for row, hit in enumerate(hits):
            source = {"dense": "稠密", "bm25": "稀疏", "hybrid": "混合"}.get(hit.source, hit.source)
            locator = " · ".join(
                str(value) for value in (hit.section_heading, hit.page_start) if value is not None
            )
            preview = QTableWidgetItem(hit.text[:120].replace("\n", " "))
            preview.setToolTip(hit.text)
            document = QTableWidgetItem(self._document_names.get(hit.doc_id, hit.doc_id))
            document.setToolTip(hit.doc_id)
            values = (
                QTableWidgetItem(str(hit.final_rank)),
                QTableWidgetItem(source),
                document,
                QTableWidgetItem(locator),
                preview,
            )
            for column, item in enumerate(values):
                self.result_table.setItem(row, column, item)
        self._show_message(f"检索完成 · 命中 {len(hits)} 条", 6000)
        self._finish()

    def _failed(self, message: str) -> None:
        self._show_message(message, 8000)
        self._finish()

    def _progressed(self, percent: int, message: str) -> None:
        self.progress.setValue(percent)
        if message:
            self._show_message(f"{message} · {percent}%", 0)

    def _finish(self) -> None:
        self._task_id = None
        self.retrieve_button.setEnabled(self._retrieve is not None)
        self.cancel_button.hide()
        self.progress.hide()
