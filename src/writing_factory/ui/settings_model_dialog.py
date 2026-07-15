"""SiliconFlow model selection and embedding-index migration dialog."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from writing_factory.llm.configuration import ModelCatalogEntry
from writing_factory.llm.settings_service import ModelKind
from writing_factory.ui.settings_dialogs import SettingsDialogBackend
from writing_factory.ui.widgets import NoWheelComboBox
from writing_factory.ui.workers import BackgroundTaskManager, TaskContext


class ModelSettingsDialog(QDialog):
    """Select one model from the cached or live SiliconFlow catalog."""

    _LABELS = {"chat": "文字生成", "embedding": "Embedding", "reranker": "Rerank"}

    def __init__(
        self,
        backend: SettingsDialogBackend,
        tasks: BackgroundTaskManager,
        kind: ModelKind,
        *,
        show_message: Callable[[str, int], None],
        on_changed: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.backend = backend
        self.tasks = tasks
        self.kind = kind
        self.show_message = show_message
        self.on_changed = on_changed
        self._task_id: str | None = None
        self.setWindowTitle(f"{self._LABELS[kind]} 模型")
        self.setMinimumWidth(620)

        layout = QVBoxLayout(self)
        self.catalog_status = QLabel()
        layout.addWidget(self.catalog_status)
        model_row = QHBoxLayout()
        self.model_combo = NoWheelComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(NoWheelComboBox.InsertPolicy.NoInsert)
        model_row.addWidget(self.model_combo, 1)
        self.refresh_button = QPushButton("联网刷新")
        self.refresh_button.clicked.connect(self._refresh)
        model_row.addWidget(self.refresh_button)
        layout.addLayout(model_row)

        compatibility = QLabel(
            "模型列表只证明该 ID 属于当前类别，不包含上下文长度、价格或 thinking 能力信息。"
        )
        compatibility.setWordWrap(True)
        compatibility.setObjectName("mutedText")
        layout.addWidget(compatibility)

        self.pending_label = QLabel()
        self.pending_label.setWordWrap(True)
        layout.addWidget(self.pending_label)
        self.rebuild_button = QPushButton("后台重建向量并切换")
        self.rebuild_button.clicked.connect(self._rebuild)
        layout.addWidget(self.rebuild_button, 0, Qt.AlignmentFlag.AlignLeft)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._load_catalog()

    def _current_model(self) -> str:
        models = self.backend.get_model_selections()
        return {
            "chat": models.chat_model,
            "embedding": models.embedding_model,
            "reranker": models.rerank_model,
        }[self.kind]

    def _load_catalog(self) -> None:
        entries, updated_at = self.backend.cached_models(self.kind)
        current = self._current_model()
        ids = sorted({current, *(entry.id for entry in entries)}, key=str.casefold)
        self.model_combo.clear()
        self.model_combo.addItems(ids)
        self.model_combo.setCurrentText(current)
        self.catalog_status.setText(
            f"已缓存 {len(entries)} 个模型"
            + (f" · 更新于 {updated_at}" if updated_at else " · 尚未联网刷新")
        )
        models = self.backend.get_model_selections()
        pending = models.pending_embedding_model if self.kind == "embedding" else None
        self.pending_label.setText(
            f"等待重建：{pending}。重建成功前仍使用 {models.embedding_model}。"
            if pending
            else ""
        )
        self.rebuild_button.setVisible(bool(pending))

    def _refresh(self) -> None:
        if self._task_id is not None:
            return
        self.refresh_button.setEnabled(False)

        def task(context: TaskContext) -> list[ModelCatalogEntry]:
            context.report_progress(20, "正在拉取模型列表")
            result = self.backend.refresh_models(self.kind)
            context.report_progress(100, "模型列表已更新")
            return result

        self._task_id = self.tasks.start(
            task,
            on_success=lambda _result: self._finish_refresh(None),
            on_error=self._finish_refresh,
            on_progress=lambda percent, message: self.show_message(
                f"{message} · {percent}%", 0
            ),
        )

    def _finish_refresh(self, error: str | None) -> None:
        self._task_id = None
        self.refresh_button.setEnabled(True)
        if error:
            self.show_message(error, 8000)
            return
        self._load_catalog()
        self.show_message("模型列表已更新", 4000)

    def _save(self) -> None:
        try:
            pending = self.backend.set_model(self.kind, self.model_combo.currentText())
        except Exception as exc:
            self.catalog_status.setText(str(exc))
            self.catalog_status.setObjectName("errorText")
            return
        self.on_changed()
        if pending:
            self._load_catalog()
            self.show_message("新 Embedding 模型已保存，完成向量重建后才会切换", 7000)
            return
        self.accept()

    def _rebuild(self) -> None:
        if self._task_id is not None:
            return
        self.rebuild_button.setEnabled(False)

        def task(context: TaskContext) -> str:
            return self.backend.rebuild_embedding_index(
                progress=context.report_progress,
                check_cancelled=context.check_cancelled,
            )

        self._task_id = self.tasks.start(
            task,
            on_success=self._rebuild_succeeded,
            on_error=self._rebuild_failed,
            on_progress=lambda percent, message: self.show_message(
                f"{message} · {percent}%", 0
            ),
        )

    def _rebuild_succeeded(self, model: str) -> None:
        self._task_id = None
        self.rebuild_button.setEnabled(True)
        self._load_catalog()
        self.on_changed()
        self.show_message(f"向量索引已切换为 {model}", 6000)

    def _rebuild_failed(self, message: str) -> None:
        self._task_id = None
        self.rebuild_button.setEnabled(True)
        self.show_message(message, 8000)
