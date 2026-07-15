"""Reusable labeled stream viewer for concurrent SiliconFlow calls."""

from __future__ import annotations

from PyQt6.QtGui import QIcon, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from writing_factory.ui.live_output_window import LiveOutputWindow
from writing_factory.ui.widgets import NoWheelComboBox


class StreamOutputPanel(QWidget):
    """Keep one independent text buffer per concurrent stream label."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buffers: dict[str, str] = {}
        self._attempt_starts: dict[str, int] = {}
        self._window: LiveOutputWindow | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        toolbar.addWidget(QLabel("当前调用"))
        self.call_combo = NoWheelComboBox()
        self.call_combo.setMinimumWidth(220)
        self.call_combo.currentIndexChanged.connect(self._show_selected_buffer)
        toolbar.addWidget(self.call_combo, 1)
        self.activity_label = QLabel("尚未收到模型输出")
        self.activity_label.setObjectName("mutedText")
        toolbar.addWidget(self.activity_label)
        self.auto_scroll_checkbox = QCheckBox("自动滚动")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.toggled.connect(self._sync_auto_scroll)
        toolbar.addWidget(self.auto_scroll_checkbox)

        self.copy_button = QToolButton()
        self.copy_button.setIcon(self._theme_icon("edit-copy", QStyle.StandardPixmap.SP_FileIcon))
        self.copy_button.setToolTip("复制当前调用的全部输出")
        self.copy_button.clicked.connect(self._copy)
        toolbar.addWidget(self.copy_button)

        self.popout_button = QToolButton()
        self.popout_button.setIcon(
            self._theme_icon("window-new", QStyle.StandardPixmap.SP_TitleBarNormalButton)
        )
        self.popout_button.setToolTip("在独立窗口中查看当前调用")
        self.popout_button.clicked.connect(self._show_window)
        toolbar.addWidget(self.popout_button)
        layout.addLayout(toolbar)

        self.output_view = QPlainTextEdit()
        self.output_view.setReadOnly(True)
        self.output_view.setPlaceholderText("模型的实时输出将在这里逐步显示")
        self.output_view.setMinimumHeight(145)
        self.output_view.document().setMaximumBlockCount(5000)
        self.output_view.verticalScrollBar().rangeChanged.connect(self._range_changed)
        layout.addWidget(self.output_view, 1)

    def clear(self) -> None:
        """Clear all call buffers before a new background run."""

        self._buffers.clear()
        self._attempt_starts.clear()
        self.call_combo.blockSignals(True)
        self.call_combo.clear()
        self.call_combo.blockSignals(False)
        self.output_view.clear()
        self.activity_label.setText("尚未收到模型输出")

    def append_stream(self, kind: str, text: str) -> None:
        """Route one worker stream event to its labeled call buffer."""

        event_kind, separator, label = kind.partition("::")
        stage = label if separator and label else "模型输出"
        if event_kind == "reasoning":
            self.activity_label.setText(f"{stage} · 模型正在生成")
            self._ensure_stage(stage)
            return
        if event_kind not in {"content", "status"} or not text:
            return
        self._ensure_stage(stage)
        buffer = self._buffers.get(stage, "")
        if event_kind == "status":
            if "中断，正在重试" in text:
                buffer = buffer[: self._attempt_starts.get(stage, len(buffer))]
            buffer += f"\n[系统] {text}\n"
            self._attempt_starts[stage] = len(buffer)
        else:
            buffer += text
        self._buffers[stage] = buffer
        self._update_combo_label(stage)
        self.activity_label.setText(f"{stage} · 已接收 {len(buffer)} 字")
        if self._selected_stage() == stage:
            self._replace_visible_text(buffer)

    def _ensure_stage(self, stage: str) -> None:
        if stage in self._buffers:
            return
        self._buffers[stage] = ""
        self._attempt_starts[stage] = 0
        self.call_combo.addItem(stage, stage)
        if self.call_combo.count() == 1:
            self.call_combo.setCurrentIndex(0)

    def _update_combo_label(self, stage: str) -> None:
        index = self.call_combo.findData(stage)
        if index >= 0:
            self.call_combo.setItemText(index, f"{stage} · {len(self._buffers[stage])} 字")

    def _selected_stage(self) -> str | None:
        value = self.call_combo.currentData()
        return value if isinstance(value, str) else None

    def _show_selected_buffer(self) -> None:
        stage = self._selected_stage()
        self._replace_visible_text(self._buffers.get(stage or "", ""))

    def _replace_visible_text(self, text: str) -> None:
        scrollbar = self.output_view.verticalScrollBar()
        previous_scroll = scrollbar.value()
        self.output_view.setPlainText(text)
        if self.auto_scroll_checkbox.isChecked():
            self._scroll_to_end()
        else:
            scrollbar.setValue(min(previous_scroll, scrollbar.maximum()))

    def _copy(self) -> None:
        QApplication.clipboard().setText(self.output_view.toPlainText())

    def _show_window(self) -> None:
        if self._window is None:
            self._window = LiveOutputWindow(
                self.output_view.document(),
                auto_scroll=self.auto_scroll_checkbox.isChecked(),
                parent=self,
            )
            self._window.auto_scroll_changed.connect(self.auto_scroll_checkbox.setChecked)
        self._window.show_output()

    def _sync_auto_scroll(self, enabled: bool) -> None:
        if self._window is not None:
            self._window.set_auto_scroll(enabled)
        if enabled:
            self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        self.output_view.moveCursor(QTextCursor.MoveOperation.End)
        self.output_view.ensureCursorVisible()
        scrollbar = self.output_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _range_changed(self, _minimum: int, maximum: int) -> None:
        if self.auto_scroll_checkbox.isChecked():
            self.output_view.verticalScrollBar().setValue(maximum)

    def _theme_icon(self, name: str, fallback: QStyle.StandardPixmap) -> QIcon:
        icon = QIcon.fromTheme(name)
        return icon if not icon.isNull() else self.style().standardIcon(fallback)
