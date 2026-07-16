"""Reusable labeled stream viewer for concurrent SiliconFlow calls."""

from __future__ import annotations

from PyQt6.QtGui import QIcon, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
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

        call_row = QHBoxLayout()
        call_row.setSpacing(6)
        call_row.addWidget(QLabel("当前调用"))
        self.call_combo = NoWheelComboBox()
        self.call_combo.setMinimumWidth(140)
        self.call_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.call_combo.currentIndexChanged.connect(self._show_selected_buffer)
        call_row.addWidget(self.call_combo, 1)

        layout.addLayout(call_row)

        options_row = QHBoxLayout()
        options_row.setSpacing(6)
        options_row.addStretch(1)
        self.auto_scroll_checkbox = QCheckBox("自动滚动")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.toggled.connect(self._sync_auto_scroll)
        options_row.addWidget(self.auto_scroll_checkbox)

        self.copy_button = QToolButton()
        self.copy_button.setIcon(self._theme_icon("edit-copy", QStyle.StandardPixmap.SP_FileIcon))
        self.copy_button.setToolTip("复制当前调用的全部输出")
        self.copy_button.clicked.connect(self._copy)
        options_row.addWidget(self.copy_button)

        self.popout_button = QToolButton()
        self.popout_button.setIcon(
            self._theme_icon("window-new", QStyle.StandardPixmap.SP_TitleBarNormalButton)
        )
        self.popout_button.setToolTip("在独立窗口中查看当前调用")
        self.popout_button.clicked.connect(self._show_window)
        options_row.addWidget(self.popout_button)
        layout.addLayout(options_row)

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
        self.call_combo.setToolTip("")

    def append_stream(self, kind: str, text: str) -> None:
        """Route one worker stream event to its labeled call buffer."""

        event_kind, separator, label = kind.partition("::")
        stage = label if separator and label else "模型输出"
        if event_kind == "reasoning":
            self._ensure_stage(stage)
            return
        if event_kind == "complete":
            self._ensure_stage(stage)
            self._attempt_starts[stage] = len(self._buffers.get(stage, ""))
            return
        if event_kind == "error" and text:
            self._ensure_stage(stage)
            committed = self._attempt_starts.get(stage, 0)
            buffer = self._buffers.get(stage, "")[:committed]
            separator = "\n" if buffer and not buffer.endswith("\n") else ""
            buffer += f"{separator}\n[失败原因] {text}\n"
            self._buffers[stage] = buffer
            self._attempt_starts[stage] = len(buffer)
            self._update_combo_label(stage)
            index = self.call_combo.findData(stage)
            if index >= 0:
                self.call_combo.setCurrentIndex(index)
            self._replace_visible_text(buffer)
            return
        if event_kind == "attempt_reset" and text:
            self._ensure_stage(stage)
            committed = self._attempt_starts.get(stage, 0)
            buffer = self._buffers.get(stage, "")[:committed]
            separator = "\n" if buffer and not buffer.endswith("\n") else ""
            buffer += f"{separator}\n[系统] {text}\n"
            self._buffers[stage] = buffer
            self._attempt_starts[stage] = len(buffer)
            self._update_combo_label(stage)
            if self._selected_stage() == stage:
                self._replace_visible_text(buffer)
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
        if self._selected_stage() == stage:
            self._replace_visible_text(buffer)

    def discard_incomplete_attempts(self) -> None:
        """Drop only uncommitted deltas while retaining completed calls."""

        for stage, buffer in tuple(self._buffers.items()):
            committed = self._attempt_starts.get(stage, 0)
            self._buffers[stage] = buffer[:committed]
            self._update_combo_label(stage)
        stage = self._selected_stage()
        self._replace_visible_text(self._buffers.get(stage or "", ""))

    def _ensure_stage(self, stage: str) -> None:
        if stage in self._buffers:
            return
        self._buffers[stage] = ""
        self._attempt_starts[stage] = 0
        self.call_combo.addItem(self._compact_stage_label(stage), stage)
        self._resize_combo_popup()
        if self.call_combo.count() == 1:
            self.call_combo.setCurrentIndex(0)

    def _update_combo_label(self, stage: str) -> None:
        index = self.call_combo.findData(stage)
        if index >= 0:
            compact = self._compact_stage_label(stage)
            self.call_combo.setItemText(
                index,
                f"{compact} · {len(self._buffers[stage])} 字",
            )
            if self.call_combo.currentIndex() == index:
                self.call_combo.setToolTip(
                    f"{stage} · {len(self._buffers[stage])} 字"
                )
            self._resize_combo_popup()

    def _selected_stage(self) -> str | None:
        value = self.call_combo.currentData()
        return value if isinstance(value, str) else None

    def _show_selected_buffer(self) -> None:
        stage = self._selected_stage()
        self._replace_visible_text(self._buffers.get(stage or "", ""))
        if stage is None:
            self.call_combo.setToolTip("")
        else:
            self.call_combo.setToolTip(
                f"{stage} · {len(self._buffers.get(stage, ''))} 字"
            )

    @staticmethod
    def _compact_stage_label(stage: str) -> str:
        """Remove repeated Map prefixes while keeping file and unit identity visible."""

        parts = [part.strip() for part in stage.split("·") if part.strip()]
        if len(parts) >= 2 and parts[0] in {
            "目标认知 Map",
            "认知 Map",
            "对照认知 Map",
            "目标谋篇 Map",
            "对照谋篇 Map",
        }:
            return " · ".join(parts[1:])
        return stage

    def _resize_combo_popup(self) -> None:
        """Keep long concurrent-call labels readable when the popup opens."""

        metrics = self.call_combo.fontMetrics()
        content_width = max(
            (
                metrics.horizontalAdvance(self.call_combo.itemText(index))
                for index in range(self.call_combo.count())
            ),
            default=0,
        )
        self.call_combo.view().setMinimumWidth(
            min(720, max(self.call_combo.width(), content_width + 44))
        )

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
