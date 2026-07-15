"""Modeless, resizable viewer for the writing pipeline's public model output."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QTextCursor, QTextDocument
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QPlainTextEdit,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class LiveOutputWindow(QWidget):
    """Display a shared QTextDocument without blocking or owning the task."""

    auto_scroll_changed = pyqtSignal(bool)

    def __init__(
        self,
        document: QTextDocument,
        *,
        auto_scroll: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("模型实时输出")
        self.resize(760, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        toolbar = QHBoxLayout()
        self.auto_scroll_checkbox = QCheckBox("自动滚动")
        self.auto_scroll_checkbox.setChecked(auto_scroll)
        self.auto_scroll_checkbox.toggled.connect(self.auto_scroll_changed)
        toolbar.addWidget(self.auto_scroll_checkbox)
        toolbar.addStretch(1)

        self.copy_button = QToolButton()
        self.copy_button.setIcon(self._icon("edit-copy", QStyle.StandardPixmap.SP_FileIcon))
        self.copy_button.setToolTip("复制全部实时输出")
        self.copy_button.clicked.connect(self.copy_all)
        toolbar.addWidget(self.copy_button)
        layout.addLayout(toolbar)

        self.output_view = QPlainTextEdit()
        self.output_view.setReadOnly(True)
        self.output_view.setDocument(document)
        layout.addWidget(self.output_view, 1)
        document.contentsChanged.connect(self._scroll_if_enabled)

    def show_output(self) -> None:
        """Show, raise, and focus the existing window while preserving its geometry."""

        self.show()
        self.raise_()
        self.activateWindow()
        self._scroll_if_enabled()

    def set_auto_scroll(self, enabled: bool) -> None:
        """Synchronize the inline and floating auto-scroll controls."""

        if self.auto_scroll_checkbox.isChecked() != enabled:
            self.auto_scroll_checkbox.setChecked(enabled)
        if enabled:
            self._scroll_if_enabled()

    def copy_all(self) -> None:
        """Copy the complete shared output buffer."""

        QApplication.clipboard().setText(self.output_view.toPlainText())

    def _scroll_if_enabled(self) -> None:
        if self.auto_scroll_checkbox.isChecked():
            self.output_view.moveCursor(QTextCursor.MoveOperation.End)
            self.output_view.ensureCursorVisible()

    def _icon(self, theme_name: str, fallback: QStyle.StandardPixmap) -> QIcon:
        icon = QIcon.fromTheme(theme_name)
        return icon if not icon.isNull() else self.style().standardIcon(fallback)
