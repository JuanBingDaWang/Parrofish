"""Reusable interaction-safe Qt widgets."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QWheelEvent
from PyQt6.QtWidgets import QComboBox, QTextEdit


class NoWheelComboBox(QComboBox):
    """Ignore wheel selection changes unless the popup list is open."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API name
        if self.view().isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


class SubmitTextEdit(QTextEdit):
    """Submit on bare Enter while modified Enter inserts a newline."""

    submit_requested = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API name
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            modifiers = event.modifiers()
            newline_modifiers = (
                Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
            )
            if modifiers & newline_modifiers:
                self.insertPlainText("\n")
            else:
                self.submit_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)
