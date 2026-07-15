"""Reusable interaction-safe Qt widgets."""

from __future__ import annotations

from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QComboBox


class NoWheelComboBox(QComboBox):
    """Ignore wheel selection changes unless the popup list is open."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API name
        if self.view().isVisible():
            super().wheelEvent(event)
            return
        event.ignore()
