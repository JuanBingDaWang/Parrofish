"""Reusable help button, dialog, and full tutorial page."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from writing_factory.ui.help_content import (
    HELP_TOPICS,
    TUTORIAL_ORDER,
    TUTORIAL_TOPICS,
)


class PageHelpDialog(QDialog):
    """Show one page's focused operating guide."""

    def __init__(self, topic_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        topic = HELP_TOPICS[topic_key]
        self.setWindowTitle(f"{topic.title} · 功能与操作")
        self.resize(680, 520)
        self.setMinimumSize(560, 400)
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(topic.allow_external_links)
        self.browser.setMarkdown(topic.markdown)
        layout.addWidget(self.browser, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def create_help_button(topic_key: str, parent: QWidget) -> QToolButton:
    """Create the compact question-mark button used beside page titles."""

    topic = HELP_TOPICS[topic_key]
    button = QToolButton(parent)
    button.setText("?")
    button.setFixedSize(24, 24)
    button.setToolTip(f"查看{topic.title}功能介绍和操作教程")
    button.setAccessibleName(f"{topic.title}帮助")
    button.clicked.connect(lambda: PageHelpDialog(topic_key, parent).exec())
    return button


class TutorialPage(QWidget):
    """Application-wide guide assembled from the same page help topics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(14)
        title_row = QHBoxLayout()
        title = QLabel("教程")
        title.setObjectName("pageTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.chapter_list = QListWidget()
        self.chapter_list.setMinimumWidth(170)
        self.chapter_list.setMaximumWidth(230)
        for key in TUTORIAL_ORDER:
            topic = TUTORIAL_TOPICS[key]
            item = QListWidgetItem(topic.title)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.chapter_list.addItem(item)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        splitter.addWidget(self.chapter_list)
        splitter.addWidget(self.browser)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([190, 700])
        layout.addWidget(splitter, 1)
        self.chapter_list.currentItemChanged.connect(self._chapter_changed)
        self.chapter_list.setCurrentRow(0)

    def _chapter_changed(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        key = current.data(Qt.ItemDataRole.UserRole)
        if isinstance(key, str):
            topic = TUTORIAL_TOPICS[key]
            self.browser.setMarkdown(f"# {topic.title}\n\n{topic.markdown}")
