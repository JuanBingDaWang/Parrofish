"""Rich Markdown transcript rendering for persistent author conversations."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Literal

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import (
    QColor,
    QFont,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from PyQt6.QtWidgets import QTextBrowser

from writing_factory.chat.models import ChatMessage, ChatSource


@dataclass
class _TranscriptEntry:
    """One presentation-only message; persisted Markdown remains unchanged."""

    role: Literal["user", "assistant", "system"]
    label: str
    content: str
    status: str = "complete"
    sources: list[ChatSource] = field(default_factory=list)
    verification: dict[str, object] | None = None


class ChatTranscriptView(QTextBrowser):
    """Render chat messages as safe Markdown with throttled streaming updates."""

    _USER_HEADER = QColor("#e8edf2")
    _ASSISTANT_HEADER = QColor("#dcece5")
    _INFO_BACKGROUND = QColor("#f2f5f7")
    _ERROR_BACKGROUND = QColor("#fdebea")
    _TEXT = QColor("#26313b")
    _MUTED_TEXT = QColor("#586573")
    _ERROR_TEXT = QColor("#9f2d25")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._entries: list[_TranscriptEntry] = []
        self._stream_entry_index: int | None = None
        self._stream_attempt_start = 0
        self._auto_scroll = True
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(80)
        self._render_timer.timeout.connect(self._render_now)
        self.setReadOnly(True)
        self.setOpenExternalLinks(False)
        self.setOpenLinks(False)
        self.setPlaceholderText("选择作者档案并开始对话")
        self.document().setMaximumBlockCount(6000)
        self.document().setDefaultStyleSheet(
            "pre, code { background-color: #f1f3f5; color: #26313b; }"
            "blockquote { color: #586573; border-left: 3px solid #aeb8c2; }"
            "a { color: #1f6a59; }"
        )

    def set_auto_scroll(self, enabled: bool) -> None:
        """Enable follow-tail behavior without moving a manually positioned view."""

        self._auto_scroll = enabled
        if enabled:
            self.scroll_to_end()

    def clear_conversation(self) -> None:
        """Clear all presentation state for a new conversation."""

        self._render_timer.stop()
        self._entries.clear()
        self._stream_entry_index = None
        self._stream_attempt_start = 0
        self.clear()

    def set_messages(self, messages: list[ChatMessage], persona_name: str) -> None:
        """Replace the view with persisted messages and their audit metadata."""

        self._render_timer.stop()
        self._entries = [
            _TranscriptEntry(
                role=message.role,
                label="你" if message.role == "user" else persona_name,
                content=message.content,
                status=message.status,
                sources=list(message.sources),
                verification=message.verification,
            )
            for message in messages
        ]
        self._stream_entry_index = None
        self._stream_attempt_start = 0
        self._render_now()

    def start_turn(self, user_message: str, persona_name: str) -> None:
        """Add a local user turn and an empty assistant stream target."""

        self._entries.extend(
            [
                _TranscriptEntry(role="user", label="你", content=user_message),
                _TranscriptEntry(role="assistant", label=persona_name, content=""),
            ]
        )
        self._stream_entry_index = len(self._entries) - 1
        self._stream_attempt_start = 0
        self._render_now()

    def append_stream(self, text: str) -> None:
        """Buffer a streaming Markdown fragment and coalesce visual updates."""

        if not text or self._stream_entry_index is None:
            return
        self._entries[self._stream_entry_index].content += text
        if not self._render_timer.isActive():
            self._render_timer.start()

    def reset_stream_attempt(self) -> None:
        """Discard only the interrupted request attempt before a retry."""

        if self._stream_entry_index is None:
            return
        entry = self._entries[self._stream_entry_index]
        entry.content = entry.content[: self._stream_attempt_start]
        self._render_now()

    def commit_stream_attempt(self) -> None:
        """Mark the currently displayed response as complete and retainable."""

        if self._stream_entry_index is None:
            return
        self._stream_attempt_start = len(
            self._entries[self._stream_entry_index].content
        )

    def append_error(self, message: str) -> None:
        """Append a local task error outside the persisted model response."""

        self._entries.append(
            _TranscriptEntry(role="system", label="任务未完成", content=message)
        )
        self._stream_entry_index = None
        self._render_now()

    def flush_stream(self) -> None:
        """Render any fragments still waiting in the throttle window."""

        if self._render_timer.isActive():
            self._render_timer.stop()
            self._render_now()

    def scroll_to_end(self) -> None:
        """Move to the transcript tail only when follow-tail is enabled."""

        if not self._auto_scroll:
            return
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _render_now(self) -> None:
        scrollbar = self.verticalScrollBar()
        previous_value = scrollbar.value()
        document = self.document()
        document.clear()
        cursor = QTextCursor(document)
        for index, entry in enumerate(self._entries):
            if index:
                cursor.insertBlock()
            self._insert_header(cursor, entry)
            cursor.insertBlock(QTextBlockFormat(), QTextCharFormat())
            if entry.content:
                cursor.insertMarkdown(
                    _safe_markdown(entry.content),
                    QTextDocument.MarkdownFeature.MarkdownDialectGitHub,
                )
            if entry.sources:
                self._insert_sources(cursor, entry.sources)
            if entry.verification:
                self._insert_verification(cursor, entry.verification)
        if self._auto_scroll:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(min(previous_value, scrollbar.maximum()))

    def _insert_header(self, cursor: QTextCursor, entry: _TranscriptEntry) -> None:
        block_format = QTextBlockFormat()
        block_format.setBackground(
            self._USER_HEADER if entry.role == "user" else self._ASSISTANT_HEADER
        )
        if entry.role == "system" or entry.status == "error":
            block_format.setBackground(self._ERROR_BACKGROUND)
        block_format.setLeftMargin(8)
        block_format.setRightMargin(8)
        block_format.setTopMargin(5)
        block_format.setBottomMargin(5)
        cursor.setBlockFormat(block_format)
        char_format = QTextCharFormat()
        char_format.setForeground(
            self._ERROR_TEXT
            if entry.role == "system" or entry.status == "error"
            else self._TEXT
        )
        char_format.setFontWeight(QFont.Weight.DemiBold)
        suffix = {
            "interrupted": "（已中断）",
            "error": "（失败）",
        }.get(entry.status, "")
        cursor.insertText(f"{entry.label}{suffix}", char_format)

    def _insert_sources(self, cursor: QTextCursor, sources: list[ChatSource]) -> None:
        lines = ["本轮检索来源"]
        for source in sources:
            kind = " · 联网" if source.source_type == "web" else ""
            url = f"\n{source.url}" if source.url else ""
            lines.append(
                f"[{source.source_key}] {source.filename}"
                f"{_source_locator(source.page_start, source.page_end)}{kind}{url}"
            )
        self._insert_info_block(cursor, "\n".join(lines))

    def _insert_verification(
        self,
        cursor: QTextCursor,
        verification: dict[str, object],
    ) -> None:
        verdict = str(verification.get("overall_verdict", ""))
        note = str(verification.get("note", ""))
        self._insert_info_block(cursor, f"中性核验：{verdict} {note}".strip())

    def _insert_info_block(self, cursor: QTextCursor, text: str) -> None:
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
        cursor.insertBlock()
        block_format = QTextBlockFormat()
        block_format.setBackground(self._INFO_BACKGROUND)
        block_format.setLeftMargin(8)
        block_format.setRightMargin(8)
        block_format.setTopMargin(5)
        block_format.setBottomMargin(5)
        cursor.setBlockFormat(block_format)
        char_format = QTextCharFormat()
        char_format.setForeground(self._MUTED_TEXT)
        cursor.insertText(text, char_format)


def _safe_markdown(value: str) -> str:
    """Keep Markdown syntax while rendering raw HTML as inert text."""

    return html.escape(value, quote=False)


def _source_locator(page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return ""
    if page_end in {None, page_start}:
        return f" · 第{page_start}页"
    return f" · 第{page_start}-{page_end}页"
