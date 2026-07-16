"""Persistent, streaming author-chat page with optional direct KB retrieval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from writing_factory.chat.models import (
    ChatConversation,
    ChatMessage,
    ChatReply,
    ChatVerificationResult,
)
from writing_factory.ui.chat_transcript import ChatTranscriptView
from writing_factory.ui.help_ui import create_help_button
from writing_factory.ui.time_format import format_china_datetime
from writing_factory.ui.widgets import NoWheelComboBox, SubmitTextEdit
from writing_factory.ui.workers import BackgroundTaskManager, TaskContext


class AuthorChatPage(QWidget):
    """Manage pinned-persona conversations and stream one answer at a time."""

    def __init__(
        self,
        tasks: BackgroundTaskManager,
        *,
        list_personas: Callable[[], list[dict[str, object]]],
        list_documents: Callable[[], list[dict[str, object]]],
        list_conversations: Callable[[], list[dict[str, object]]],
        load_conversation: Callable[[str], ChatConversation | None],
        create_conversation: Callable[..., str] | None,
        rename_conversation: Callable[[str, str], None] | None,
        delete_conversations: Callable[[set[str]], int] | None,
        list_messages: Callable[[str], list[ChatMessage]],
        send_message: Callable[..., ChatReply] | None,
        verify_message: Callable[..., ChatVerificationResult] | None,
        show_message: Callable[[str, int], None],
    ) -> None:
        super().__init__()
        self._tasks = tasks
        self._list_personas = list_personas
        self._list_documents = list_documents
        self._list_conversations = list_conversations
        self._load_conversation = load_conversation
        self._create_conversation = create_conversation
        self._rename_conversation = rename_conversation
        self._delete_conversations = delete_conversations
        self._list_messages = list_messages
        self._send_message = send_message
        self._verify_message = verify_message
        self._show_message = show_message
        self._conversation_id: str | None = None
        self._task_id: str | None = None
        self._conversation_records: list[dict[str, object]] = []
        self._personas: list[dict[str, object]] = []
        self._documents: list[dict[str, object]] = []
        self._last_assistant_message_id: str | None = None
        self._failed_stream_stage = ""
        self._failed_stream_reason = ""
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel("作者对话")
        title.setObjectName("pageTitle")
        header.addWidget(title)
        self.help_button = create_help_button("author_chat", self)
        header.addWidget(self.help_button)
        header.addStretch(1)
        layout.addLayout(header)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        layout.addWidget(self.main_splitter, 1)

        history = QWidget()
        history.setMinimumWidth(190)
        history.setMaximumWidth(260)
        history_layout = QVBoxLayout(history)
        history_layout.setContentsMargins(0, 0, 8, 0)
        history_buttons = QHBoxLayout()
        self.new_button = QToolButton()
        self.new_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        self.new_button.setToolTip("新建作者对话")
        self.new_button.clicked.connect(self.new_conversation)
        history_buttons.addWidget(self.new_button)
        self.rename_button = QToolButton()
        self.rename_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        )
        self.rename_button.setToolTip("重命名所选会话")
        self.rename_button.clicked.connect(self.rename_current)
        history_buttons.addWidget(self.rename_button)
        self.delete_button = QToolButton()
        self.delete_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        self.delete_button.setToolTip("删除所选会话")
        self.delete_button.clicked.connect(self.delete_selected)
        history_buttons.addWidget(self.delete_button)
        history_buttons.addStretch(1)
        history_layout.addLayout(history_buttons)
        self.conversation_list = QListWidget()
        self.conversation_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.conversation_list.setWordWrap(True)
        self.conversation_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.conversation_list.currentItemChanged.connect(self._conversation_changed)
        history_layout.addWidget(self.conversation_list, 1)
        self.main_splitter.addWidget(history)

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(8, 0, 0, 0)
        workspace_layout.setSpacing(8)
        config_row = QGridLayout()
        config_row.setHorizontalSpacing(8)
        config_row.setVerticalSpacing(6)
        config_row.addWidget(QLabel("作者档案"), 0, 0)
        self.persona_combo = NoWheelComboBox()
        self.persona_combo.setMinimumWidth(170)
        config_row.addWidget(self.persona_combo, 0, 1)
        config_row.addWidget(QLabel("知识库"), 0, 2)
        self.knowledge_combo = NoWheelComboBox()
        self.knowledge_combo.addItem("不使用", "none")
        self.knowledge_combo.addItem("全部可用文档", "all")
        self.knowledge_combo.addItem("选择文档", "selected")
        self.knowledge_combo.currentIndexChanged.connect(self._knowledge_mode_changed)
        config_row.addWidget(self.knowledge_combo, 0, 3)
        config_row.addWidget(QLabel("回答依据"), 1, 0)
        self.answer_policy_combo = NoWheelComboBox()
        self.answer_policy_combo.addItem("通用知识辅助", "general_assisted")
        self.answer_policy_combo.addItem("严格证据", "strict_evidence")
        self.answer_policy_combo.setToolTip(
            "通用知识辅助会保留模型的一般知识能力；严格证据要求具体事实来自知识库"
        )
        config_row.addWidget(self.answer_policy_combo, 1, 1, 1, 3)
        self.web_search_checkbox = QCheckBox("使用博查联网检索")
        self.web_search_checkbox.setToolTip(
            "按设置中的条目数检索公开网页；未勾选时保持原有对话流程"
        )
        config_row.addWidget(self.web_search_checkbox, 2, 1, 1, 3)
        self.allow_persona_sources = QCheckBox("允许检索作者目标语料")
        self.allow_persona_sources.setToolTip(
            "仅在确实需要查询作者原文时开启；默认隔离蒸馏目标语料"
        )
        config_row.addWidget(self.allow_persona_sources, 3, 1, 1, 3)
        config_row.setColumnStretch(1, 1)
        workspace_layout.addLayout(config_row)

        self.document_list = QListWidget()
        self.document_list.setMaximumHeight(120)
        self.document_list.setMinimumHeight(80)
        self.document_list.hide()
        workspace_layout.addWidget(self.document_list)

        transcript_toolbar = QHBoxLayout()
        self.conversation_status = QLabel("新对话")
        self.conversation_status.setObjectName("mutedText")
        transcript_toolbar.addWidget(self.conversation_status)
        transcript_toolbar.addStretch(1)
        self.auto_scroll_checkbox = QCheckBox("自动滚动")
        self.auto_scroll_checkbox.setChecked(True)
        transcript_toolbar.addWidget(self.auto_scroll_checkbox)
        self.verify_button = QPushButton("核验本条")
        self.verify_button.setEnabled(False)
        self.verify_button.clicked.connect(self.verify_latest)
        transcript_toolbar.addWidget(self.verify_button)
        workspace_layout.addLayout(transcript_toolbar)

        self.transcript = ChatTranscriptView()
        self.auto_scroll_checkbox.toggled.connect(self.transcript.set_auto_scroll)
        workspace_layout.addWidget(self.transcript, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedHeight(18)
        self.progress.hide()
        workspace_layout.addWidget(self.progress)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("mutedText")
        self.progress_label.hide()
        workspace_layout.addWidget(self.progress_label)

        composer_row = QHBoxLayout()
        self.message_input = SubmitTextEdit()
        self.message_input.setPlaceholderText("输入要与作者模型讨论的内容")
        self.message_input.setMaximumHeight(100)
        self.message_input.submit_requested.connect(self.send_current)
        composer_row.addWidget(self.message_input, 1)
        actions = QVBoxLayout()
        self.send_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward),
            "发送",
        )
        self.send_button.clicked.connect(self.send_current)
        actions.addWidget(self.send_button)
        self.stop_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop),
            "停止",
        )
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_current)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        composer_row.addLayout(actions)
        workspace_layout.addLayout(composer_row)
        self.main_splitter.addWidget(workspace)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([210, 520])

    def refresh(self) -> None:
        self._personas = [item for item in self._list_personas() if item.get("status") == "ready"]
        current_persona = self.persona_combo.currentData()
        self.persona_combo.blockSignals(True)
        self.persona_combo.clear()
        self.persona_combo.addItem("请选择", None)
        for persona in self._personas:
            self.persona_combo.addItem(
                f"{persona.get('name', '')} · v{persona.get('version_number', 1)}",
                persona.get("persona_id"),
            )
        index = self.persona_combo.findData(current_persona)
        self.persona_combo.setCurrentIndex(max(0, index))
        self.persona_combo.blockSignals(False)

        self._documents = [item for item in self._list_documents() if item.get("status") == "ready"]
        self.document_list.clear()
        for document in self._documents:
            item = QListWidgetItem(str(document.get("filename", "")))
            item.setData(Qt.ItemDataRole.UserRole, document.get("doc_id"))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.document_list.addItem(item)
        self._reload_conversations(select_id=self._conversation_id)
        self._update_actions()

    def new_conversation(self) -> None:
        if self._task_id is not None:
            return
        self._conversation_id = None
        self.conversation_list.clearSelection()
        self.conversation_list.setCurrentItem(None)
        self.persona_combo.setCurrentIndex(0)
        self.knowledge_combo.setCurrentIndex(0)
        self.answer_policy_combo.setCurrentIndex(0)
        self.web_search_checkbox.setChecked(False)
        self.allow_persona_sources.setChecked(False)
        for index in range(self.document_list.count()):
            self.document_list.item(index).setCheckState(Qt.CheckState.Unchecked)
        self.transcript.clear_conversation()
        self.conversation_status.setText("新对话")
        self._last_assistant_message_id = None
        self._set_config_enabled(True)
        self._update_actions()

    def send_current(self) -> None:
        if self._task_id is not None or self._send_message is None:
            return
        text = self.message_input.toPlainText().strip()
        if not text:
            self._show_message("请输入对话内容", 4000)
            return
        if self._conversation_id is None:
            if self._create_conversation is None:
                return
            persona_id = self.persona_combo.currentData()
            if not isinstance(persona_id, str):
                self._show_message("请选择一个已经完成的作者档案", 5000)
                return
            selected = self._selected_document_ids()
            if self.knowledge_combo.currentData() == "selected" and not selected:
                self._show_message("选择文档模式下请至少勾选一篇知识库文档", 5000)
                return
            allowed = set()
            if self.allow_persona_sources.isChecked():
                allowed = {str(item.get("doc_id", "")) for item in self._documents}
            self._conversation_id = self._create_conversation(
                persona_id=persona_id,
                knowledge_mode=str(self.knowledge_combo.currentData() or "none"),
                answer_policy=str(
                    self.answer_policy_combo.currentData() or "general_assisted"
                ),
                use_web_search=self.web_search_checkbox.isChecked(),
                selected_doc_ids=selected,
                allowed_persona_doc_ids=allowed,
            )
            self._set_config_enabled(False)
        conversation_id = self._conversation_id
        self.message_input.clear()
        self.transcript.start_turn(text, self._active_persona_name())
        self._failed_stream_stage = ""
        self._failed_stream_reason = ""
        self._set_running(True)

        def task(context: TaskContext) -> ChatReply:
            return self._send_message(
                conversation_id=conversation_id,
                user_message=text,
                context=context,
            )

        self._task_id = self._tasks.start(
            task,
            on_success=self._send_succeeded,
            on_error=self._task_failed,
            on_progress=self._progressed,
            on_stream=self._streamed,
        )

    def stop_current(self) -> None:
        if self._task_id is None:
            return
        self._tasks.cancel(self._task_id)
        self.stop_button.setEnabled(False)
        self.progress_label.setText("正在安全停止")

    def verify_latest(self) -> None:
        if (
            self._task_id is not None
            or self._verify_message is None
            or self._last_assistant_message_id is None
        ):
            return
        message_id = self._last_assistant_message_id
        self._failed_stream_stage = ""
        self._failed_stream_reason = ""
        self._set_running(True)
        self.progress_label.setText("中性核验本条回答")

        def task(context: TaskContext) -> ChatVerificationResult:
            return self._verify_message(message_id=message_id, context=context)

        self._task_id = self._tasks.start(
            task,
            on_success=self._verification_succeeded,
            on_error=self._task_failed,
            on_progress=self._progressed,
            on_stream=self._streamed,
        )

    def rename_current(self) -> None:
        if self._conversation_id is None or self._rename_conversation is None:
            return
        record = next(
            (
                item
                for item in self._conversation_records
                if item.get("conversation_id") == self._conversation_id
            ),
            {},
        )
        value, accepted = QInputDialog.getText(
            self,
            "重命名作者对话",
            "名称",
            text=str(record.get("title", "")),
        )
        if accepted and value.strip():
            self._rename_conversation(self._conversation_id, value)
            self._reload_conversations(select_id=self._conversation_id)

    def delete_selected(self) -> None:
        if self._delete_conversations is None or self._task_id is not None:
            return
        identifiers = {
            str(item.data(Qt.ItemDataRole.UserRole))
            for item in self.conversation_list.selectedItems()
            if item.data(Qt.ItemDataRole.UserRole)
        }
        if not identifiers and self._conversation_id:
            identifiers.add(self._conversation_id)
        if not identifiers:
            return
        removed = self._delete_conversations(identifiers)
        if self._conversation_id in identifiers:
            self.new_conversation()
        self._reload_conversations()
        self._show_message(f"已删除 {removed} 个作者对话", 4000)

    def _conversation_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None or self._task_id is not None:
            return
        conversation_id = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(conversation_id, str):
            return
        conversation = self._load_conversation(conversation_id)
        if conversation is None:
            return
        self._conversation_id = conversation_id
        self._load_config(conversation)
        self._render_messages(self._list_messages(conversation_id), conversation.persona_name)
        self.conversation_status.setText(
            f"{conversation.persona_name} · v{conversation.persona_version} · "
            f"{self._knowledge_label(conversation.knowledge_mode)} · "
            f"{self._answer_policy_label(conversation.answer_policy)}"
            f"{' · 联网检索' if conversation.use_web_search else ''}"
        )
        self._set_config_enabled(False)
        self._update_actions()

    def _load_config(self, conversation: ChatConversation) -> None:
        index = self.persona_combo.findData(conversation.persona_id)
        if index < 0:
            self.persona_combo.addItem(
                f"{conversation.persona_name} · v{conversation.persona_version}（快照）",
                conversation.persona_id,
            )
            index = self.persona_combo.count() - 1
        self.persona_combo.setCurrentIndex(index)
        mode_index = self.knowledge_combo.findData(conversation.knowledge_mode)
        self.knowledge_combo.setCurrentIndex(max(0, mode_index))
        policy_index = self.answer_policy_combo.findData(conversation.answer_policy)
        self.answer_policy_combo.setCurrentIndex(max(0, policy_index))
        self.web_search_checkbox.setChecked(conversation.use_web_search)
        selected = set(conversation.selected_doc_ids)
        for item_index in range(self.document_list.count()):
            item = self.document_list.item(item_index)
            item.setCheckState(
                Qt.CheckState.Checked
                if item.data(Qt.ItemDataRole.UserRole) in selected
                else Qt.CheckState.Unchecked
            )
        self.allow_persona_sources.setChecked(bool(conversation.allowed_persona_doc_ids))
        self._knowledge_mode_changed()

    def _render_messages(self, messages: list[ChatMessage], persona_name: str) -> None:
        self._last_assistant_message_id = None
        for message in messages:
            if message.role == "assistant" and message.status == "complete":
                self._last_assistant_message_id = message.message_id
        self.transcript.set_messages(messages, persona_name)

    def _streamed(self, kind: str, text: str) -> None:
        event_kind, separator, label = kind.partition("::")
        stage = label if separator else "作者对话回答"
        if event_kind == "reasoning":
            self.progress_label.setText(f"{stage} · 模型正在生成")
            return
        if event_kind == "status":
            if "中断，正在重试" in text and "作者对话回答" in stage:
                self.transcript.reset_stream_attempt()
            self.progress_label.setText(f"{stage} · {text}")
            return
        if event_kind == "attempt_reset":
            if "作者对话回答" in stage:
                self.transcript.reset_stream_attempt()
            self.progress_label.setText(f"{stage} · {text}")
            return
        if event_kind == "complete" and "作者对话回答" in stage:
            self.transcript.commit_stream_attempt()
            self.progress_label.setText(f"{stage} · 本次回答完整完成")
            return
        if event_kind == "error":
            self._failed_stream_stage = stage
            self._failed_stream_reason = text
            if "作者对话回答" in stage:
                self.transcript.reset_stream_attempt()
            self.progress_label.setText(f"{stage} · 失败：{text}")
            return
        if event_kind == "content" and text and "作者对话回答" in stage:
            self.transcript.append_stream(text)

    def _send_succeeded(self, result: Any) -> None:
        reply = result if isinstance(result, ChatReply) else None
        if reply is not None:
            self._last_assistant_message_id = reply.message.message_id
        self._finish_task()
        self._reload_conversations(select_id=self._conversation_id)
        if self._conversation_id:
            conversation = self._load_conversation(self._conversation_id)
            if conversation:
                self._render_messages(
                    self._list_messages(self._conversation_id),
                    conversation.persona_name,
                )
        self._show_message("作者回答完成", 4000)

    def _verification_succeeded(self, result: Any) -> None:
        label = result.overall_verdict if isinstance(result, ChatVerificationResult) else "完成"
        self._finish_task()
        if self._conversation_id:
            conversation = self._load_conversation(self._conversation_id)
            if conversation:
                self._render_messages(
                    self._list_messages(self._conversation_id),
                    conversation.persona_name,
                )
        self._show_message(f"本条回答核验完成 · {label}", 5000)

    def _task_failed(self, message: str) -> None:
        self._finish_task()
        self._reload_conversations(select_id=self._conversation_id)
        persisted_terminal = False
        if self._conversation_id:
            conversation = self._load_conversation(self._conversation_id)
            if conversation:
                messages = self._list_messages(self._conversation_id)
                self._render_messages(
                    messages,
                    conversation.persona_name,
                )
                persisted_terminal = bool(
                    messages
                    and messages[-1].role == "assistant"
                    and messages[-1].status in {"error", "interrupted"}
                )
        if message == "任务已取消":
            if not persisted_terminal:
                self.transcript.append_error("已停止；当前不完整回答未保存")
            self._show_message("作者对话已停止 · 当前不完整回答未保存", 8000)
        else:
            stage = self._failed_stream_stage or "作者对话任务"
            reason = self._failed_stream_reason or message
            if not persisted_terminal:
                self.transcript.append_error(f"{stage}失败：{reason}")
            self._show_message(f"{stage}失败：{reason}", 8000)

    def _progressed(self, percent: int, message: str) -> None:
        self.progress.setValue(percent)
        if message:
            self.progress_label.setText(message)

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        self.progress_label.setVisible(running)
        self.send_button.setEnabled(not running and self._send_message is not None)
        self.stop_button.setEnabled(running)
        self.new_button.setEnabled(not running)
        self.rename_button.setEnabled(not running and self._conversation_id is not None)
        self.delete_button.setEnabled(not running)
        self.verify_button.setEnabled(
            not running
            and self._verify_message is not None
            and self._last_assistant_message_id is not None
        )

    def _finish_task(self) -> None:
        self._task_id = None
        self.transcript.flush_stream()
        self.progress.setValue(0)
        self._set_running(False)
        self._update_actions()

    def _set_config_enabled(self, enabled: bool) -> None:
        self.persona_combo.setEnabled(enabled)
        self.knowledge_combo.setEnabled(enabled)
        self.answer_policy_combo.setEnabled(enabled)
        self.web_search_checkbox.setEnabled(enabled)
        self.document_list.setEnabled(enabled)
        self.allow_persona_sources.setEnabled(enabled)

    def _update_actions(self) -> None:
        running = self._task_id is not None
        self.rename_button.setEnabled(not running and self._conversation_id is not None)
        self.delete_button.setEnabled(not running and bool(self._conversation_records))
        self.verify_button.setEnabled(
            not running
            and self._verify_message is not None
            and self._last_assistant_message_id is not None
        )
        self.send_button.setEnabled(not running and self._send_message is not None)

    def _reload_conversations(self, *, select_id: str | None = None) -> None:
        self._conversation_records = self._list_conversations()
        self.conversation_list.blockSignals(True)
        self.conversation_list.clear()
        target_item: QListWidgetItem | None = None
        for record in self._conversation_records:
            title = str(record.get("title", "新对话"))
            persona = str(record.get("persona_name", ""))
            updated = format_china_datetime(record.get("updated_at"))
            item = QListWidgetItem(f"{title}\n{persona} · {updated}")
            identifier = str(record.get("conversation_id", ""))
            item.setData(Qt.ItemDataRole.UserRole, identifier)
            item.setToolTip(f"{title}\n{persona}\n{updated}")
            self.conversation_list.addItem(item)
            if identifier == select_id:
                target_item = item
        self.conversation_list.blockSignals(False)
        if target_item is not None:
            self.conversation_list.setCurrentItem(target_item)
        self._update_actions()

    def _knowledge_mode_changed(self) -> None:
        mode = str(self.knowledge_combo.currentData() or "none")
        self.document_list.setVisible(mode == "selected")
        self.allow_persona_sources.setVisible(mode != "none")

    def _selected_document_ids(self) -> set[str]:
        return {
            str(self.document_list.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.document_list.count())
            if self.document_list.item(index).checkState() == Qt.CheckState.Checked
        }

    def _active_persona_name(self) -> str:
        if self._conversation_id:
            conversation = self._load_conversation(self._conversation_id)
            if conversation:
                return conversation.persona_name
        text = self.persona_combo.currentText().split(" · ", 1)[0]
        return text or "作者模型"

    @staticmethod
    def _knowledge_label(mode: str) -> str:
        return {"none": "无知识库", "all": "全部可用文档", "selected": "指定文档"}.get(
            mode,
            mode,
        )

    @staticmethod
    def _answer_policy_label(policy: str) -> str:
        return {
            "general_assisted": "通用知识辅助",
            "strict_evidence": "严格证据",
        }.get(policy, policy)
