"""Author-chat persistence, retrieval policy, memory, and UI smoke tests."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import QLabel

from tests.test_distill_pipeline import _persona
from writing_factory.chat.models import (
    ChatConversation,
    ChatMessage,
    ChatReply,
    ChatSource,
)
from writing_factory.chat.repository import ChatRepository
from writing_factory.chat.service import AuthorChatService
from writing_factory.distill.runtime import build_runtime_persona
from writing_factory.kb.models import FusedHit, RetrievalResult
from writing_factory.llm.bocha import BochaSearchResult, BochaWebPage
from writing_factory.llm.models import ChatResult
from writing_factory.store import Database
from writing_factory.ui.chat_transcript import ChatTranscriptView
from writing_factory.ui.main_window import MainWindow
from writing_factory.ui.stream_output_panel import StreamOutputPanel
from writing_factory.ui.widgets import SubmitTextEdit
from writing_factory.ui.workers import TaskCancelled


class FakePersonaRepository:
    def __init__(self) -> None:
        self.runtime = build_runtime_persona(_persona("persona_chat"))

    def load_runtime(self, persona_id: str):
        return self.runtime if persona_id == "persona_chat" else None

    def list_personas(self, _kb_id: str):
        return [
            {
                "persona_id": "persona_chat",
                "name": "测试作者",
                "version_number": 3,
                "status": "ready",
            }
        ]

    def load_source_roles(self, _persona_id: str):
        return SimpleNamespace(target_doc_ids=frozenset({"author_target"}))


class FakeKBRepository:
    def list_documents(self, _kb_id: str):
        return [
            {"doc_id": "author_target", "filename": "作者原文.pdf", "status": "ready"},
            {"doc_id": "fact_doc", "filename": "事实材料.pdf", "status": "ready"},
        ]


class FakeRetriever:
    def __init__(self) -> None:
        self.requests = []

    def search(self, request, **_kwargs):
        self.requests.append(request)
        return RetrievalResult(
            query=request.query,
            hits=(
                FusedHit(
                    chunk_id="fact_chunk",
                    doc_id="fact_doc",
                    text="这是用于作者对话测试的普通事实材料。",
                    source="hybrid",
                    final_rank=1,
                    page_start=4,
                ),
            ),
        )


class FakeSiliconFlow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def chat(self, messages, **kwargs):
        step_id = str(kwargs.get("step_id", ""))
        self.calls.append((step_id, list(messages)))
        if step_id == "chat.summary":
            return ChatResult(content="用户持续讨论同一问题，尚未作出最终选择。", model="fake")
        if step_id == "chat.verify":
            return ChatResult(
                content=json.dumps(
                    {
                        "overall_verdict": "supported",
                        "assessments": [
                            {
                                "claim": "测试事实",
                                "source_keys": ["S1"],
                                "verdict": "supported",
                                "rationale": "来源支持",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                model="fake",
            )
        return ChatResult(content="基于材料回答 [S1]，无效来源 [S99]。", model="fake")

    @contextmanager
    def stream_stage(self, _label: str):
        yield


def test_author_chat_isolates_target_sources_summarizes_and_verifies(tmp_path) -> None:
    database = Database(tmp_path / "chat.db")
    database.initialize()
    repository = ChatRepository(database)
    retriever = FakeRetriever()
    siliconflow = FakeSiliconFlow()
    service = AuthorChatService(
        repository=repository,
        persona_repository=FakePersonaRepository(),
        kb_repository=FakeKBRepository(),
        retriever=retriever,
        siliconflow=siliconflow,
        kb_id="kb_default",
        recent_rounds=lambda: 1,
    )
    conversation_id = service.create_conversation(
        persona_id="persona_chat",
        knowledge_mode="all",
        selected_doc_ids=set(),
        allowed_persona_doc_ids=set(),
    )
    conversation = repository.load_conversation(conversation_id)
    assert conversation is not None
    assert conversation.answer_policy == "general_assisted"
    assert conversation.persona_version == 3
    assert "composition_dna" not in conversation.runtime_persona

    first = service.send(
        conversation_id=conversation_id,
        user_message="第一个问题",
        progress=lambda _percent, _message: None,
        check_cancelled=lambda: None,
    )
    second = service.send(
        conversation_id=conversation_id,
        user_message="第二个问题",
        progress=lambda _percent, _message: None,
        check_cancelled=lambda: None,
    )

    assert isinstance(first, ChatReply)
    assert "S99" not in first.message.content
    assert first.message.sources[0].filename == "事实材料.pdf"
    assert all(request.use_rewrite is False for request in retriever.requests)
    assert all(request.use_hyde is False for request in retriever.requests)
    assert all(request.filters.doc_ids == {"fact_doc"} for request in retriever.requests)
    assert any(step_id == "chat.summary" for step_id, _messages in siliconflow.calls)
    refreshed = repository.load_conversation(conversation_id)
    assert refreshed is not None
    assert refreshed.summary_text
    verification = service.verify(second.message.message_id)
    assert verification.overall_verdict == "supported"
    saved = repository.load_message(second.message.message_id)
    assert saved is not None
    assert saved.verification is not None
    assert saved.verification["overall_verdict"] == "supported"


def test_author_chat_persists_answer_policy_and_builds_policy_prompt(tmp_path) -> None:
    database = Database(tmp_path / "answer_policy.db")
    database.initialize()
    repository = ChatRepository(database)
    siliconflow = FakeSiliconFlow()
    service = AuthorChatService(
        repository=repository,
        persona_repository=FakePersonaRepository(),
        kb_repository=FakeKBRepository(),
        retriever=FakeRetriever(),
        siliconflow=siliconflow,
        kb_id="kb_default",
        recent_rounds=lambda: 6,
    )
    conversation_id = service.create_conversation(
        persona_id="persona_chat",
        knowledge_mode="none",
        answer_policy="strict_evidence",
        selected_doc_ids=set(),
        allowed_persona_doc_ids=set(),
    )

    service.send(
        conversation_id=conversation_id,
        user_message="介绍一个档案没有涉及的概念",
        progress=lambda _percent, _message: None,
        check_cancelled=lambda: None,
    )

    conversation = repository.load_conversation(conversation_id)
    reply_call = next(messages for step, messages in siliconflow.calls if step == "chat.reply")
    assert conversation is not None
    assert conversation.answer_policy == "strict_evidence"
    assert "严格证据" in reply_call[0]["content"]
    assert "档案未涉及而拒答" in reply_call[0]["content"]


def test_author_chat_can_use_web_without_local_knowledge_base(tmp_path) -> None:
    class NoLocalRetriever(FakeRetriever):
        def __init__(self) -> None:
            super().__init__()
            self.repository = FakeKBRepository()

        def search(self, _request, **_kwargs):
            raise AssertionError("无本地知识库模式不应调用本地检索")

    class FakeBocha:
        def search(self, query: str, *, count: int, check_cancelled=None):
            if check_cancelled is not None:
                check_cancelled()
            assert query == "最近有哪些值得讨论的变化"
            assert count == 4
            return BochaSearchResult(
                query=query,
                pages=(
                    BochaWebPage(
                        title="近期行业观察",
                        url="https://example.org/latest",
                        summary="一条可追溯的联网搜索摘要。",
                        site_name="示例站点",
                        date_published="2026-07-01",
                    ),
                ),
            )

    database = Database(tmp_path / "chat_web.db")
    database.initialize()
    repository = ChatRepository(database)
    local = NoLocalRetriever()
    service = AuthorChatService(
        repository=repository,
        persona_repository=FakePersonaRepository(),
        kb_repository=FakeKBRepository(),
        retriever=local,
        siliconflow=FakeSiliconFlow(),
        bocha=FakeBocha(),
        kb_id="kb_default",
        recent_rounds=lambda: 6,
        web_search_result_count=lambda: 4,
    )
    conversation_id = service.create_conversation(
        persona_id="persona_chat",
        knowledge_mode="none",
        use_web_search=True,
        selected_doc_ids=set(),
        allowed_persona_doc_ids=set(),
    )

    reply = service.send(
        conversation_id=conversation_id,
        user_message="最近有哪些值得讨论的变化",
        progress=lambda _percent, _message: None,
        check_cancelled=lambda: None,
    )

    conversation = repository.load_conversation(conversation_id)
    assert conversation is not None and conversation.use_web_search
    assert local.requests == []
    assert reply.message.sources[0].source_type == "web"
    assert reply.message.sources[0].url == "https://example.org/latest"


def test_stream_output_panel_separates_concurrent_labels_and_retries(qtbot) -> None:
    panel = StreamOutputPanel()
    qtbot.addWidget(panel)
    panel.append_stream("content::Map A", "A 的失败片段")
    panel.append_stream("content::Map B", "B 的完整输出")
    panel.append_stream(
        "attempt_reset::Map A",
        "第 1/2 次流式尝试未完整结束：连接中断",
    )
    panel.append_stream("content::Map A", "A 的重试输出")

    panel.call_combo.setCurrentIndex(panel.call_combo.findData("Map A"))
    assert "失败片段" not in panel.output_view.toPlainText()
    assert "第 1/2 次流式尝试未完整结束" in panel.output_view.toPlainText()
    assert "A 的重试输出" in panel.output_view.toPlainText()
    panel.call_combo.setCurrentIndex(panel.call_combo.findData("Map B"))
    assert panel.output_view.toPlainText() == "B 的完整输出"


def test_stream_output_panel_discards_only_uncommitted_cancelled_output(qtbot) -> None:
    panel = StreamOutputPanel()
    qtbot.addWidget(panel)
    stage = "认知 Map · 一篇名称较长的非虚构作品.pdf · abc123"
    panel.append_stream(f"content::{stage}", "已经完整完成")
    panel.append_stream(f"complete::{stage}", "done")
    panel.append_stream(f"content::{stage}", "不完整残片")

    panel.discard_incomplete_attempts()

    assert panel.output_view.toPlainText() == "已经完整完成"
    assert not hasattr(panel, "activity_label")
    assert panel.call_combo.toolTip().startswith(stage)
    assert "认知 Map" not in panel.call_combo.currentText()
    assert panel.call_combo.view().minimumWidth() > panel.call_combo.minimumWidth()


def test_stream_output_panel_selects_failed_file_and_keeps_failure_reason(qtbot) -> None:
    panel = StreamOutputPanel()
    qtbot.addWidget(panel)
    failed = "认知 Map · 失败论文.pdf · bad123"
    other = "认知 Map · 其他论文.pdf · ok1234"
    panel.append_stream(f"content::{failed}", "不完整残片")
    panel.append_stream(f"content::{other}", "另一调用的输出")
    panel.call_combo.setCurrentIndex(panel.call_combo.findData(other))

    panel.append_stream(f"error::{failed}", "Map Schema 缺少 mental_candidates")
    panel.discard_incomplete_attempts()

    assert panel.call_combo.currentData() == failed
    assert "失败论文.pdf" in panel.call_combo.currentText()
    assert "不完整残片" not in panel.output_view.toPlainText()
    assert "[失败原因] Map Schema 缺少 mental_candidates" in panel.output_view.toPlainText()


def test_cancelled_author_reply_persists_only_empty_interrupted_marker(tmp_path) -> None:
    cancelled = False

    class CancelledSiliconFlow(FakeSiliconFlow):
        def chat(self, messages, **kwargs):
            nonlocal cancelled
            if kwargs.get("step_id") == "chat.reply":
                cancelled = True
                raise TaskCancelled("planned cancellation")
            return super().chat(messages, **kwargs)

    def check_cancelled() -> None:
        if cancelled:
            raise TaskCancelled("planned cancellation")

    database = Database(tmp_path / "cancelled_chat.db")
    database.initialize()
    repository = ChatRepository(database)
    service = AuthorChatService(
        repository=repository,
        persona_repository=FakePersonaRepository(),
        kb_repository=FakeKBRepository(),
        retriever=FakeRetriever(),
        siliconflow=CancelledSiliconFlow(),
        kb_id="kb_default",
        recent_rounds=lambda: 4,
    )
    conversation_id = service.create_conversation(
        persona_id="persona_chat",
        knowledge_mode="none",
        selected_doc_ids=set(),
        allowed_persona_doc_ids=set(),
    )

    with pytest.raises(TaskCancelled):
        service.send(
            conversation_id=conversation_id,
            user_message="这条回答会被停止",
            progress=lambda _percent, _message: None,
            check_cancelled=check_cancelled,
        )

    messages = repository.list_messages(conversation_id)
    assert [(item.role, item.status, item.content) for item in messages] == [
        ("user", "complete", "这条回答会被停止"),
        ("assistant", "interrupted", ""),
    ]


def test_failed_author_reply_persists_reason_without_response_fragment(tmp_path) -> None:
    class FailedSiliconFlow(FakeSiliconFlow):
        def chat(self, messages, **kwargs):
            if kwargs.get("step_id") == "chat.reply":
                raise ValueError("作者回答结构处理失败")
            return super().chat(messages, **kwargs)

    database = Database(tmp_path / "failed_chat.db")
    database.initialize()
    repository = ChatRepository(database)
    service = AuthorChatService(
        repository=repository,
        persona_repository=FakePersonaRepository(),
        kb_repository=FakeKBRepository(),
        retriever=FakeRetriever(),
        siliconflow=FailedSiliconFlow(),
        kb_id="kb_default",
        recent_rounds=lambda: 4,
    )
    conversation_id = service.create_conversation(
        persona_id="persona_chat",
        knowledge_mode="none",
        selected_doc_ids=set(),
        allowed_persona_doc_ids=set(),
    )

    with pytest.raises(ValueError, match="作者回答结构处理失败"):
        service.send(
            conversation_id=conversation_id,
            user_message="这条回答会失败",
            progress=lambda _percent, _message: None,
            check_cancelled=lambda: None,
        )

    messages = repository.list_messages(conversation_id)
    assert messages[-1].status == "error"
    assert messages[-1].content == "失败原因：作者回答结构处理失败"
    assert "回答残片" not in messages[-1].content


def test_chat_transcript_renders_safe_markdown_and_message_metadata(qtbot) -> None:
    view = ChatTranscriptView()
    qtbot.addWidget(view)
    source = ChatSource(
        source_key="S1",
        doc_id="doc_one",
        chunk_id="chunk_one",
        filename="事实材料.pdf",
        excerpt="来源摘录",
        page_start=3,
    )
    view.set_messages(
        [
            ChatMessage(
                message_id="user_one",
                conversation_id="chat_one",
                sequence=1,
                role="user",
                content="请解释这个问题",
                created_at="2026-07-15T00:00:00+00:00",
            ),
            ChatMessage(
                message_id="assistant_one",
                conversation_id="chat_one",
                sequence=2,
                role="assistant",
                content="**核心判断**\n\n- 第一项\n- 第二项\n\n<script>危险内容</script>",
                sources=[source],
                verification={"overall_verdict": "supported", "note": "来源支持"},
                created_at="2026-07-15T00:00:01+00:00",
            ),
        ],
        "测试作者",
    )

    bold_cursor = view.document().find("核心判断")
    user_cursor = view.document().find("你")
    plain_text = view.toPlainText()

    assert bold_cursor.charFormat().fontWeight() >= QFont.Weight.Bold
    assert user_cursor.blockFormat().background().color().name() == "#e8edf2"
    assert "第一项" in plain_text
    assert "<script>危险内容</script>" in plain_text
    assert "本轮检索来源" in plain_text
    assert "[S1] 事实材料.pdf · 第3页" in plain_text
    assert "中性核验：supported 来源支持" in plain_text
    assert view.openExternalLinks() is False


def test_chat_transcript_throttles_stream_and_discards_failed_attempt(qtbot) -> None:
    view = ChatTranscriptView()
    qtbot.addWidget(view)
    view.start_turn("继续讨论", "测试作者")
    view.append_stream("失败的 **片段**")
    qtbot.waitUntil(lambda: "失败的" in view.toPlainText(), timeout=500)

    view.reset_stream_attempt()
    view.append_stream("新的 **完整回答**")
    qtbot.waitUntil(lambda: "完整回答" in view.toPlainText(), timeout=500)

    bold_cursor = view.document().find("完整回答")
    assert "失败的" not in view.toPlainText()
    assert bold_cursor.charFormat().fontWeight() >= QFont.Weight.Bold


def test_chat_transcript_marks_persisted_business_failure(qtbot) -> None:
    view = ChatTranscriptView()
    qtbot.addWidget(view)
    view.set_messages(
        [
            ChatMessage(
                message_id="assistant_failed",
                conversation_id="chat_failed",
                sequence=2,
                role="assistant",
                content="失败原因：作者回答结构处理失败",
                status="error",
                created_at="2026-07-16T00:00:00+00:00",
            )
        ],
        "测试作者",
    )

    header = view.document().find("测试作者（失败）")
    assert not header.isNull()
    assert header.blockFormat().background().color().name() == "#fdebea"
    assert "作者回答结构处理失败" in view.toPlainText()


def test_chat_transcript_preserves_scroll_position_when_follow_tail_is_off(qtbot) -> None:
    view = ChatTranscriptView()
    qtbot.addWidget(view)
    view.resize(420, 180)
    view.show()
    view.start_turn("查看较早内容", "测试作者")
    view.append_stream("\n\n".join(f"第 {index} 段内容" for index in range(80)))
    qtbot.waitUntil(lambda: view.verticalScrollBar().maximum() > 0, timeout=500)
    view.set_auto_scroll(False)
    scrollbar = view.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum() // 3)
    expected = scrollbar.value()

    view.append_stream("\n\n新增的流式内容")
    qtbot.waitUntil(lambda: "新增的流式内容" in view.toPlainText(), timeout=500)

    assert scrollbar.value() == expected


def test_chat_composer_submits_on_enter_and_modified_enter_inserts_newline(qtbot) -> None:
    composer = SubmitTextEdit()
    qtbot.addWidget(composer)
    submitted: list[str] = []
    composer.submit_requested.connect(lambda: submitted.append(composer.toPlainText()))
    composer.setPlainText("第一行")
    composer.moveCursor(QTextCursor.MoveOperation.End)

    qtbot.keyPress(composer, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    qtbot.keyPress(composer, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)

    assert composer.toPlainText() == "第一行\n\n"
    assert submitted == []

    qtbot.keyPress(composer, Qt.Key.Key_Enter)

    assert submitted == ["第一行\n\n"]


def test_navigation_help_tutorial_and_author_chat_history(qtbot) -> None:
    conversations: list[dict[str, object]] = []
    messages = {}

    def create_conversation(**_kwargs):
        conversations.append(
            {
                "conversation_id": "chat_one",
                "persona_id": "persona_chat",
                "persona_name": "测试作者",
                "persona_version": 1,
                "title": "新对话",
                "knowledge_mode": "none",
                "updated_at": "2026-07-15T00:00:00+00:00",
            }
        )
        return "chat_one"

    def load_conversation(identifier: str):
        if identifier != "chat_one":
            return None
        return ChatConversation(
            conversation_id="chat_one",
            kb_id="kb_default",
            persona_id="persona_chat",
            persona_name="测试作者",
            persona_version=1,
            title="测试问题",
            knowledge_mode="none",
            runtime_persona={},
            created_at="2026-07-15T00:00:00+00:00",
            updated_at="2026-07-15T00:00:00+00:00",
        )

    def send_message(**kwargs):
        message = ChatMessage(
            message_id="assistant_one",
            conversation_id="chat_one",
            sequence=2,
            role="assistant",
            content="测试回答",
            created_at="2026-07-15T00:00:01+00:00",
        )
        messages["chat_one"] = [message]
        return ChatReply(message=message)

    window = MainWindow(
        lambda: ChatResult(content="OK", model="fake"),
        list_personas=lambda: [
            {
                "persona_id": "persona_chat",
                "name": "测试作者",
                "version_number": 1,
                "status": "ready",
            }
        ],
        list_chat_conversations=lambda: conversations,
        load_chat_conversation=load_conversation,
        create_chat_conversation=create_conversation,
        list_chat_messages=lambda identifier: messages.get(identifier, []),
        send_chat_message=send_message,
    )
    qtbot.addWidget(window)
    assert window.author_chat_page.answer_policy_combo.currentData() == "general_assisted"
    assert not window.author_chat_page.web_search_checkbox.isChecked()
    window.show()
    qtbot.wait(0)
    page_titles = window.findChildren(QLabel, "pageTitle")
    labels = [window.navigation.item(index).text() for index in range(window.navigation.count())]
    assert page_titles
    assert all(title.font().weight() >= QFont.Weight.Bold for title in page_titles)
    assert labels == ["项目", "知识库", "作者档案", "作者对话", "写作任务", "设置", "教程"]
    assert window.project_page.help_button.accessibleName() == "项目帮助"
    assert window.knowledge_page.help_button.accessibleName() == "知识库帮助"
    assert window.persona_page.help_button.accessibleName() == "作者档案帮助"
    assert window.writing_task_page.help_button.accessibleName() == "写作任务帮助"
    assert window.tutorial_page.chapter_list.count() == 8
    window.navigation.setCurrentRow(3)
    assert window.pages.currentWidget() is window.author_chat_page
    window.author_chat_page.persona_combo.setCurrentIndex(1)
    window.author_chat_page.message_input.setPlainText("测试问题")
    qtbot.keyPress(window.author_chat_page.message_input, Qt.Key.Key_Return)
    assert conversations[0]["conversation_id"] == "chat_one"
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)
