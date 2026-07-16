"""Retrieval-grounded, streaming author chat with rolling summary memory."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from writing_factory.chat.models import (
    AnswerPolicy,
    ChatConversation,
    ChatReply,
    ChatSource,
    ChatVerificationResult,
    KnowledgeMode,
)
from writing_factory.chat.prompts import (
    reply_messages,
    summary_messages,
    verification_messages,
)
from writing_factory.chat.repository import ChatRepository
from writing_factory.eval.injection import InjectionDetector
from writing_factory.kb.models import MetadataFilter, RetrievalRequest

if TYPE_CHECKING:
    from writing_factory.kb.retrieval import HybridRetriever
    from writing_factory.llm import SiliconFlowClient
    from writing_factory.store.kb_repository import KnowledgeBaseRepository
    from writing_factory.store.persona_repository import PersonaRepository

ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]
logger = logging.getLogger(__name__)


class AuthorChatService:
    """Perform one optional direct retrieval and one persona-grounded answer call."""

    def __init__(
        self,
        *,
        repository: ChatRepository,
        persona_repository: PersonaRepository,
        kb_repository: KnowledgeBaseRepository,
        retriever: HybridRetriever,
        siliconflow: SiliconFlowClient,
        bocha=None,
        kb_id: str,
        recent_rounds: Callable[[], int],
        web_search_result_count: Callable[[], int] = lambda: 5,
    ) -> None:
        self.repository = repository
        self.persona_repository = persona_repository
        self.kb_repository = kb_repository
        self.retriever = retriever
        self.siliconflow = siliconflow
        self.bocha = bocha
        self.kb_id = kb_id
        self.recent_rounds = recent_rounds
        self.web_search_result_count = web_search_result_count

    def create_conversation(
        self,
        *,
        persona_id: str,
        knowledge_mode: KnowledgeMode,
        answer_policy: AnswerPolicy = "general_assisted",
        use_web_search: bool = False,
        selected_doc_ids: set[str],
        allowed_persona_doc_ids: set[str],
    ) -> str:
        runtime = self.persona_repository.load_runtime(persona_id)
        if runtime is None:
            raise ValueError("请选择一个已经完成的作者档案")
        records = self.persona_repository.list_personas(self.kb_id)
        record = next(
            (item for item in records if item.get("persona_id") == persona_id),
            None,
        )
        if record is None:
            raise ValueError("作者档案版本不存在")
        roles = self.persona_repository.load_source_roles(persona_id)
        if roles is not None:
            target_ids = set(roles.target_doc_ids)
        else:
            target_ids = self._legacy_persona_source_ids(persona_id)
        payload = runtime.model_dump(mode="json")
        payload.pop("composition_dna", None)
        return self.repository.create_conversation(
            kb_id=self.kb_id,
            persona_id=persona_id,
            persona_name=str(record.get("name", runtime.name)),
            persona_version=int(record.get("version_number", 1)),
            runtime_persona=payload,
            knowledge_mode=knowledge_mode,
            answer_policy=answer_policy,
            use_web_search=use_web_search,
            selected_doc_ids=selected_doc_ids,
            allowed_persona_doc_ids=allowed_persona_doc_ids & target_ids,
            target_persona_doc_ids=target_ids,
        )

    def send(
        self,
        *,
        conversation_id: str,
        user_message: str,
        progress: ProgressCallback,
        check_cancelled: CancellationCheck,
    ) -> ChatReply:
        conversation = self._require_conversation(conversation_id)
        text = user_message.strip()
        if not text:
            raise ValueError("请输入对话内容")
        user_turn = self.repository.append_message(
            conversation_id=conversation_id,
            role="user",
            content=text,
        )
        if user_turn.sequence == 1:
            self.repository.rename_conversation(conversation_id, text[:36])
        sources: list[ChatSource] = []
        try:
            check_cancelled()
            progress(10, "准备作者对话上下文")
            sources = self._retrieve(conversation, text, progress, check_cancelled)
            messages = self.repository.list_messages(conversation_id)
            recent_limit = max(1, min(20, self.recent_rounds())) * 2
            earlier = [item for item in messages if item.sequence < user_turn.sequence]
            recent = earlier[-recent_limit:]
            recent_payload = [
                {"role": item.role, "content": item.content}
                for item in recent
                if item.status == "complete"
            ]
            check_cancelled()
            progress(55, "流式生成作者回答")
            result = self.siliconflow.chat(
                reply_messages(
                    runtime_persona=conversation.runtime_persona,
                    summary=conversation.summary_text,
                    recent_messages=recent_payload,
                    sources=sources,
                    user_message=text,
                    answer_policy=conversation.answer_policy,
                ),
                thinking=False,
                temperature=0.6,
                max_tokens=8192,
                use_cache=False,
                request_attempts=2,
                stream=True,
                step_id="chat.reply",
            )
            check_cancelled()
        except Exception as exc:
            if _cancelled(check_cancelled):
                self.repository.append_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content="",
                    status="interrupted",
                )
            else:
                self.repository.append_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=f"失败原因：{_safe_failure_reason(exc)}",
                    status="error",
                )
            raise
        content = self._remove_unknown_source_keys(result.content, sources)
        assistant = self.repository.append_message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            status="complete",
            sources=sources,
        )
        progress(90, "更新较早对话摘要")
        try:
            self._update_summary(conversation_id, check_cancelled)
        except Exception as exc:
            check_cancelled()
            logger.warning("Author chat summary deferred: %s", type(exc).__name__)
            progress(96, "回答已保存，较早对话摘要将在后续重试")
        progress(100, "作者回答完成")
        return ChatReply(message=assistant, retrieved_sources=sources)

    def verify(self, message_id: str) -> ChatVerificationResult:
        message = self.repository.load_message(message_id)
        if message is None or message.role != "assistant":
            raise ValueError("只能核验已经保存的作者回答")
        if not message.sources:
            result = ChatVerificationResult(
                overall_verdict="insufficient",
                note="本条回答没有知识库来源，无法进行事实支持度核验。",
            )
        else:
            response = self.siliconflow.chat(
                verification_messages(answer=message.content, sources=message.sources),
                thinking=False,
                temperature=0.0,
                max_tokens=8192,
                response_format="json_object",
                use_cache=True,
                request_attempts=2,
                stream=False,
                step_id="chat.verify",
            )
            result = ChatVerificationResult.model_validate_json(response.content)
            allowed = {item.source_key for item in message.sources}
            if any(set(assessment.source_keys) - allowed for assessment in result.assessments):
                raise ValueError("核验结果引用了本条回答之外的来源键")
        self.repository.save_verification(message_id, result.model_dump(mode="json"))
        return result

    def _retrieve(
        self,
        conversation: ChatConversation,
        query: str,
        progress: ProgressCallback,
        check_cancelled: CancellationCheck,
    ) -> list[ChatSource]:
        if conversation.knowledge_mode == "none" and not conversation.use_web_search:
            progress(45, "本轮不使用知识库")
            return []
        ready = {
            str(item["doc_id"])
            for item in self.kb_repository.list_documents(conversation.kb_id)
            if item.get("status") == "ready"
        }
        selected = (
            ready
            if conversation.knowledge_mode == "all"
            else ready & set(conversation.selected_doc_ids)
        )
        target = set(conversation.target_persona_doc_ids)
        explicitly_allowed = set(conversation.allowed_persona_doc_ids) & target
        allowed = selected - (target - explicitly_allowed)
        if not allowed and not conversation.use_web_search:
            raise ValueError("本会话没有可用知识库文档；所选文档可能均被作者语料隔离")
        check_cancelled()
        progress(20, "直接检索并重排知识库")
        active_retriever = self.retriever
        if conversation.use_web_search:
            if self.bocha is None:
                raise ValueError("已启用联网检索，但博查 API 客户端不可用")
            bocha_transport = getattr(self.bocha, "transport", None)
            if bocha_transport is not None and not bocha_transport.credential_configured:
                raise ValueError("已启用联网检索，请先在设置页配置博查 API Key")
            from writing_factory.kb.web_retrieval import WebAugmentedRetriever

            active_retriever = WebAugmentedRetriever(
                self.retriever,
                self.bocha,
                result_count=self.web_search_result_count(),
            )
        result = active_retriever.search(
            RetrievalRequest(
                kb_id=conversation.kb_id,
                query=query,
                filters=MetadataFilter(doc_ids=allowed),
                use_rewrite=False,
                use_hyde=False,
                use_rerank=True,
                top_k=8,
            ),
            progress=lambda percent, message: progress(20 + round(percent * 0.25), message),
            check_cancelled=check_cancelled,
        )
        content = "\n\n".join(item.text for item in result.hits)
        InjectionDetector().enforce(self.siliconflow, content)
        filenames = {
            str(item["doc_id"]): str(item.get("filename", ""))
            for item in self.kb_repository.list_documents(conversation.kb_id)
        }
        return [
            ChatSource(
                source_key=f"S{index}",
                doc_id=hit.doc_id,
                chunk_id=hit.chunk_id,
                filename=(
                    hit.title or hit.site_name or "联网网页"
                    if hit.source == "web"
                    else filenames.get(hit.doc_id, hit.doc_id)
                ),
                source_type="web" if hit.source == "web" else "local",
                url=hit.url,
                site_name=hit.site_name,
                date_published=hit.date_published,
                excerpt=hit.text[:3000],
                page_start=hit.page_start,
                page_end=hit.page_end,
                section_heading=hit.section_heading,
            )
            for index, hit in enumerate(result.hits, start=1)
        ]

    def _update_summary(
        self,
        conversation_id: str,
        check_cancelled: CancellationCheck,
    ) -> None:
        conversation = self._require_conversation(conversation_id)
        messages = [
            item
            for item in self.repository.list_messages(conversation_id)
            if item.status == "complete"
        ]
        keep = max(1, min(20, self.recent_rounds())) * 2
        cutoff = len(messages) - keep
        if cutoff <= 0:
            return
        archived = [
            item
            for item in messages[:cutoff]
            if item.sequence > conversation.summary_through_sequence
        ]
        if not archived:
            return
        check_cancelled()
        result = self.siliconflow.chat(
            summary_messages(
                previous_summary=conversation.summary_text,
                archived_messages=[
                    {
                        "sequence": item.sequence,
                        "role": item.role,
                        "content": item.content,
                    }
                    for item in archived
                ],
            ),
            thinking=False,
            temperature=0.0,
            max_tokens=4096,
            use_cache=False,
            request_attempts=2,
            stream=False,
            step_id="chat.summary",
        )
        self.repository.save_summary(
            conversation_id,
            result.content.strip(),
            archived[-1].sequence,
        )

    def _require_conversation(self, conversation_id: str) -> ChatConversation:
        conversation = self.repository.load_conversation(conversation_id)
        if conversation is None:
            raise ValueError("作者对话不存在或已被删除")
        return conversation

    def _legacy_persona_source_ids(self, persona_id: str) -> set[str]:
        """Conservatively isolate all sources when an old profile lacks role metadata."""

        loader = getattr(self.persona_repository, "load_ready", None)
        if not callable(loader):
            return set()
        loaded = loader(persona_id)
        if loaded is None:
            return set()
        persona, _markdown = loaded
        return {item.doc_id for item in persona.source_info}

    @staticmethod
    def _remove_unknown_source_keys(content: str, sources: list[ChatSource]) -> str:
        allowed = {item.source_key for item in sources}

        def replace(match: re.Match[str]) -> str:
            return match.group(0) if match.group(1) in allowed else ""

        return re.sub(r"\[(S\d+)\]", replace, content)


def _cancelled(check: CancellationCheck) -> bool:
    try:
        check()
    except Exception:
        return True
    return False


def _safe_failure_reason(error: Exception) -> str:
    detail = str(error).strip() or type(error).__name__
    return detail[:1200]
