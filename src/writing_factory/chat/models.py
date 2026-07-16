"""Typed contracts for persistent author conversations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

KnowledgeMode = Literal["none", "all", "selected"]
AnswerPolicy = Literal["general_assisted", "strict_evidence"]
ChatMessageStatus = Literal["complete", "interrupted", "error"]
ChatVerificationVerdict = Literal["supported", "partial", "unsupported", "insufficient"]


class ChatSource(BaseModel):
    """One retrieval result snapshotted with an assistant message."""

    model_config = ConfigDict(frozen=True)

    source_key: str
    doc_id: str
    chunk_id: str
    filename: str
    excerpt: str
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    source_type: Literal["local", "web"] = "local"
    url: str | None = None
    site_name: str | None = None
    date_published: str | None = None


class ChatMessage(BaseModel):
    """One persisted user or assistant turn."""

    model_config = ConfigDict(frozen=True)

    message_id: str
    conversation_id: str
    sequence: int = Field(ge=1)
    role: Literal["user", "assistant"]
    content: str
    status: ChatMessageStatus = "complete"
    sources: list[ChatSource] = Field(default_factory=list)
    verification: dict[str, object] | None = None
    created_at: str


class ChatConversation(BaseModel):
    """Pinned persona version and memory state for one conversation."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    kb_id: str
    persona_id: str | None = None
    persona_name: str
    persona_version: int = Field(ge=1)
    title: str
    knowledge_mode: KnowledgeMode = "none"
    answer_policy: AnswerPolicy = "general_assisted"
    use_web_search: bool = False
    selected_doc_ids: list[str] = Field(default_factory=list)
    allowed_persona_doc_ids: list[str] = Field(default_factory=list)
    target_persona_doc_ids: list[str] = Field(default_factory=list)
    runtime_persona: dict[str, object]
    summary_text: str = ""
    summary_through_sequence: int = Field(default=0, ge=0)
    created_at: str
    updated_at: str


class ChatReply(BaseModel):
    """Completed assistant turn and the source cards used for it."""

    model_config = ConfigDict(frozen=True)

    message: ChatMessage
    retrieved_sources: list[ChatSource] = Field(default_factory=list)


class ClaimVerification(BaseModel):
    """Neutral support judgment for one factual statement."""

    model_config = ConfigDict(frozen=True)

    claim: str
    source_keys: list[str] = Field(default_factory=list)
    verdict: ChatVerificationVerdict
    rationale: str


class ChatVerificationResult(BaseModel):
    """On-demand neutral check of one assistant message."""

    model_config = ConfigDict(frozen=True)

    overall_verdict: ChatVerificationVerdict
    assessments: list[ClaimVerification] = Field(default_factory=list)
    note: str = ""
