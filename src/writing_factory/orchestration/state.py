"""LangGraph 写作循环状态定义。

所有状态值均为 JSON 可序列化的 dict/list/primitive，确保 SQLite checkpointer
可以正确持久化和恢复。
"""

from __future__ import annotations

from typing import TypedDict


class SectionState(TypedDict, total=False):
    """单节在写作循环中的状态。"""

    section_id: str
    heading: str
    # pending → drafting → drafted → verifying → verified → revising
    # → polishing → polished | error
    status: str
    # 序列化的中间产物
    draft_json: str | None  # SectionDraft.model_dump_json()
    evidence_pack_json: str | None  # EvidencePack.model_dump_json()
    verified_draft_json: str | None  # VerifiedDraft.model_dump_json()
    polished_section_json: str | None  # PolishedSection.model_dump_json()
    # 修订计数（防止无限循环）
    revision_count: int
    source_key_offset: int
    target_length_chars: int
    elapsed_seconds: float
    # 上一节结论（用于节间衔接）
    previous_conclusion: str | None
    # 下一节目的（用于铺垫）
    next_purpose: str | None
    # 错误信息
    error: str | None


class WritingState(TypedDict, total=False):
    """全篇写作循环状态，由 LangGraph SQLite checkpointer 持久化。

    设计原则：
    - 所有 Pydantic 模型序列化为 JSON 字符串存储
    - 节点函数在边界处进行序列化/反序列化
    - 状态更新通过返回部分 dict 完成
    """

    # ── 配置（只读） ──
    context_json: str  # GenerationContext.model_dump_json()
    persona_id: str
    kb_id: str

    # ── 流水线产物 ──
    thesis_json: str | None  # ThesisStatement.model_dump_json()
    outline_json: str | None  # AnnotatedOutline.model_dump_json()

    # ── 逐节循环 ──
    sections: list[dict]  # list[SectionState]
    current_section_index: int

    # ── 跨节累积状态 ──
    term_registry_json: str  # JSON dict[str, str]
    source_key_counter: int  # 单调递增，全篇唯一
    accumulated_evidence_json: str  # JSON list[dict] — EvidenceItem.model_dump()
    claims_made_json: str  # JSON list[str] — 已核对并写入前文的论断

    # ── 最终组装 ──
    reference_list_json: str | None  # ReferenceList.model_dump_json()
    final_draft_json: str | None  # PolishedDraft.model_dump_json()

    # ── 阶段 6：一致性与全局打磨 ──
    term_consistency_json: str | None  # TermConsistencyReport.model_dump_json()
    structure_review_json: str | None  # StructureReview.model_dump_json()
    global_polish_json: str | None  # GlobalPolishResult.model_dump_json()

    # ── 控制 ──
    # Pipeline status constants are declared below.
    status: str
    error: str | None


# ── 状态常量 ──────────────────────────────────────────────────

SECTION_STATUS_PENDING = "pending"
SECTION_STATUS_DRAFTING = "drafting"
SECTION_STATUS_DRAFTED = "drafted"
SECTION_STATUS_VERIFYING = "verifying"
SECTION_STATUS_VERIFIED = "verified"
SECTION_STATUS_REVISING = "revising"
SECTION_STATUS_POLISHING = "polishing"
SECTION_STATUS_POLISHED = "polished"
SECTION_STATUS_ERROR = "error"

PIPELINE_STATUS_IDLE = "idle"
PIPELINE_STATUS_TOPIC = "topic_selecting"
PIPELINE_STATUS_FRAMEWORK = "framework_building"
PIPELINE_STATUS_EVIDENCE_PREFETCH = "evidence_prefetch"
PIPELINE_STATUS_DRAFTING = "drafting"
PIPELINE_STATUS_VERIFYING = "verifying"
PIPELINE_STATUS_REVISING = "revising"
PIPELINE_STATUS_POLISHING = "polishing"
PIPELINE_STATUS_TERM_REVIEW = "term_review"
PIPELINE_STATUS_STRUCTURE_REVIEW = "structure_review"
PIPELINE_STATUS_GLOBAL_POLISH = "global_polish"
PIPELINE_STATUS_ASSEMBLING = "assembling"
PIPELINE_STATUS_DONE = "done"
PIPELINE_STATUS_ERROR = "error"

# 每节最多修订次数
MAX_REVISIONS_PER_SECTION = 3
