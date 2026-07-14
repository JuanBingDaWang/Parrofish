"""写作流水线及其事实来源隔离契约。

生成流水线五步：
    1. 选题 (topic_selection) → ThesisStatement
    2. 框架 (framework)    → AnnotatedOutline
    3. 起草 (drafting)     → SectionDraft
    4. 核对 (verification) → VerifiedDraft
    5. 打磨 (polishing)    → PolishedDraft → PolishedSection
"""

# ── 数据模型 ──────────────────────────────────────────────────
from writing_factory.generate.drafting import (
    build_evidence_pack_for_section,
    draft_section,
)
from writing_factory.generate.framework import build_framework
from writing_factory.generate.models import (
    AnnotatedOutline,
    CitationStyle,
    Claim,
    ClaimType,
    EvidenceItem,
    EvidencePack,
    GenerationContext,
    GenerationOptions,
    OutlineEvidence,
    OutlineNode,
    PolishedDraft,
    PolishedSection,
    ReferenceItem,
    ReferenceList,
    SectionDraft,
    SectionDraftOutput,
    ThesisStatement,
    VerificationDecision,
    VerificationResponse,
    VerificationVerdict,
    VerifiedClaim,
    VerifiedDraft,
)
from writing_factory.generate.polishing import (
    assemble_polished_draft,
    polish_section,
)

# ── source_policy ─────────────────────────────────────────────
from writing_factory.generate.source_policy import (
    GenerationSourcePolicy,
    build_generation_source_policy,
    find_suspicious_source_overlap,
    task_document_filter,
)

# ── 流水线函数 ────────────────────────────────────────────────
from writing_factory.generate.topic_selection import select_topic
from writing_factory.generate.verification import verify_section

__all__ = [
    # ── 数据模型 ──
    "AnnotatedOutline",
    "CitationStyle",
    "Claim",
    "ClaimType",
    "EvidenceItem",
    "EvidencePack",
    "GenerationContext",
    "GenerationOptions",
    "OutlineEvidence",
    "OutlineNode",
    "PolishedDraft",
    "PolishedSection",
    "ReferenceItem",
    "ReferenceList",
    "SectionDraft",
    "SectionDraftOutput",
    "ThesisStatement",
    "VerifiedClaim",
    "VerifiedDraft",
    "VerificationDecision",
    "VerificationResponse",
    "VerificationVerdict",
    # ── 选题 ──
    "select_topic",
    # ── 框架 ──
    "build_framework",
    # ── 起草 ──
    "build_evidence_pack_for_section",
    "draft_section",
    # ── 核对 ──
    "verify_section",
    # ── 打磨 ──
    "assemble_polished_draft",
    "polish_section",
    # ── source_policy ──
    "GenerationSourcePolicy",
    "build_generation_source_policy",
    "find_suspicious_source_overlap",
    "task_document_filter",
]
