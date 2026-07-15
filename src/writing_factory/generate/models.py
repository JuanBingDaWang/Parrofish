"""阶段 4 生成流水线数据模型 —— 选题、框架、起草、核对、打磨。

所有模型均为 frozen Pydantic，遵循项目统一的数据契约约定。
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from writing_factory.nonfiction import NonfictionGenre

# ---------------------------------------------------------------------------
# 枚举与字面量类型
# ---------------------------------------------------------------------------

ClaimType = Literal["fact", "interpretation", "common"]
"""论断类型：
- fact: 可验证的事实性陈述，必须绑定 source_key
- interpretation: 作者的分析/推理/论证，不需要 source_key
- common: 学界常识，不需要 source_key
"""

VerificationVerdict = Literal["supported", "partial", "unsupported"]
"""核对结果：
- supported: chunk 原文完全支持该论断
- partial: chunk 原文部分支持，但存在偏差或遗漏
- unsupported: chunk 原文不支持该论断
"""

CitationStyle = Literal["gb-t-7714", "apa", "mla"]
QualityPreset = Literal["strict", "balanced", "fast_draft", "custom"]
GenerationQualityStatus = Literal["verified_final", "unverified_draft"]
DocumentForm = Literal["paragraph", "short_text", "paper"]
CitationDisplay = Literal["auto", "bibliography", "internal_only"]
ResolvedCitationDisplay = Literal["bibliography", "internal_only"]


def drafting_unit_range(
    document_form: DocumentForm,
    target_length_chars: int,
) -> tuple[int, int]:
    """Return the permitted number of leaf drafting units for one output form."""

    if document_form == "paragraph":
        return 1, 1
    if document_form == "short_text":
        return (1, 3) if target_length_chars <= 2500 else (2, 4)
    if target_length_chars <= 2000:
        return 3, 5
    if target_length_chars <= 5000:
        return 4, 8
    ideal = max(6, min(14, round(target_length_chars / 750)))
    return max(5, ideal - 2), min(16, ideal + 2)


class GenerationOptions(BaseModel):
    """Per-task quality and cost controls persisted with every checkpoint."""

    model_config = ConfigDict(frozen=True)

    preset: QualityPreset = Field(default="strict", description="质量预设")
    document_form: DocumentForm = Field(
        default="paper", description="文稿长度与层级形态；paper 是旧版长文标识"
    )
    genre: NonfictionGenre = Field(default="general_nonfiction", description="目标非虚构文体")
    citation_display: CitationDisplay = Field(
        default="auto", description="引用在最终成稿中的显示方式"
    )
    target_length_chars: int = Field(default=5000, ge=500, le=100000, description="目标中文字数")
    use_hyde: bool = Field(default=True, description="是否在写作检索中使用 HyDE 假设文档")
    use_query_rewrite: bool = Field(default=True, description="是否在写作检索中使用查询改写")
    topic_refinement: bool = Field(default=True, description="是否使用 LLM 锐化选题")
    framework_generation: bool = Field(default=True, description="是否使用 LLM 构建提纲")
    fact_verification: bool = Field(default=True, description="是否执行 LLM 事实语义核验")
    section_polish: bool = Field(default=True, description="是否执行逐节文风打磨")
    section_drift_check: bool = Field(default=True, description="是否核对逐节打磨后的事实漂移")
    term_review: bool = Field(default=True, description="是否执行全文术语审查")
    structure_review: bool = Field(default=True, description="是否执行全文结构审查")
    global_polish: bool = Field(default=True, description="是否执行全局一致性打磨")
    global_drift_check: bool = Field(default=True, description="是否核对全局打磨后的事实漂移")

    @model_validator(mode="before")
    @classmethod
    def apply_missing_preset_defaults(cls, value: object) -> object:
        """Give legacy task JSON the current preset semantics without overriding saved choices."""

        if not isinstance(value, dict):
            return value
        compatibility_defaults: dict[str, object] = {}
        if "genre" not in value:
            compatibility_defaults["genre"] = (
                "academic_paper"
                if value.get("document_form", "paper") == "paper"
                else "general_nonfiction"
            )
        if "citation_display" not in value:
            compatibility_defaults["citation_display"] = "bibliography"
        preset = value.get("preset", "strict")
        if preset == "balanced":
            defaults = {
                "section_polish": False,
                "section_drift_check": False,
                "term_review": False,
                "structure_review": False,
                "global_polish": False,
                "global_drift_check": False,
            }
        elif preset == "fast_draft":
            defaults = {
                "use_hyde": False,
                "use_query_rewrite": False,
                "topic_refinement": False,
                "framework_generation": False,
                "fact_verification": False,
                "section_polish": False,
                "section_drift_check": False,
                "term_review": False,
                "structure_review": False,
                "global_polish": False,
                "global_drift_check": False,
            }
        else:
            defaults = {}
        return {**compatibility_defaults, **defaults, **value}

    @classmethod
    def from_preset(
        cls,
        preset: QualityPreset,
        *,
        target_length_chars: int,
        document_form: DocumentForm = "paper",
        genre: NonfictionGenre = "general_nonfiction",
        citation_display: CitationDisplay = "auto",
    ) -> GenerationOptions:
        """Build one of the supported user-facing quality profiles."""

        if preset == "strict":
            return cls(
                target_length_chars=target_length_chars,
                document_form=document_form,
                genre=genre,
                citation_display=citation_display,
            )
        if preset == "balanced":
            return cls(
                preset=preset,
                document_form=document_form,
                genre=genre,
                citation_display=citation_display,
                target_length_chars=target_length_chars,
                section_polish=False,
                section_drift_check=False,
                term_review=False,
                structure_review=False,
                global_polish=False,
                global_drift_check=False,
            )
        if preset == "fast_draft":
            return cls(
                preset=preset,
                document_form=document_form,
                genre=genre,
                citation_display=citation_display,
                target_length_chars=target_length_chars,
                use_hyde=False,
                use_query_rewrite=False,
                topic_refinement=False,
                framework_generation=False,
                fact_verification=False,
                section_polish=False,
                section_drift_check=False,
                term_review=False,
                structure_review=False,
                global_polish=False,
                global_drift_check=False,
            )
        return cls(
            preset=preset,
            document_form=document_form,
            genre=genre,
            citation_display=citation_display,
            target_length_chars=target_length_chars,
        )

    @property
    def quality_status(self) -> GenerationQualityStatus:
        """Only fully checked transformation chains qualify as verified final copy."""

        transformations_checked = (
            (not self.section_polish or self.section_drift_check)
            and (not self.global_polish or self.global_drift_check)
        )
        return (
            "verified_final"
            if self.fact_verification and transformations_checked
            else "unverified_draft"
        )

    @property
    def resolved_citation_display(self) -> ResolvedCitationDisplay:
        """Keep formal citations visible only where the genre normally expects them."""

        if self.citation_display != "auto":
            return self.citation_display
        return (
            "bibliography"
            if self.genre in {"academic_paper", "research_report", "policy_brief"}
            else "internal_only"
        )


# ---------------------------------------------------------------------------
# 4a — 选题
# ---------------------------------------------------------------------------


class ThesisStatement(BaseModel):
    """Persona 锐化后、经 KB 可行性验证的非虚构创作意图。"""

    model_config = ConfigDict(frozen=True)

    suggested_title: str = Field(default="", description="建议文稿标题，使用简体中文")
    thesis_text: str = Field(description="一句话中心信息或核心论点，使用简体中文")
    angle: str = Field(description="persona 选择的独特切入角度，使用简体中文")
    kb_support_assessment: str = Field(
        description="KB 检索后对论点可行性的评估：证据是否充足、哪些方面薄弱，使用简体中文"
    )
    persona_id: str = Field(description="生成该论点的 persona 标识")
    genre: NonfictionGenre = Field(default="general_nonfiction", description="目标非虚构文体")
    purpose: str = Field(default="传达中心信息", description="本次文本的沟通目的")
    audience: str = Field(default="一般读者", description="目标受众")
    desired_effect: str = Field(default="准确理解", description="希望读者获得的认识或行动效果")


# ---------------------------------------------------------------------------
# 4a — 框架
# ---------------------------------------------------------------------------


class OutlineEvidence(BaseModel):
    """Code-attached evidence candidate for one outline node."""

    model_config = ConfigDict(frozen=True)

    source_key: str
    chunk_id: str
    doc_id: str
    verbatim_excerpt: str
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None


class OutlineNode(BaseModel):
    """提纲中的一个节点，带修辞目的与候选证据映射。"""

    model_config = ConfigDict(frozen=True)

    node_id: str = Field(description="节点唯一标识，如 '1'、'1.1'、'2.3'")
    heading: str = Field(description="本内容单元标题；无需标题时可为空，使用简体中文")
    rhetorical_purpose: str = Field(
        description="本内容单元在全文中的功能，如提出问题、解释概念、举证、比较、建议或收束"
    )
    relation_to_previous: str = Field(
        default="", description="与前一内容单元的承接、递进、转折等关系"
    )
    candidate_source_keys: list[str] = Field(
        default_factory=list,
        description="候选证据 source_key 列表，如 [S1], [S2]；由框架阶段检索后填入",
    )
    candidate_evidence: list[OutlineEvidence] = Field(
        default_factory=list,
        description="代码逐节点检索后锁定的候选证据与全局 source_key",
    )
    children: list[OutlineNode] = Field(default_factory=list, description="子节点，支持递归嵌套")


class AnnotatedOutline(BaseModel):
    """Persona 定论证骨架 + 逐节点检索证据映射后的带注释提纲。"""

    model_config = ConfigDict(frozen=True)

    thesis: ThesisStatement = Field(description="锚定论点")
    root_nodes: list[OutlineNode] = Field(description="顶层提纲节点")
    term_registry: dict[str, str] = Field(
        default_factory=dict,
        description="术语登记表：术语名 → 定义，用于保持全文术语一致性",
    )
    kb_id: str = Field(description="检索所用的知识库标识")


# ---------------------------------------------------------------------------
# 4b — 起草（证据包 + 结构化草稿）
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """证据包中的单条证据：逐字摘录 + source_key + chunk 溯源。"""

    model_config = ConfigDict(frozen=True)

    source_key: str = Field(description="本节内唯一 source_key，如 [S1]、[S2]")
    chunk_id: str = Field(description="KB 中该 chunk 的标识，用于溯源核对")
    doc_id: str = Field(description="来源文档标识")
    verbatim_excerpt: str = Field(
        description="从 chunk 原文中逐字摘录的文本片段，不得改写；用于起草时的事实锚定"
    )
    page_start: int | None = Field(default=None, description="起始页码")
    page_end: int | None = Field(default=None, description="结束页码")
    section_heading: str | None = Field(default=None, description="所属章节标题")


class EvidencePack(BaseModel):
    """起草阶段喂给 LLM 的逐节证据包。"""

    model_config = ConfigDict(frozen=True)

    section_id: str = Field(description="对应的提纲节点标识")
    items: list[EvidenceItem] = Field(default_factory=list, description="本节可用证据列表")

    @model_validator(mode="after")
    def validate_unique_source_keys(self) -> EvidencePack:
        """Keep the code-assigned source-key namespace unambiguous."""

        keys = [item.source_key for item in self.items]
        if len(keys) != len(set(keys)):
            raise ValueError("EvidencePack 中的 source_key 必须唯一")
        return self


class Claim(BaseModel):
    """起草产出的单条带类型论断。"""

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(description="论断唯一标识，如 'sec1.2_clm3'")
    text: str = Field(description="论断正文，使用简体中文")
    claim_type: ClaimType = Field(description="fact / interpretation / common")
    source_keys: list[str] = Field(
        default_factory=list,
        description="fact 类型论断必须绑定至少一个 source_key；interpretation/common 可为空",
    )
    paragraph_index: int = Field(ge=0, description="所在段落序号，从 0 开始")

    @model_validator(mode="after")
    def validate_fact_source_keys(self) -> Claim:
        """A factual claim without evidence is invalid at the data boundary."""

        if self.claim_type == "fact" and not self.source_keys:
            raise ValueError("fact 类型论断必须绑定至少一个 source_key")
        if len(self.source_keys) != len(set(self.source_keys)):
            raise ValueError("同一论断不能重复绑定相同 source_key")
        return self


class SectionDraftOutput(BaseModel):
    """模型负责生成的章节内容，不包含由代码冻结的证据包。"""

    model_config = ConfigDict(frozen=True)

    section_id: str = Field(description="对应的提纲节点标识")
    heading: str = Field(description="本节标题")
    paragraphs: list[str] = Field(description="按序排列的段落正文")
    claims: list[Claim] = Field(description="本节所有论断，带类型与 source_key")

    @model_validator(mode="after")
    def validate_claim_locations(self) -> SectionDraftOutput:
        """Validate model-owned claim ids, paragraph indexes, and inline markers."""

        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("SectionDraftOutput 中的 claim_id 必须唯一")
        for claim in self.claims:
            if claim.paragraph_index >= len(self.paragraphs):
                raise ValueError(f"claim '{claim.claim_id}' 的 paragraph_index 越界")
            if claim.claim_type == "fact":
                paragraph = self.paragraphs[claim.paragraph_index]
                missing_markers = [key for key in claim.source_keys if f"[{key}]" not in paragraph]
                if missing_markers:
                    raise ValueError(
                        f"claim '{claim.claim_id}' 所在段落缺少引用标记: {missing_markers}"
                    )
        return self


class SectionDraft(SectionDraftOutput):
    """代码附回冻结 EvidencePack 后的完整单节结构化草稿。"""

    evidence_pack: EvidencePack = Field(description="本节使用的证据包")

    @model_validator(mode="after")
    def validate_claim_traceability(self) -> SectionDraft:
        """Ensure every factual claim resolves inside this immutable evidence pack."""

        if self.section_id != self.evidence_pack.section_id:
            raise ValueError("SectionDraft 与 EvidencePack 的 section_id 不一致")
        valid_keys = {item.source_key for item in self.evidence_pack.items}
        for claim in self.claims:
            unknown = set(claim.source_keys) - valid_keys
            if unknown:
                raise ValueError(
                    f"claim '{claim.claim_id}' 引用了本节证据包之外的 source_key: {sorted(unknown)}"
                )
        return self


# ---------------------------------------------------------------------------
# 4c — 核对
# ---------------------------------------------------------------------------


class VerificationDecision(BaseModel):
    """中性核对模型对一条事实论断返回的最小判定。"""

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(description="必须原样返回输入中的事实论断标识")
    verdict: VerificationVerdict = Field(description="supported / partial / unsupported")
    verifier_rationale: str = Field(
        min_length=1,
        max_length=500,
        description="简短判定理由，使用简体中文",
    )
    matched_chunk_text: str | None = Field(
        default=None,
        description="用于判定的关键原文短片段；不得返回完整证据包",
    )

    @field_validator("verifier_rationale")
    @classmethod
    def reject_repetitive_rationale(cls, value: str) -> str:
        """Reject looping verifier text before it can enter a checkpoint or cache."""

        compact = re.sub(r"\s+", "", value)
        if len(compact) < 160:
            return value
        span_length = 40
        seen: set[str] = set()
        for start in range(len(compact) - span_length + 1):
            span = compact[start : start + span_length]
            if span in seen:
                continue
            seen.add(span)
            if compact.count(span) >= 4:
                raise ValueError("核验理由出现机械重复")
        return value


class VerificationResponse(BaseModel):
    """核对模型的最小输出；原 Claim 与计数由代码附回。"""

    model_config = ConfigDict(frozen=True)

    section_id: str = Field(description="对应的章节标识")
    verified_claims: list[VerificationDecision] = Field(description="仅包含事实论断的逐条判定")


class VerifiedClaim(BaseModel):
    """核对后的单条论断验证结果。"""

    model_config = ConfigDict(frozen=True)

    claim: Claim = Field(description="原始论断")
    verdict: VerificationVerdict = Field(description="supported / partial / unsupported")
    verifier_rationale: str = Field(description="核对者（中性角色）的判定理由，使用简体中文")
    matched_chunk_text: str | None = Field(
        default=None,
        description="用于比对的 chunk 原文片段；unsupported 时可为空",
    )


class VerifiedDraft(BaseModel):
    """核对后的单节草稿。"""

    model_config = ConfigDict(frozen=True)

    section_id: str = Field(description="对应的提纲节点标识")
    verified_claims: list[VerifiedClaim] = Field(description="逐 claim 核对结果")
    unsupported_count: int = Field(default=0, description="unsupported 论断数")
    partial_count: int = Field(default=0, description="partial 论断数")
    supported_count: int = Field(default=0, description="supported 论断数")
    semantic_verification_performed: bool = Field(
        default=True,
        description="是否执行了中性 LLM 语义核验；False 表示仅通过结构安全门",
    )

    @model_validator(mode="after")
    def validate_verdict_counts(self) -> VerifiedDraft:
        """Prevent stale counters from bypassing the verification routing gate."""

        actual = {
            verdict: sum(1 for item in self.verified_claims if item.verdict == verdict)
            for verdict in ("supported", "partial", "unsupported")
        }
        expected = {
            "supported": self.supported_count,
            "partial": self.partial_count,
            "unsupported": self.unsupported_count,
        }
        if actual != expected:
            raise ValueError(f"核对计数与逐条 verdict 不一致: actual={actual}, expected={expected}")
        return self


# ---------------------------------------------------------------------------
# 参考文献
# ---------------------------------------------------------------------------


class ReferenceItem(BaseModel):
    """单条格式化参考文献。"""

    model_config = ConfigDict(frozen=True)

    source_key: str = Field(description="如 [S1]、[S2]，与 EvidenceItem 对应")
    citation_text: str = Field(description="按指定样式格式化后的完整引文")
    doc_id: str = Field(description="来源文档标识")
    chunk_id: str = Field(description="关联 chunk 标识")


class ReferenceList(BaseModel):
    """全篇参考文献列表。"""

    model_config = ConfigDict(frozen=True)

    items: list[ReferenceItem] = Field(description="按 source_key 排序的文献列表")
    style: CitationStyle = Field(default="gb-t-7714", description="引用样式")


# ---------------------------------------------------------------------------
# 4c — 打磨
# ---------------------------------------------------------------------------


class PolishedSection(BaseModel):
    """打磨后的单节成稿。"""

    model_config = ConfigDict(frozen=True)

    section_id: str = Field(description="对应的提纲节点标识")
    heading: str = Field(default="", description="本节标题")
    polished_text: str = Field(description="文风打磨后的正文，事实内容已冻结")
    fact_drift_detected: bool = Field(default=False, description="轻量核对是否检测到事实漂移")
    reverted_to_verified: bool = Field(
        default=False,
        description="候选打磨发生漂移或核对失败时，是否已回退到冻结事实版本",
    )
    safety_note: str = Field(default="", description="打磨安全门的处理说明")
    style_polish_performed: bool = Field(default=True, description="是否执行了 LLM 文风打磨")
    drift_check_performed: bool = Field(default=True, description="是否执行了中性 LLM 防漂移检查")


class PolishedDraft(BaseModel):
    """打磨后的全篇成稿。"""

    model_config = ConfigDict(frozen=True)

    title: str = Field(default="", description="文稿标题；单个段落可为空")
    sections: list[PolishedSection] = Field(description="逐节打磨后正文")
    reference_list: ReferenceList = Field(description="代码拼装的参考文献列表")
    citation_display: ResolvedCitationDisplay = Field(
        default="bibliography", description="成稿是否显示代码拼装的引用标记"
    )
    thesis: ThesisStatement = Field(description="锚定论点")
    fact_drift_free: bool = Field(default=True, description="全篇打磨后是否无事实漂移")
    quality_status: GenerationQualityStatus = Field(
        default="verified_final",
        description="verified_final 或明确标记的 unverified_draft",
    )
    quality_notes: list[str] = Field(default_factory=list, description="未执行质量步骤的透明说明")


# ---------------------------------------------------------------------------
# 生成上下文（贯穿全流水线）
# ---------------------------------------------------------------------------


class GenerationContext(BaseModel):
    """贯穿选题→框架→起草→核对→打磨的上下文容器。"""

    model_config = ConfigDict(frozen=True)

    kb_id: str = Field(description="检索所用的知识库标识")
    task_description: str = Field(description="写作任务的自然语言描述")
    citation_style: CitationStyle = Field(default="gb-t-7714", description="引用样式")
    persona_id: str | None = Field(default=None, description="当前使用的 persona 标识")
    task_id: str | None = Field(default=None, description="当前写作任务标识，用于持久化与恢复")
    allowed_doc_ids: tuple[str, ...] = Field(
        default=(), description="本任务允许作为事实来源的文档白名单"
    )
    excluded_persona_doc_ids: tuple[str, ...] = Field(
        default=(), description="默认排除的作者蒸馏来源文档"
    )
    source_policy_id: str | None = Field(default=None, description="事实来源隔离策略标识")
    generation_options: GenerationOptions = Field(
        default_factory=GenerationOptions,
        description="本任务冻结的篇幅与质量选项",
    )


# ---------------------------------------------------------------------------
# 阶段 6 — 一致性与全局打磨
# ---------------------------------------------------------------------------


class TermIssue(BaseModel):
    """术语不一致问题条目。"""

    model_config = ConfigDict(frozen=True)

    term: str = Field(description="存在不一致的术语/概念")
    occurrences: list[dict] = Field(
        description="各处使用情况的列表，每项含 section_id, text_snippet",
    )
    suggested_standard: str = Field(description="建议统一为哪个术语/表达")
    section_ids: list[str] = Field(description="涉及章节标识列表")


class TermConsistencyReport(BaseModel):
    """术语一致性审查报告。"""

    model_config = ConfigDict(frozen=True)

    issues: list[TermIssue] = Field(default_factory=list, description="发现的术语不一致问题列表")
    consistent_terms: list[str] = Field(
        default_factory=list,
        description="全文一致使用的关键术语列表",
    )
    reviewer_note: str = Field(
        default="",
        description="审查者对术语状况的总体评价与建议，使用简体中文",
    )


class StructureIssue(BaseModel):
    """结构性问题条目。"""

    model_config = ConfigDict(frozen=True)

    issue_type: str = Field(
        description="问题类型：section_balance（节篇幅失衡）/ logical_gap（逻辑跳跃）/ "
        "missing_transition（缺过渡段）/ overlong（本节过长）/ "
        "redundant（内容重叠）/ structural（整体结构问题）",
    )
    description: str = Field(description="问题描述，使用简体中文")
    section_ids: list[str] = Field(description="涉及章节标识列表")
    suggestion: str = Field(description="改进建议，使用简体中文")


class StructureReview(BaseModel):
    """全文结构审查报告。"""

    model_config = ConfigDict(frozen=True)

    issues: list[StructureIssue] = Field(default_factory=list, description="发现的结构问题列表")
    overall_assessment: str = Field(
        description="总体结构评价：论证是否连贯、推进是否有力、有无重大缺陷，使用简体中文",
    )


class GlobalPolishResult(BaseModel):
    """全局一致性打磨结果（利用 1M 上下文做的最后一次全篇审查）。"""

    model_config = ConfigDict(frozen=True)

    sections: list[PolishedSection] = Field(
        description="全局打磨后的逐节正文，含新增过渡段落和术语修正",
    )
    transitions_added: list[str] = Field(
        default_factory=list,
        description="新增/修改的过渡段落说明，使用简体中文",
    )
    global_consistency_notes: str = Field(
        default="",
        description="全局一致性说明：哪些问题已修复、哪些需关注，使用简体中文",
    )
