"""Map 提取、全局归并与 PersonaSpec 的类型契约。"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from writing_factory.distill.academic import (
    AcademicModelValidation,
    AttributionScope,
    Specificity,
)
from writing_factory.distill.composition_models import CompositionDNA
from writing_factory.distill.options import LEGACY_DISTILLATION_OPTIONS, DistillationOptions

UnitScore = Annotated[float, Field(ge=-1.0, le=1.0)]
Confidence = Literal["high", "medium", "low", "inferred"]
PersonaMode = Literal["person", "topic"]


class ExtractedEvidence(BaseModel):
    """Map 阶段锚定到一个不可变原文切片的证据。"""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="输入中真实存在的切片标识，不得编造")
    domain: str = Field(description="该证据所属的具体研究领域，使用简体中文")
    summary: str = Field(description="证据如何支持候选项的简体中文摘要，不得重构引文")
    confidence: Confidence = Field(default="medium", description="证据判断的置信度")


class MapMentalCandidate(BaseModel):
    """尚未通过全局三重验证的局部心智模型候选。"""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="候选认知操作的简体中文名称")
    description: str = Field(description="候选认知操作如何运行的简体中文描述")
    evidence: list[ExtractedEvidence] = Field(min_length=1, description="支持该候选的证据")
    generative_rationale: str = Field(description="为何可能用于推断新问题立场，使用简体中文")
    exclusivity_rationale: str = Field(description="为何可能具有作者区分度，使用简体中文")


class MapHeuristicCandidate(BaseModel):
    """有原文证据支持的条件式决策启发。"""

    model_config = ConfigDict(frozen=True)

    rule: str = Field(description="简体中文的条件式决策规则")
    trigger: str = Field(description="启用该规则的条件，使用简体中文")
    example: str = Field(description="语料中可验证的应用示例，使用简体中文")
    evidence: list[ExtractedEvidence] = Field(min_length=1, description="支持该规则的证据")


class MapTensionCandidate(BaseModel):
    """不能被静默调和的两侧立场候选。"""

    model_config = ConfigDict(frozen=True)

    side_a: str = Field(description="张力一侧的简体中文表述")
    side_b: str = Field(description="张力另一侧的简体中文表述")
    tension_type: Literal["temporal", "domain", "essential", "school"] = Field(
        description="张力类型：时间、领域、本质或流派"
    )
    evidence: list[ExtractedEvidence] = Field(min_length=2, description="分别支持两侧立场的证据")


class MapInformationGap(BaseModel):
    """一个 Map 单元局部观察到的信息不足，不等同于全语料结论。"""

    model_config = ConfigDict(frozen=True)

    dimension: str = Field(description="信息不足所影响的分析维度，使用简体中文")
    description: str = Field(description="本单元具体缺少什么信息，使用简体中文")
    reason: str = Field(description="为何当前单元不足以支持判断，使用简体中文")
    resolvable_by_more_sources: bool = Field(description="更多来源是否可能补足该缺口")
    confidence: Confidence = Field(default="medium", description="该局部缺口判断的置信度")


class MapResult(BaseModel):
    """一个独立语料单元的结构化提取结果。"""

    model_config = ConfigDict(frozen=True)

    unit_id: str = Field(description="必须与输入完全一致的语料单元标识")
    mental_candidates: list[MapMentalCandidate] = Field(
        default_factory=list, description="尚待全局验证的心智模型候选"
    )
    heuristic_candidates: list[MapHeuristicCandidate] = Field(
        default_factory=list, description="有证据支持的决策启发候选"
    )
    tensions: list[MapTensionCandidate] = Field(
        default_factory=list, description="需要保留的张力候选"
    )
    value_signals: list[str] = Field(default_factory=list, description="简体中文的价值取向信号")
    anti_pattern_signals: list[str] = Field(
        default_factory=list, description="简体中文的反模式信号"
    )
    style_observations: list[str] = Field(
        default_factory=list, description="简体中文的表达风格观察"
    )
    information_gaps: list[MapInformationGap] = Field(
        default_factory=list, description="仅对当前单元成立的局部信息不足"
    )


class SourceSegment(BaseModel):
    """Immutable source data sent to one map call as data, never instructions."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    document_title: str
    filename: str
    text: str = Field(repr=False)
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None


class SourceUnit(BaseModel):
    """Bounded group of source segments processed independently by map."""

    model_config = ConfigDict(frozen=True)

    unit_id: str
    segments: list[SourceSegment] = Field(min_length=1)


class SentenceFingerprint(BaseModel):
    """Deterministic corpus statistics; the LLM never invents these values."""

    model_config = ConfigDict(frozen=True)

    character_count: int = Field(ge=0)
    sentence_count: int = Field(ge=0)
    paragraph_count: int = Field(ge=0)
    average_sentence_length: float = Field(ge=0)
    question_ratio: float = Field(ge=0, le=1)
    analogy_per_1000_chars: float = Field(ge=0)
    first_person_per_1000_chars: float = Field(ge=0)
    certainty_ratio: float = Field(ge=0, le=1)
    transition_per_1000_chars: float = Field(ge=0)


class StyleTags(BaseModel):
    """女娲七个风格轴；-1 偏向字段名左侧，+1 偏向右侧。"""

    model_config = ConfigDict(frozen=True)

    formal_to_colloquial: UnitScore = Field(default=0.0, description="正式到口语：-1 正式，+1 口语")
    abstract_to_concrete: UnitScore = Field(default=0.0, description="抽象到具体：-1 抽象，+1 具体")
    cautious_to_assertive: UnitScore = Field(
        default=0.0, description="谨慎到断言：-1 谨慎，+1 断言"
    )
    academic_to_popular: UnitScore = Field(default=0.0, description="学术到通俗：-1 学术，+1 通俗")
    long_to_short: UnitScore = Field(default=0.0, description="长句到短句：-1 长句，+1 短句")
    setup_to_conclusion_first: UnitScore = Field(
        default=0.0, description="铺垫到结论先行：-1 铺垫，+1 结论先行"
    )
    data_to_narrative: UnitScore = Field(default=0.0, description="数据到叙事：-1 数据，+1 叙事")


class PersonaEvidence(BaseModel):
    """Stable evidence copied from map results into the final PersonaSpec."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    chunk_id: str
    doc_id: str
    domain: str
    summary: str
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    confidence: Confidence


class TripleValidation(BaseModel):
    """Recorded decision for Nüwa's three mental-model tests."""

    model_config = ConfigDict(frozen=True)

    cross_domain: bool
    generative: bool
    exclusive: bool
    generative_rationale: str
    exclusivity_rationale: str

    @property
    def passed(self) -> bool:
        """A mental model must pass all three checks."""

        return self.cross_domain and self.generative and self.exclusive


class MentalModel(BaseModel):
    """One runnable lens with evidence, applicability, and failure conditions."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str | None = None
    name: str
    description: str
    cross_domain_evidence: list[PersonaEvidence] = Field(min_length=2)
    applicability: str
    limits: str
    validation: TripleValidation
    specificity: Specificity = "author_distinctive"
    attribution_scope: AttributionScope = "author_specific"
    academic_validation: AcademicModelValidation | None = None

    @model_validator(mode="after")
    def require_distinct_documents(self) -> MentalModel:
        documents = {item.doc_id for item in self.cross_domain_evidence}
        if len(documents) < 2:
            raise ValueError("Mental model evidence must cover at least two documents")
        if self.academic_validation is not None:
            if not self.academic_validation.eligible:
                raise ValueError("Academic mental model did not pass recurrence validation")
        return self


class DecisionHeuristic(BaseModel):
    """A reusable if-then rule downgraded from or adjacent to mental models."""

    model_config = ConfigDict(frozen=True)

    rule: str
    trigger: str
    example: str
    evidence: list[PersonaEvidence] = Field(min_length=1)


class LexicalMarker(BaseModel):
    """带置信度和证据的口癖或禁忌词。"""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="口癖或禁忌表达；原文外语词可以保留")
    confidence: Confidence = Field(description="该词汇标记的置信度")
    evidence_ids: list[str] = Field(default_factory=list, description="支持该标记的证据标识")


class ExpressionDNA(BaseModel):
    """Quantitative fingerprint plus reducer-classified style constraints."""

    model_config = ConfigDict(frozen=True)

    sentence_fingerprint: SentenceFingerprint
    style_tags: StyleTags
    taboo_words: list[LexicalMarker] = Field(default_factory=list)
    tics: list[LexicalMarker] = Field(default_factory=list)
    style_rules: list[str] = Field(default_factory=list)


class CoreTension(BaseModel):
    """A temporal, domain, essential, or school-level unresolved tension."""

    model_config = ConfigDict(frozen=True)

    side_a: str
    side_b: str
    tension_type: Literal["temporal", "domain", "essential", "school"]
    evidence: list[PersonaEvidence] = Field(min_length=2)
    interpretation: str


class SchoolPosition(BaseModel):
    """主题模式中一个流派的立场。"""

    model_config = ConfigDict(frozen=True)

    label: str = Field(description="流派或立场的简体中文标签")
    position: str = Field(description="该立场的简体中文说明")
    evidence_ids: list[str] = Field(min_length=1, description="支持该立场的证据标识")


class SchoolDivergence(BaseModel):
    """主题模式中必须保留、不能平均化的流派分歧。"""

    model_config = ConfigDict(frozen=True)

    question: str = Field(description="分歧围绕的简体中文问题")
    positions: list[SchoolPosition] = Field(min_length=2, description="至少两个不同立场")


class InformationGap(BaseModel):
    """经全语料重新判定后仍未解决的信息不足。"""

    model_config = ConfigDict(frozen=True)

    gap_id: str = Field(description="程序生成的稳定全局缺口标识")
    dimension: str = Field(description="受影响的分析维度，使用简体中文")
    description: str = Field(description="全语料仍缺少什么，使用简体中文")
    supporting_gap_ids: list[str] = Field(min_length=1, description="所依据的局部缺口标识")
    source_doc_ids: list[str] = Field(min_length=1, description="观察到该缺口的来源文档标识")
    reviewed_document_count: int = Field(ge=1, description="全局复核时检查的语料文档总数")
    unresolved_reason: str = Field(description="汇总全部来源后仍不能解决的原因，使用简体中文")
    confidence: Confidence = Field(description="全局缺口判断的置信度")


class SourceInfo(BaseModel):
    """Document-level provenance included in the serialized profile."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    filename: str
    source_type: Literal["primary", "secondary", "unknown"] = "primary"
    corpus_role: Literal["target", "control"] = "target"
    domain: str = ""
    chunk_count: int = Field(ge=1)


class PersonaSpec(BaseModel):
    """Authoritative serializable output consumed by later writing stages."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(default=1, ge=1)
    id: str
    name: str
    mode: PersonaMode
    output_language: Literal["zh-CN"] = Field(
        default="zh-CN", description="档案全部可读字段采用的输出语言"
    )
    distillation_options: DistillationOptions = Field(
        default_factory=lambda: LEGACY_DISTILLATION_OPTIONS,
        description="本版本实际启用的蒸馏质量步骤",
    )
    mental_models: list[MentalModel] = Field(min_length=3, max_length=7)
    academic_conventions: list[MentalModel] = Field(default_factory=list)
    decision_heuristics: list[DecisionHeuristic] = Field(default_factory=list)
    expression_dna: ExpressionDNA
    composition_dna: CompositionDNA = Field(
        default_factory=CompositionDNA,
        description="按非虚构文体组织的全文、章节、段落、句群与过渡规律",
    )
    core_tensions: list[CoreTension] = Field(default_factory=list)
    school_divergences: list[SchoolDivergence] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    anti_patterns: list[str] = Field(default_factory=list)
    evidence_registry: list[PersonaEvidence] = Field(min_length=1)
    source_info: list[SourceInfo] = Field(min_length=1)
    research_date: date
    declared_limits: list[str] = Field(min_length=3)
    information_gaps: list[InformationGap] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_mode_contract(self) -> PersonaSpec:
        if self.mode == "topic" and not self.school_divergences:
            raise ValueError("Topic mode must preserve at least one school divergence")
        identifiers = [model.name.strip().casefold() for model in self.mental_models]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Mental model names must be unique")
        if self.distillation_options.preset == "legacy":
            if any(
                model.academic_validation is None and not model.validation.passed
                for model in self.mental_models
            ):
                raise ValueError("历史完整档案的核心模型必须通过三重验证")
        elif self.distillation_options.cross_document_validation:
            if any(model.academic_validation is None for model in self.mental_models):
                raise ValueError("跨文档质量模式必须保存代码汇总的验证记录")
        return self


class ReduceMentalModel(BaseModel):
    """只引用登记证据标识的全局心智模型提案。"""

    model_config = ConfigDict(frozen=True)

    candidate_id: str | None = Field(default=None, description="新版蒸馏中必须复制已选候选标识")
    name: str = Field(description="心智模型的简体中文名称")
    description: str = Field(description="该认知操作如何运行的简体中文描述")
    evidence_ids: list[str] = Field(min_length=2, description="至少跨两个领域的登记证据标识")
    applicability: str = Field(description="适用问题和使用条件，使用简体中文")
    limits: str = Field(description="失效条件和适用边界，使用简体中文")
    generative: bool = Field(description="是否能生成对新问题的可检验推断")
    exclusive: bool = Field(description="是否具有作者区分度而非通用非虚构写作惯例")
    generative_rationale: str = Field(description="生成力判断理由，使用简体中文")
    exclusivity_rationale: str = Field(description="排他性判断理由，使用简体中文")


class ReduceHeuristic(BaseModel):
    """全局归并后的决策启发提案。"""

    model_config = ConfigDict(frozen=True)

    rule: str = Field(description="简体中文的条件式决策规则")
    trigger: str = Field(description="启用条件，使用简体中文")
    example: str = Field(description="有证据支持的应用示例，使用简体中文")
    evidence_ids: list[str] = Field(min_length=1, description="支持该规则的登记证据标识")


class ReduceTension(BaseModel):
    """全语料仍不能调和的核心张力提案。"""

    model_config = ConfigDict(frozen=True)

    side_a: str = Field(description="张力一侧的简体中文表述")
    side_b: str = Field(description="张力另一侧的简体中文表述")
    tension_type: Literal["temporal", "domain", "essential", "school"] = Field(
        description="张力类型：时间、领域、本质或流派"
    )
    evidence_ids: list[str] = Field(min_length=2, description="分别支持两侧的登记证据标识")
    interpretation: str = Field(description="为何应保留而不能强行调和，使用简体中文")


class ReduceInformationGap(BaseModel):
    """Reduce 对局部缺口进行全局复核后仍需保留的提案。"""

    model_config = ConfigDict(frozen=True)

    dimension: str = Field(description="受影响的分析维度，使用简体中文")
    description: str = Field(description="全语料仍缺少什么，使用简体中文")
    supporting_gap_ids: list[str] = Field(min_length=1, description="引用的局部缺口标识")
    reviewed_document_count: int = Field(ge=1, description="必须等于输入语料清单中的文档总数")
    unresolved_reason: str = Field(description="为何更多现有来源仍未补足，使用简体中文")
    confidence: Confidence = Field(description="全局缺口判断的置信度")


class AcademicSupplementResult(BaseModel):
    """新版在代码选模后仅需模型补充的档案组成部分。"""

    model_config = ConfigDict(frozen=True)

    decision_heuristics: list[ReduceHeuristic] = Field(
        default_factory=list, max_length=8, description="归并后的非虚构写作启发式"
    )
    style_tags: StyleTags = Field(description="七个量化表达风格轴")
    taboo_words: list[LexicalMarker] = Field(
        default_factory=list, max_length=8, description="有证据或明确推断标记的禁忌词"
    )
    tics: list[LexicalMarker] = Field(
        default_factory=list, max_length=8, description="有证据或明确推断标记的口癖"
    )
    style_rules: list[str] = Field(
        default_factory=list, max_length=10, description="简体中文的可执行表达规则"
    )
    core_tensions: list[ReduceTension] = Field(
        default_factory=list, max_length=5, description="全语料仍不能调和的核心张力"
    )
    school_divergences: list[SchoolDivergence] = Field(
        default_factory=list, max_length=5, description="主题模式中的流派分歧"
    )
    values: list[str] = Field(default_factory=list, max_length=10, description="简体中文的价值取向")
    anti_patterns: list[str] = Field(
        default_factory=list, max_length=10, description="简体中文的反模式"
    )
    declared_limits: list[str] = Field(
        min_length=3, max_length=6, description="三至六条简体中文诚实边界"
    )
    information_gaps: list[ReduceInformationGap] = Field(
        default_factory=list,
        max_length=8,
        description="经全语料复核后仍未解决的信息不足",
    )


class ReduceResult(BaseModel):
    """要求全局归并模型返回的受验证 JSON 结构。"""

    model_config = ConfigDict(frozen=True)

    mental_models: list[ReduceMentalModel] = Field(
        min_length=3, max_length=7, description="经代码或三重验证选定的 3 至 7 个心智模型"
    )
    academic_conventions: list[ReduceMentalModel] = Field(
        default_factory=list,
        max_length=7,
        description="有证据但不进入核心列表的领域或通用非虚构写作惯例",
    )
    decision_heuristics: list[ReduceHeuristic] = Field(
        default_factory=list, description="降级或归并后的决策启发"
    )
    style_tags: StyleTags = Field(description="七个量化表达风格轴")
    taboo_words: list[LexicalMarker] = Field(
        default_factory=list, description="有证据或明确推断标记的禁忌词"
    )
    tics: list[LexicalMarker] = Field(
        default_factory=list, description="有证据或明确推断标记的口癖"
    )
    style_rules: list[str] = Field(default_factory=list, description="简体中文的可执行表达规则")
    core_tensions: list[ReduceTension] = Field(
        default_factory=list, description="必须保留的核心张力"
    )
    school_divergences: list[SchoolDivergence] = Field(
        default_factory=list, description="主题模式中的流派分歧"
    )
    values: list[str] = Field(default_factory=list, description="简体中文的价值取向")
    anti_patterns: list[str] = Field(default_factory=list, description="简体中文的反模式")
    declared_limits: list[str] = Field(min_length=3, description="至少三条简体中文诚实边界")
    information_gaps: list[ReduceInformationGap] = Field(
        default_factory=list, description="经全语料复核后仍未解决的信息不足"
    )


class DistillationOutcome(BaseModel):
    """Result returned to the UI with explicit reuse and run identifiers."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    persona: PersonaSpec
    markdown: str
    reused: bool = False
