"""Evidence-backed composition-DNA contracts for nonfiction personas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from writing_factory.nonfiction import NonfictionGenre

CompositionScope = Literal["document", "section", "paragraph", "sentence", "transition"]
CompositionSpecificity = Literal[
    "author_distinctive",
    "genre_conventional",
    "cross_genre_author",
    "unverified",
    "provisional",
]
CompositionConfidence = Literal["high", "medium", "low"]


class CompositionEvidence(BaseModel):
    """Audit-only structural evidence anchored to an immutable source chunk."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(description="结构证据的稳定审计标识")
    chunk_id: str = Field(description="证据所在的知识库切片标识")
    doc_id: str = Field(description="证据所在的源文档标识")
    summary: str = Field(description="证据支持哪一项结构观察的简体中文摘要")
    page_start: int | None = Field(default=None, description="证据起始页码")
    page_end: int | None = Field(default=None, description="证据结束页码")
    section_heading: str | None = Field(default=None, description="证据所在的原文章节标题")


class CompositionPattern(BaseModel):
    """One reusable and explicitly conditional composition rule."""

    model_config = ConfigDict(frozen=True)

    pattern_id: str = Field(description="谋篇模式的稳定标识")
    name: str = Field(description="谋篇模式的简体中文名称")
    scope: CompositionScope = Field(description="模式所在的结构尺度")
    description: str = Field(description="模式如何组织内容及其作用")
    sequence: list[str] = Field(default_factory=list, description="按顺序排列的修辞功能")
    relations: list[str] = Field(default_factory=list, description="内容单元之间的关系")
    applicability: str = Field(description="模式适用的文体任务、目的或受众条件")
    variability: str = Field(description="模式在不同文本中允许怎样变化")
    supporting_doc_ids: list[str] = Field(min_length=1, description="复现该模式的目标文档标识")
    recurrence_document_count: int = Field(ge=1, description="去重后的复现文档数量")
    specificity: CompositionSpecificity = Field(description="模式的作者区分度或暂定状态")
    confidence: CompositionConfidence = Field(description="由复现文档数量确定的置信度")
    evidence: list[CompositionEvidence] = Field(min_length=1, description="仅供审计的结构证据锚点")

    @model_validator(mode="after")
    def validate_recurrence(self) -> CompositionPattern:
        if self.recurrence_document_count != len(set(self.supporting_doc_ids)):
            raise ValueError("谋篇模式的复现数必须等于去重后的支持文档数")
        return self


class GenreCompositionProfile(BaseModel):
    """Composition repertoire observed within one nonfiction genre."""

    model_config = ConfigDict(frozen=True)

    genre: NonfictionGenre = Field(description="该侧写对应的非虚构文体枚举")
    genre_label: str = Field(description="该文体的简体中文显示名")
    source_document_count: int = Field(ge=1, description="该文体的目标语料文档数")
    typical_purposes: list[str] = Field(
        default_factory=list, description="该作者在此文体中的常见沟通目的"
    )
    audience_tendencies: list[str] = Field(
        default_factory=list, description="该作者在此文体中的常见受众取向"
    )
    heading_strategy: str = Field(default="", description="标题与层级的使用策略")
    paragraph_strategy: str = Field(default="", description="段落和句群的组织策略")
    patterns: list[CompositionPattern] = Field(
        default_factory=list, description="该文体下复现的条件式谋篇模式"
    )
    declared_limits: list[str] = Field(
        default_factory=list, description="该文体侧写不能可靠外推的边界"
    )


class CompositionDNA(BaseModel):
    """Hierarchical nonfiction organization rules separated from expression DNA."""

    model_config = ConfigDict(frozen=True)

    genre_profiles: list[GenreCompositionProfile] = Field(
        default_factory=list, description="按非虚构文体分开的谋篇侧写"
    )
    cross_genre_patterns: list[CompositionPattern] = Field(
        default_factory=list, description="跨至少两种文体复现的作者谋篇模式"
    )
    information_gaps: list[str] = Field(
        default_factory=list, description="完整语料仍无法判断的结构维度"
    )


class DocumentPatternCandidate(BaseModel):
    """One structural observation within a complete source document."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="单篇文档中观察到的谋篇模式名称")
    scope: CompositionScope = Field(description="观察所在的结构尺度")
    description: str = Field(description="该结构观察的简体中文说明")
    sequence: list[str] = Field(default_factory=list, description="按顺序排列的抽象修辞功能")
    relations: list[str] = Field(default_factory=list, description="内容单元之间的抽象关系")
    applicability: str = Field(description="该结构观察在本文中的适用场景")
    variability: str = Field(description="该结构观察可能发生的变化")
    evidence_chunk_ids: list[str] = Field(min_length=1, description="支持观察的真实切片标识")


class DocumentCompositionProfile(BaseModel):
    """Ordered, whole-document structural Map result."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="被分析的源文档标识")
    genre: NonfictionGenre = Field(description="从完整文档判断的非虚构文体")
    genre_label: str = Field(description="非虚构文体的简体中文显示名")
    purpose: str = Field(description="文档的主要沟通目的")
    audience: str = Field(description="文档面向的主要受众")
    heading_strategy: str = Field(description="文档怎样使用标题与层级")
    paragraph_strategy: str = Field(description="文档怎样组织段落与句群")
    patterns: list[DocumentPatternCandidate] = Field(
        default_factory=list, description="完整文档中的结构观察"
    )
    information_gaps: list[str] = Field(
        default_factory=list, description="该文档无法支持判断的结构维度"
    )


class ReducedCompositionPattern(BaseModel):
    """Cross-document pattern proposal referencing only source IDs."""

    model_config = ConfigDict(frozen=True)

    pattern_id: str = Field(description="归并后谋篇模式的稳定标识")
    name: str = Field(description="归并后谋篇模式的简体中文名称")
    scope: CompositionScope = Field(description="模式所在的结构尺度")
    description: str = Field(description="模式如何组织内容及其作用")
    sequence: list[str] = Field(default_factory=list, description="跨文档复现的修辞功能序列")
    relations: list[str] = Field(default_factory=list, description="跨文档复现的单元关系")
    applicability: str = Field(description="模式可用于新任务的条件")
    variability: str = Field(description="模式允许的结构变体")
    evidence_chunk_ids: list[str] = Field(min_length=1, description="支持模式的目标语料切片标识")
    supporting_doc_ids: list[str] = Field(min_length=1, description="支持模式的目标语料文档标识")
    recurrence_document_count: int = Field(ge=1, description="去重后的复现文档数量")
    specificity: CompositionSpecificity = Field(description="相对对照语料判断的模式区分度")
    confidence: CompositionConfidence = Field(description="模型建议的置信级别，最终由代码校准")

    @model_validator(mode="after")
    def validate_recurrence(self) -> ReducedCompositionPattern:
        if self.recurrence_document_count != len(set(self.supporting_doc_ids)):
            raise ValueError("谋篇候选复现数与支持文档数不一致")
        return self


class ReducedGenreCompositionProfile(BaseModel):
    """Reduced genre profile before code attaches source metadata."""

    model_config = ConfigDict(frozen=True)

    genre: NonfictionGenre = Field(description="该侧写对应的非虚构文体枚举")
    genre_label: str = Field(description="该文体的简体中文显示名")
    source_document_count: int = Field(ge=1, description="该文体目标语料的文档数量")
    typical_purposes: list[str] = Field(
        default_factory=list, description="目标语料在此文体中的常见目的"
    )
    audience_tendencies: list[str] = Field(
        default_factory=list, description="目标语料在此文体中的受众取向"
    )
    heading_strategy: str = Field(description="归并后的标题与层级策略")
    paragraph_strategy: str = Field(description="归并后的段落与句群策略")
    patterns: list[ReducedCompositionPattern] = Field(
        default_factory=list, description="该文体内归并出的谋篇模式"
    )
    declared_limits: list[str] = Field(default_factory=list, description="侧写的可靠性与外推边界")


class CompositionReduceResult(BaseModel):
    """Global composition Reduce output validated before publication."""

    model_config = ConfigDict(frozen=True)

    genre_profiles: list[ReducedGenreCompositionProfile] = Field(
        min_length=1, description="覆盖全部目标文体且互不重复的侧写"
    )
    cross_genre_patterns: list[ReducedCompositionPattern] = Field(
        default_factory=list, description="跨至少两种目标文体复现的作者模式"
    )
    information_gaps: list[str] = Field(
        default_factory=list, description="全局归并后仍无法判断的结构维度"
    )
