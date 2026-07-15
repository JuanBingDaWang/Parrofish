"""从完整审计档案构造不携带旧语料事实的运行时作者模型。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from writing_factory.distill.academic import AttributionScope, Specificity
from writing_factory.distill.composition_models import (
    CompositionConfidence,
    CompositionDNA,
    CompositionPattern,
    CompositionScope,
    CompositionSpecificity,
)
from writing_factory.distill.models import (
    Confidence,
    PersonaMode,
    PersonaSpec,
    SentenceFingerprint,
    StyleTags,
)
from writing_factory.nonfiction import NonfictionGenre


class RuntimeLexicalMarker(BaseModel):
    """不携带证据 ID 的运行时词汇规则。"""

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: Confidence


class RuntimeExpressionDNA(BaseModel):
    """表达约束的安全投影，移除全部证据关联。"""

    model_config = ConfigDict(frozen=True)

    sentence_fingerprint: SentenceFingerprint
    style_tags: StyleTags
    taboo_words: list[RuntimeLexicalMarker] = Field(default_factory=list)
    tics: list[RuntimeLexicalMarker] = Field(default_factory=list)
    style_rules: list[str] = Field(default_factory=list)


class RuntimeMentalModel(BaseModel):
    """可传给写作模型的认知操作，不包含证据、旧案例或来源信息。"""

    model_config = ConfigDict(frozen=True)

    candidate_id: str | None = None
    name: str
    description: str
    applicability: str
    limits: str
    specificity: Specificity
    attribution_scope: AttributionScope


class RuntimeCompositionPattern(BaseModel):
    """Reusable structural rule with all source anchors removed."""

    model_config = ConfigDict(frozen=True)

    name: str
    scope: CompositionScope
    description: str
    sequence: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    applicability: str
    variability: str
    specificity: CompositionSpecificity
    confidence: CompositionConfidence


class RuntimeGenreCompositionProfile(BaseModel):
    """Safe genre overlay selected by a new writing task."""

    model_config = ConfigDict(frozen=True)

    genre: NonfictionGenre
    genre_label: str
    typical_purposes: list[str] = Field(default_factory=list)
    audience_tendencies: list[str] = Field(default_factory=list)
    heading_strategy: str = ""
    paragraph_strategy: str = ""
    patterns: list[RuntimeCompositionPattern] = Field(default_factory=list)
    declared_limits: list[str] = Field(default_factory=list)


class RuntimeCompositionDNA(BaseModel):
    """Runtime-only composition DNA without documents, chunks, or old examples."""

    model_config = ConfigDict(frozen=True)

    genre_profiles: list[RuntimeGenreCompositionProfile] = Field(default_factory=list)
    cross_genre_patterns: list[RuntimeCompositionPattern] = Field(default_factory=list)
    information_gaps: list[str] = Field(default_factory=list)


class RuntimePersonaSpec(BaseModel):
    """生成阶段允许使用的 PersonaSpec 安全投影。"""

    model_config = ConfigDict(frozen=True)

    projection_version: int = 2
    persona_id: str
    name: str
    mode: PersonaMode
    mental_models: list[RuntimeMentalModel] = Field(min_length=3, max_length=7)
    decision_rules: list[str] = Field(default_factory=list)
    expression_dna: RuntimeExpressionDNA
    composition_dna: RuntimeCompositionDNA = Field(default_factory=RuntimeCompositionDNA)
    values: list[str] = Field(default_factory=list)
    anti_patterns: list[str] = Field(default_factory=list)
    declared_limits: list[str] = Field(default_factory=list)


def build_runtime_persona(persona: PersonaSpec) -> RuntimePersonaSpec:
    """确定性删除证据锚点、来源、旧案例和来源文档事实。"""

    return RuntimePersonaSpec(
        persona_id=persona.id,
        name=persona.name,
        mode=persona.mode,
        mental_models=[
            RuntimeMentalModel(
                candidate_id=model.candidate_id,
                name=model.name,
                description=model.description,
                applicability=model.applicability,
                limits=model.limits,
                specificity=model.specificity,
                attribution_scope=model.attribution_scope,
            )
            for model in persona.mental_models
        ],
        decision_rules=[f"当{item.trigger}时，{item.rule}" for item in persona.decision_heuristics],
        expression_dna=RuntimeExpressionDNA(
            sentence_fingerprint=persona.expression_dna.sentence_fingerprint,
            style_tags=persona.expression_dna.style_tags,
            taboo_words=[
                RuntimeLexicalMarker(text=item.text, confidence=item.confidence)
                for item in persona.expression_dna.taboo_words
            ],
            tics=[
                RuntimeLexicalMarker(text=item.text, confidence=item.confidence)
                for item in persona.expression_dna.tics
            ],
            style_rules=persona.expression_dna.style_rules,
        ),
        composition_dna=_runtime_composition(persona.composition_dna),
        values=persona.values,
        anti_patterns=persona.anti_patterns,
        declared_limits=persona.declared_limits,
    )


def _runtime_composition(composition: CompositionDNA) -> RuntimeCompositionDNA:
    def safe_pattern(item: CompositionPattern) -> RuntimeCompositionPattern:
        return RuntimeCompositionPattern(
            name=item.name,
            scope=item.scope,
            description=item.description,
            sequence=item.sequence,
            relations=item.relations,
            applicability=item.applicability,
            variability=item.variability,
            specificity=item.specificity,
            confidence=item.confidence,
        )

    return RuntimeCompositionDNA(
        genre_profiles=[
            RuntimeGenreCompositionProfile(
                genre=item.genre,
                genre_label=item.genre_label,
                typical_purposes=item.typical_purposes,
                audience_tendencies=item.audience_tendencies,
                heading_strategy=item.heading_strategy,
                paragraph_strategy=item.paragraph_strategy,
                patterns=[safe_pattern(pattern) for pattern in item.patterns],
                declared_limits=item.declared_limits,
            )
            for item in composition.genre_profiles
        ],
        cross_genre_patterns=[safe_pattern(item) for item in composition.cross_genre_patterns],
        information_gaps=composition.information_gaps,
    )
