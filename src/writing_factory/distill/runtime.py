"""从完整审计档案构造不携带旧语料事实的运行时作者模型。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from writing_factory.distill.academic import AttributionScope, Specificity
from writing_factory.distill.models import (
    Confidence,
    PersonaMode,
    PersonaSpec,
    SentenceFingerprint,
    StyleTags,
)


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


class RuntimePersonaSpec(BaseModel):
    """生成阶段允许使用的 PersonaSpec 安全投影。"""

    model_config = ConfigDict(frozen=True)

    projection_version: int = 1
    persona_id: str
    name: str
    mode: PersonaMode
    mental_models: list[RuntimeMentalModel] = Field(min_length=3, max_length=7)
    decision_rules: list[str] = Field(default_factory=list)
    expression_dna: RuntimeExpressionDNA
    values: list[str] = Field(default_factory=list)
    anti_patterns: list[str] = Field(default_factory=list)
    declared_limits: list[str] = Field(default_factory=list)


def build_runtime_persona(persona: PersonaSpec) -> RuntimePersonaSpec:
    """确定性删除证据锚点、来源、旧案例和论文事实。"""

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
        values=persona.values,
        anti_patterns=persona.anti_patterns,
        declared_limits=persona.declared_limits,
    )
