"""Code-based Nüwa quality checks independent of the synthesis response."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from writing_factory.distill.language import (
    OutputLanguageError,
    validate_persona_language,
)
from writing_factory.distill.models import PersonaSpec
from writing_factory.distill.runtime import build_runtime_persona


class StaticQualityReport(BaseModel):
    """Hard invariant results plus non-fabricating quality warnings."""

    model_config = ConfigDict(frozen=True)

    checks: dict[str, bool]
    warnings: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """All hard checks must pass before a profile is published."""

        return all(self.checks.values())


def run_static_quality_check(spec: PersonaSpec) -> StaticQualityReport:
    """Check structural fidelity without asking the synthesizing model to self-grade."""

    registry_ids = {item.evidence_id for item in spec.evidence_registry}
    referenced_ids: set[str] = set()
    for model in [*spec.mental_models, *spec.academic_conventions]:
        referenced_ids.update(item.evidence_id for item in model.cross_domain_evidence)
    for heuristic in spec.decision_heuristics:
        referenced_ids.update(item.evidence_id for item in heuristic.evidence)
    for tension in spec.core_tensions:
        referenced_ids.update(item.evidence_id for item in tension.evidence)
    for divergence in spec.school_divergences:
        for position in divergence.positions:
            referenced_ids.update(position.evidence_ids)
    try:
        validate_persona_language(spec)
        output_language = True
    except OutputLanguageError:
        output_language = False
    options = spec.distillation_options
    if options.preset == "legacy":
        model_validation = all(
            (
                model.academic_validation.eligible
                if model.academic_validation is not None
                else model.validation.passed
            )
            and len({item.doc_id for item in model.cross_domain_evidence}) >= 2
            for model in spec.mental_models
        )
    elif options.cross_document_validation:
        model_validation = all(
            model.academic_validation is not None
            and model.academic_validation.eligible
            and len({item.doc_id for item in model.cross_domain_evidence}) >= 2
            for model in spec.mental_models
        )
    else:
        model_validation = all(
            len({item.doc_id for item in model.cross_domain_evidence}) >= 2
            for model in spec.mental_models
        )
    runtime = build_runtime_persona(spec).model_dump(mode="json")
    runtime_text = str(runtime)
    source_doc_ids = {item.doc_id for item in spec.source_info}
    composition_patterns = [
        pattern
        for profile in spec.composition_dna.genre_profiles
        for pattern in profile.patterns
    ] + list(spec.composition_dna.cross_genre_patterns)
    checks = {
        "mental_model_count": 3 <= len(spec.mental_models) <= 7,
        "triple_validation": model_validation,
        "model_limits": all(bool(model.limits.strip()) for model in spec.mental_models),
        "evidence_traceability": referenced_ids.issubset(registry_ids),
        "runtime_evidence_isolation": not any(
            key in runtime_text
            for key in ("evidence_id", "chunk_id", "source_info", "research_context")
        ),
        "expression_fingerprint": spec.expression_dna.sentence_fingerprint.character_count > 0,
        "composition_evidence_traceability": all(
            evidence.doc_id in source_doc_ids
            for pattern in composition_patterns
            for evidence in pattern.evidence
        ),
        "honest_boundaries": len(spec.declared_limits) >= 3,
        "source_transparency": bool(spec.source_info),
        "topic_divergence": spec.mode != "topic" or bool(spec.school_divergences),
        "output_language": output_language,
    }
    warnings: list[str] = []
    if len(spec.core_tensions) < 2:
        warnings.append("可验证的核心张力少于 2 对；保持信息不足，不自动补造")
    if len(spec.decision_heuristics) < 5:
        warnings.append("可验证的决策启发式少于 Nüwa 建议的 5 条")
    if spec.mode == "person" and len(spec.source_info) < 2:
        warnings.append("人物模式来源文档少于 2 份，跨文档稳定性有限")
    if not spec.expression_dna.style_rules:
        warnings.append("表达风格规则为空，后续 Voice Check 风险较高")
    if not spec.composition_dna.genre_profiles:
        warnings.append("尚无可复用的谋篇 DNA；生成框架将仅依据任务和心智模型")
    if not options.cross_document_validation:
        warnings.append("本版本未运行跨文档复现与聚类，核心模型仅作为基础候选使用")
    if not options.generative_validation:
        warnings.append("本版本未运行留出语料生成力验证")
    if not options.exclusivity_validation:
        warnings.append("本版本未运行对照语料排他性验证，不宣称作者独特性")
    warnings.extend(
        f"信息不足（{gap.dimension}）：{gap.description}；{gap.unresolved_reason}"
        for gap in spec.information_gaps
    )
    return StaticQualityReport(checks=checks, warnings=warnings)
