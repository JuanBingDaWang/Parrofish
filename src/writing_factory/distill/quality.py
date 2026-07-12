"""Code-based Nüwa quality checks independent of the synthesis response."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from writing_factory.distill.models import PersonaSpec


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
    for model in spec.mental_models:
        referenced_ids.update(item.evidence_id for item in model.cross_domain_evidence)
    for heuristic in spec.decision_heuristics:
        referenced_ids.update(item.evidence_id for item in heuristic.evidence)
    for tension in spec.core_tensions:
        referenced_ids.update(item.evidence_id for item in tension.evidence)
    for divergence in spec.school_divergences:
        for position in divergence.positions:
            referenced_ids.update(position.evidence_ids)
    checks = {
        "mental_model_count": 3 <= len(spec.mental_models) <= 7,
        "triple_validation": all(
            model.validation.passed
            and len({item.domain.casefold() for item in model.cross_domain_evidence}) >= 2
            for model in spec.mental_models
        ),
        "model_limits": all(bool(model.limits.strip()) for model in spec.mental_models),
        "evidence_traceability": referenced_ids.issubset(registry_ids),
        "expression_fingerprint": spec.expression_dna.sentence_fingerprint.character_count > 0,
        "honest_boundaries": len(spec.declared_limits) >= 3,
        "source_transparency": bool(spec.source_info),
        "topic_divergence": spec.mode != "topic" or bool(spec.school_divergences),
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
    warnings.extend(f"信息不足：{gap}" for gap in spec.information_gaps)
    return StaticQualityReport(checks=checks, warnings=warnings)
