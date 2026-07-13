"""简体中文输出约束与确定性质量门。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from writing_factory.distill.academic import (
    CandidateClusterResult,
    ExclusivityBatchResult,
    PaperProfile,
    ValidationBatchResult,
)
from writing_factory.distill.models import (
    AcademicSupplementResult,
    MapResult,
    PersonaSpec,
    ReduceResult,
)

OutputLanguage = Literal["zh-CN"]
DEFAULT_OUTPUT_LANGUAGE: OutputLanguage = "zh-CN"


class OutputLanguageError(ValueError):
    """模型可读字段没有遵守约定输出语言。"""


def validate_map_language(result: MapResult, language: OutputLanguage) -> None:
    """检查 Map 的主要可读字段使用简体中文。"""

    if language != "zh-CN":
        return
    primary: list[str] = []
    supporting: list[str] = []
    for candidate in result.mental_candidates:
        primary.extend((candidate.name, candidate.description))
        supporting.extend((candidate.generative_rationale, candidate.exclusivity_rationale))
        for evidence in candidate.evidence:
            primary.extend((evidence.domain, evidence.summary))
    for candidate in result.heuristic_candidates:
        primary.extend((candidate.rule, candidate.trigger, candidate.example))
        for evidence in candidate.evidence:
            primary.extend((evidence.domain, evidence.summary))
    for tension in result.tensions:
        primary.extend((tension.side_a, tension.side_b))
        for evidence in tension.evidence:
            primary.extend((evidence.domain, evidence.summary))
    for gap in result.information_gaps:
        primary.extend((gap.dimension, gap.description, gap.reason))
    supporting.extend(result.value_signals)
    supporting.extend(result.anti_pattern_signals)
    supporting.extend(result.style_observations)
    _validate_chinese(primary, supporting, context="Map")


def validate_reduce_language(result: ReduceResult, language: OutputLanguage) -> None:
    """检查 Reduce 最终可见字段使用简体中文。"""

    if language != "zh-CN":
        return
    primary: list[str] = []
    supporting: list[str] = []
    for model in [*result.mental_models, *result.academic_conventions]:
        primary.extend(
            (
                model.name,
                model.description,
                model.applicability,
                model.limits,
                model.generative_rationale,
                model.exclusivity_rationale,
            )
        )
    for heuristic in result.decision_heuristics:
        primary.extend((heuristic.rule, heuristic.trigger, heuristic.example))
    for tension in result.core_tensions:
        primary.extend((tension.side_a, tension.side_b, tension.interpretation))
    for divergence in result.school_divergences:
        primary.append(divergence.question)
        for position in divergence.positions:
            primary.extend((position.label, position.position))
    for gap in result.information_gaps:
        primary.extend((gap.dimension, gap.description, gap.unresolved_reason))
    supporting.extend(result.style_rules)
    supporting.extend(result.values)
    supporting.extend(result.anti_patterns)
    supporting.extend(result.declared_limits)
    _validate_chinese(primary, supporting, context="Reduce")


def validate_academic_supplement_language(result: AcademicSupplementResult) -> None:
    """检查学术 v2 短 Reduce 的全部可读字段。"""

    primary: list[str] = []
    supporting: list[str] = []
    for heuristic in result.decision_heuristics:
        primary.extend((heuristic.rule, heuristic.trigger, heuristic.example))
    for tension in result.core_tensions:
        primary.extend((tension.side_a, tension.side_b, tension.interpretation))
    for gap in result.information_gaps:
        primary.extend((gap.dimension, gap.description, gap.unresolved_reason))
    supporting.extend(result.style_rules)
    supporting.extend(result.values)
    supporting.extend(result.anti_patterns)
    supporting.extend(result.declared_limits)
    _validate_chinese(primary, supporting, context="学术档案补充")


def validate_persona_language(spec: PersonaSpec) -> None:
    """发布前再次检查 PersonaSpec 的全部主要可读字段。"""

    if spec.output_language != "zh-CN":
        return
    primary: list[str] = []
    supporting: list[str] = []
    for model in [*spec.mental_models, *spec.academic_conventions]:
        primary.extend((model.name, model.description, model.applicability, model.limits))
        for evidence in model.cross_domain_evidence:
            primary.extend((evidence.domain, evidence.summary))
    for heuristic in spec.decision_heuristics:
        primary.extend((heuristic.rule, heuristic.trigger, heuristic.example))
    for tension in spec.core_tensions:
        primary.extend((tension.side_a, tension.side_b, tension.interpretation))
    for gap in spec.information_gaps:
        primary.extend((gap.dimension, gap.description, gap.unresolved_reason))
    supporting.extend(spec.expression_dna.style_rules)
    supporting.extend(spec.values)
    supporting.extend(spec.anti_patterns)
    supporting.extend(spec.declared_limits)
    _validate_chinese(primary, supporting, context="PersonaSpec")


def validate_academic_language(result: object) -> None:
    """检查论文归并、聚类和中性验证新增阶段的中文可读字段。"""

    primary: list[str] = []
    supporting: list[str] = []
    if isinstance(result, PaperProfile):
        for item in result.candidates:
            primary.extend((item.name, item.description, item.applicability, item.limits))
            supporting.append(item.research_context)
    elif isinstance(result, CandidateClusterResult):
        for item in result.candidates:
            primary.extend((item.name, item.description, item.applicability, item.limits))
            supporting.append(item.attribution_rationale)
    elif isinstance(result, (ValidationBatchResult, ExclusivityBatchResult)):
        primary.extend(item.rationale for item in result.assessments)
    else:
        return
    _validate_chinese(primary, supporting, context="学术蒸馏中间结果")


def _validate_chinese(
    primary: Iterable[str],
    supporting: Iterable[str],
    *,
    context: str,
) -> None:
    primary_values = [value.strip() for value in primary if value.strip()]
    supporting_values = [value.strip() for value in supporting if value.strip()]
    non_chinese = [value for value in primary_values if not _contains_cjk(value)]
    if non_chinese:
        raise OutputLanguageError(f"{context} 主要字段未使用简体中文: {non_chinese[0][:80]}")
    combined = "".join([*primary_values, *supporting_values])
    cjk = sum(_is_cjk(character) for character in combined)
    latin = sum(character.isascii() and character.isalpha() for character in combined)
    if combined and cjk < 20:
        raise OutputLanguageError(f"{context} 简体中文内容不足")
    if latin and cjk / (cjk + latin) < 0.35:
        raise OutputLanguageError(f"{context} 可读字段的中文比例过低")


def _contains_cjk(value: str) -> bool:
    return any(_is_cjk(character) for character in value)


def _is_cjk(character: str) -> bool:
    return "\u3400" <= character <= "\u9fff"
