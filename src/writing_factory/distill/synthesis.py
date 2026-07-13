"""Evidence registry, Nüwa reduce call, and code-enforced PersonaSpec assembly."""

from __future__ import annotations

import hashlib
import json
from datetime import date

from pydantic import ValidationError

from writing_factory.distill.expression import ExpressionStatistics
from writing_factory.distill.extraction import StructuredDistillationError
from writing_factory.distill.language import (
    DEFAULT_OUTPUT_LANGUAGE,
    OutputLanguage,
    OutputLanguageError,
    validate_reduce_language,
)
from writing_factory.distill.models import (
    CoreTension,
    DecisionHeuristic,
    ExpressionDNA,
    InformationGap,
    MapResult,
    MentalModel,
    PersonaEvidence,
    PersonaMode,
    PersonaSpec,
    ReduceInformationGap,
    ReduceResult,
    SourceInfo,
    SourceUnit,
    StyleTags,
    TripleValidation,
)
from writing_factory.distill.prompts import reduce_messages
from writing_factory.llm import SiliconFlowClient


class CandidateBundleBuilder:
    """Assign stable evidence IDs and strip raw source text before reduction."""

    def build(
        self, map_results: list[MapResult], units: tuple[SourceUnit, ...]
    ) -> tuple[
        dict[str, PersonaEvidence],
        dict[str, dict[str, object]],
        dict[str, object],
    ]:
        """Return a final evidence registry and compact reducer candidate bundle."""

        segment_by_chunk = {
            segment.chunk_id: segment for unit in units for segment in unit.segments
        }
        unit_by_id = {unit.unit_id: unit for unit in units}
        registry: dict[str, PersonaEvidence] = {}
        gap_registry: dict[str, dict[str, object]] = {}

        def register(item) -> str:
            segment = segment_by_chunk[item.chunk_id]
            digest = hashlib.sha256(
                f"{item.chunk_id}|{item.domain}|{item.summary}".encode()
            ).hexdigest()[:24]
            identifier = f"ev_{digest}"
            registry.setdefault(
                identifier,
                PersonaEvidence(
                    evidence_id=identifier,
                    chunk_id=item.chunk_id,
                    doc_id=segment.doc_id,
                    domain=item.domain,
                    summary=item.summary,
                    page_start=segment.page_start,
                    page_end=segment.page_end,
                    section_heading=segment.section_heading,
                    confidence=item.confidence,
                ),
            )
            return identifier

        mental_candidates: list[dict[str, object]] = []
        heuristic_candidates: list[dict[str, object]] = []
        tensions: list[dict[str, object]] = []
        values: list[str] = []
        anti_patterns: list[str] = []
        style_observations: list[str] = []
        for result in map_results:
            for candidate in result.mental_candidates:
                mental_candidates.append(
                    {
                        "name": candidate.name,
                        "description": candidate.description,
                        "evidence_ids": [register(item) for item in candidate.evidence],
                        "generative_rationale": candidate.generative_rationale,
                        "exclusivity_rationale": candidate.exclusivity_rationale,
                    }
                )
            for candidate in result.heuristic_candidates:
                heuristic_candidates.append(
                    {
                        "rule": candidate.rule,
                        "trigger": candidate.trigger,
                        "example": candidate.example,
                        "evidence_ids": [register(item) for item in candidate.evidence],
                    }
                )
            for tension in result.tensions:
                tensions.append(
                    {
                        "side_a": tension.side_a,
                        "side_b": tension.side_b,
                        "tension_type": tension.tension_type,
                        "evidence_ids": [register(item) for item in tension.evidence],
                    }
                )
            values.extend(result.value_signals)
            anti_patterns.extend(result.anti_pattern_signals)
            style_observations.extend(result.style_observations)
            source_doc_ids = sorted(
                {segment.doc_id for segment in unit_by_id[result.unit_id].segments}
            )
            for gap in result.information_gaps:
                digest = hashlib.sha256(
                    (f"{result.unit_id}|{gap.dimension}|{gap.description}|{gap.reason}").encode()
                ).hexdigest()[:24]
                gap_id = f"gap_{digest}"
                record: dict[str, object] = {
                    "gap_id": gap_id,
                    "unit_id": result.unit_id,
                    "source_doc_ids": source_doc_ids,
                    **gap.model_dump(mode="json"),
                }
                gap_registry.setdefault(gap_id, record)
        bundle: dict[str, object] = {
            "evidence_registry": [item.model_dump(mode="json") for item in registry.values()],
            "mental_candidates": mental_candidates,
            "heuristic_candidates": heuristic_candidates,
            "tensions": tensions,
            "value_signals": self._unique(values),
            "anti_pattern_signals": self._unique(anti_patterns),
            "style_observations": self._unique(style_observations),
            "local_information_gaps": list(gap_registry.values()),
        }
        return registry, gap_registry, bundle

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class PersonaSynthesizer:
    """Reduce candidates and enforce evidence and mode invariants in code."""

    REQUIRED_LIMITS = (
        "无法捕捉作者的直觉与灵感",
        "本档案只是调研截止日的语料快照",
        "公开表达不等于作者的真实想法",
    )

    def __init__(
        self,
        siliconflow: SiliconFlowClient,
        *,
        output_language: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE,
        max_attempts: int = 4,
    ) -> None:
        self.siliconflow = siliconflow
        self.output_language = output_language
        self.max_attempts = max_attempts

    def synthesize(
        self,
        *,
        persona_id: str,
        name: str,
        mode: PersonaMode,
        map_results: list[MapResult],
        units: tuple[SourceUnit, ...],
        source_info: tuple[SourceInfo, ...],
        expression: ExpressionStatistics,
        research_date: date,
    ) -> PersonaSpec:
        """Run reduce with one repair attempt and build the authoritative spec."""

        registry, gap_registry, bundle = CandidateBundleBuilder().build(map_results, units)
        messages = reduce_messages(
            name=name,
            mode=mode,
            candidate_bundle=bundle,
            expression=expression,
            source_info=source_info,
            output_language=self.output_language,
        )
        last_error = "未知校验错误"
        for attempt in range(self.max_attempts):
            active_messages = messages
            if attempt:
                active_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "上一次提案违反了契约。请返回修正后的完整 JSON 对象，不要解释。"
                            f"校验错误：{last_error}"
                        ),
                    },
                ]
            result = self.siliconflow.chat(
                active_messages,
                thinking=True,
                reasoning_effort="high",
                temperature=0.0,
                max_tokens=8192,
                seed=17,
                response_format="json_object",
                use_cache=True,
                request_timeout_seconds=1200.0,
                request_attempts=2,
                stream=True,
            )
            try:
                reduced = ReduceResult.model_validate(json.loads(result.content))
                validate_reduce_language(reduced, self.output_language)
                return self._assemble(
                    persona_id=persona_id,
                    name=name,
                    mode=mode,
                    reduced=reduced,
                    registry=registry,
                    gap_registry=gap_registry,
                    source_info=source_info,
                    expression=expression,
                    research_date=research_date,
                )
            except (
                json.JSONDecodeError,
                ValidationError,
                StructuredDistillationError,
                OutputLanguageError,
            ) as exc:
                last_error = str(exc)[:3000]
        raise StructuredDistillationError(
            f"Persona 归并在 {self.max_attempts} 次尝试后失败：{last_error}"
        )

    def _assemble(
        self,
        *,
        persona_id: str,
        name: str,
        mode: PersonaMode,
        reduced: ReduceResult,
        registry: dict[str, PersonaEvidence],
        gap_registry: dict[str, dict[str, object]],
        source_info: tuple[SourceInfo, ...],
        expression: ExpressionStatistics,
        research_date: date,
    ) -> PersonaSpec:
        self._validate_all_references(
            reduced,
            registry,
            gap_registry,
            source_info,
        )
        mental_models: list[MentalModel] = []
        for item in reduced.mental_models:
            evidence = self._resolve(item.evidence_ids, registry)
            domains = {entry.domain.strip().casefold() for entry in evidence}
            validation = TripleValidation(
                cross_domain=len(domains) >= 2,
                generative=item.generative,
                exclusive=item.exclusive,
                generative_rationale=item.generative_rationale,
                exclusivity_rationale=item.exclusivity_rationale,
            )
            mental_models.append(
                MentalModel(
                    name=item.name,
                    description=item.description,
                    cross_domain_evidence=evidence,
                    applicability=item.applicability,
                    limits=item.limits,
                    validation=validation,
                )
            )
        heuristics = [
            DecisionHeuristic(
                rule=item.rule,
                trigger=item.trigger,
                example=item.example,
                evidence=self._resolve(item.evidence_ids, registry),
            )
            for item in reduced.decision_heuristics
        ]
        tensions = [
            CoreTension(
                side_a=item.side_a,
                side_b=item.side_b,
                tension_type=item.tension_type,
                evidence=self._resolve(item.evidence_ids, registry),
                interpretation=item.interpretation,
            )
            for item in reduced.core_tensions
        ]
        information_gaps = [
            self._assemble_gap(item, gap_registry) for item in reduced.information_gaps
        ]
        limits = self._unique_readable_text([*reduced.declared_limits, *self.REQUIRED_LIMITS])
        style_tags = reduced.style_tags if mode == "person" else StyleTags()
        style_rules = list(reduced.style_rules)
        taboo_words = reduced.taboo_words
        tics = reduced.tics
        if mode == "topic":
            style_rules = ["使用中性、专业表达，不模拟任何具体作者", *style_rules]
            taboo_words = []
            tics = []
        return PersonaSpec(
            id=persona_id,
            name=name,
            mode=mode,
            output_language=self.output_language,
            mental_models=mental_models,
            decision_heuristics=heuristics,
            expression_dna=ExpressionDNA(
                sentence_fingerprint=expression.fingerprint,
                style_tags=style_tags,
                taboo_words=taboo_words,
                tics=tics,
                style_rules=style_rules,
            ),
            core_tensions=tensions,
            school_divergences=reduced.school_divergences,
            values=reduced.values,
            anti_patterns=reduced.anti_patterns,
            evidence_registry=list(registry.values()),
            source_info=list(source_info),
            research_date=research_date,
            declared_limits=limits,
            information_gaps=information_gaps,
        )

    @staticmethod
    def _resolve(
        identifiers: list[str], registry: dict[str, PersonaEvidence]
    ) -> list[PersonaEvidence]:
        return list(dict.fromkeys(registry[identifier] for identifier in identifiers))

    @staticmethod
    def _unique_readable_text(values: list[str]) -> list[str]:
        """按忽略空白和句末标点的形式去重，同时保留首个原始表述。"""

        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            key = normalized.rstrip("。.!！?？").casefold()
            if key and key not in seen:
                seen.add(key)
                unique.append(normalized)
        return unique

    @staticmethod
    def _validate_all_references(
        reduced: ReduceResult,
        registry: dict[str, PersonaEvidence],
        gap_registry: dict[str, dict[str, object]],
        source_info: tuple[SourceInfo, ...],
    ) -> None:
        identifiers: list[str] = []
        for item in reduced.mental_models:
            identifiers.extend(item.evidence_ids)
        for item in reduced.decision_heuristics:
            identifiers.extend(item.evidence_ids)
        for item in reduced.core_tensions:
            identifiers.extend(item.evidence_ids)
        for divergence in reduced.school_divergences:
            for position in divergence.positions:
                identifiers.extend(position.evidence_ids)
        ungrounded_markers: list[str] = []
        for marker in [*reduced.taboo_words, *reduced.tics]:
            identifiers.extend(marker.evidence_ids)
            if not marker.evidence_ids and marker.confidence != "inferred":
                ungrounded_markers.append(marker.text)
        if ungrounded_markers:
            marker_values = "、".join(ungrounded_markers)
            raise StructuredDistillationError(
                f"以下词汇标记没有 evidence_ids：{marker_values}。"
                "没有证据的词汇标记必须把 confidence 设为 inferred，"
                "否则必须逐字复制有效 evidence_id"
            )
        unknown_evidence_ids = set(identifiers) - registry.keys()
        if unknown_evidence_ids:
            unknown_values = ", ".join(sorted(unknown_evidence_ids))
            raise StructuredDistillationError(
                f"归并结果引用了未知 evidence_id：{unknown_values}。"
                "只能逐字复制候选包 evidence_registry 中已有的 evidence_id"
            )
        gap_ids = {gap_id for gap in reduced.information_gaps for gap_id in gap.supporting_gap_ids}
        unknown_gap_ids = gap_ids - gap_registry.keys()
        if unknown_gap_ids:
            unknown_values = ", ".join(sorted(unknown_gap_ids))
            allowed_values = ", ".join(sorted(gap_registry))
            raise StructuredDistillationError(
                f"归并结果引用了未知 gap_id：{unknown_values}。"
                f"只能从以下合法值中逐字复制：{allowed_values}"
            )
        if any(gap.reviewed_document_count != len(source_info) for gap in reduced.information_gaps):
            raise StructuredDistillationError("信息不足没有基于完整语料清单重新判定")

    @staticmethod
    def _assemble_gap(
        item: ReduceInformationGap,
        gap_registry: dict[str, dict[str, object]],
    ) -> InformationGap:
        source_doc_ids = sorted(
            {
                str(doc_id)
                for gap_id in item.supporting_gap_ids
                for doc_id in gap_registry[gap_id]["source_doc_ids"]
            }
        )
        digest = hashlib.sha256(
            "|".join(sorted(item.supporting_gap_ids)).encode("utf-8")
        ).hexdigest()[:24]
        return InformationGap(
            gap_id=f"corpus_gap_{digest}",
            dimension=item.dimension,
            description=item.description,
            supporting_gap_ids=item.supporting_gap_ids,
            source_doc_ids=source_doc_ids,
            reviewed_document_count=item.reviewed_document_count,
            unresolved_reason=item.unresolved_reason,
            confidence=item.confidence,
        )
