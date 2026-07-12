"""Evidence registry, Nüwa reduce call, and code-enforced PersonaSpec assembly."""

from __future__ import annotations

import hashlib
import json
from datetime import date

from pydantic import ValidationError

from writing_factory.distill.expression import ExpressionStatistics
from writing_factory.distill.extraction import StructuredDistillationError
from writing_factory.distill.models import (
    CoreTension,
    DecisionHeuristic,
    ExpressionDNA,
    MapResult,
    MentalModel,
    PersonaEvidence,
    PersonaMode,
    PersonaSpec,
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
    ) -> tuple[dict[str, PersonaEvidence], dict[str, object]]:
        """Return a final evidence registry and compact reducer candidate bundle."""

        segment_by_chunk = {
            segment.chunk_id: segment for unit in units for segment in unit.segments
        }
        registry: dict[str, PersonaEvidence] = {}

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
        gaps: list[str] = []
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
            gaps.extend(result.insufficient_dimensions)
        bundle: dict[str, object] = {
            "evidence_registry": [item.model_dump(mode="json") for item in registry.values()],
            "mental_candidates": mental_candidates,
            "heuristic_candidates": heuristic_candidates,
            "tensions": tensions,
            "value_signals": self._unique(values),
            "anti_pattern_signals": self._unique(anti_patterns),
            "style_observations": self._unique(style_observations),
            "information_gaps": self._unique(gaps),
        }
        return registry, bundle

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

    def __init__(self, siliconflow: SiliconFlowClient, *, max_attempts: int = 2) -> None:
        self.siliconflow = siliconflow
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

        registry, bundle = CandidateBundleBuilder().build(map_results, units)
        messages = reduce_messages(
            name=name,
            mode=mode,
            candidate_bundle=bundle,
            expression=expression,
        )
        last_error = "unknown validation error"
        for attempt in range(self.max_attempts):
            active_messages = messages
            if attempt:
                active_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "The previous proposal violated the contract. Return a corrected full "
                            f"JSON object. Validation error: {last_error}"
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
                request_timeout_seconds=600.0,
                request_attempts=1,
                stream=True,
            )
            try:
                reduced = ReduceResult.model_validate(json.loads(result.content))
                return self._assemble(
                    persona_id=persona_id,
                    name=name,
                    mode=mode,
                    reduced=reduced,
                    registry=registry,
                    source_info=source_info,
                    expression=expression,
                    research_date=research_date,
                )
            except (
                json.JSONDecodeError,
                ValidationError,
                StructuredDistillationError,
            ) as exc:
                last_error = str(exc)[:500]
        raise StructuredDistillationError(
            f"Persona reduction failed after {self.max_attempts} attempts: {last_error}"
        )

    def _assemble(
        self,
        *,
        persona_id: str,
        name: str,
        mode: PersonaMode,
        reduced: ReduceResult,
        registry: dict[str, PersonaEvidence],
        source_info: tuple[SourceInfo, ...],
        expression: ExpressionStatistics,
        research_date: date,
    ) -> PersonaSpec:
        self._validate_all_references(reduced, registry)
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
        limits = list(dict.fromkeys([*reduced.declared_limits, *self.REQUIRED_LIMITS]))
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
            information_gaps=reduced.information_gaps,
        )

    @staticmethod
    def _resolve(
        identifiers: list[str], registry: dict[str, PersonaEvidence]
    ) -> list[PersonaEvidence]:
        return list(dict.fromkeys(registry[identifier] for identifier in identifiers))

    @staticmethod
    def _validate_all_references(
        reduced: ReduceResult, registry: dict[str, PersonaEvidence]
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
        for marker in [*reduced.taboo_words, *reduced.tics]:
            identifiers.extend(marker.evidence_ids)
            if not marker.evidence_ids and marker.confidence != "inferred":
                raise StructuredDistillationError(
                    "Lexical markers without evidence must be marked inferred"
                )
        if set(identifiers) - registry.keys():
            raise StructuredDistillationError("Reducer cited unknown evidence_id values")
