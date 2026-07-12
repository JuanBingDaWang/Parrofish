"""Source-backed map/reduce and recoverable distillation service tests."""

from __future__ import annotations

import json
from datetime import date

import pytest

from writing_factory.distill.expression import ExpressionAnalyzer
from writing_factory.distill.extraction import (
    PersonaMapExtractor,
    StructuredDistillationError,
)
from writing_factory.distill.models import (
    DistillationOutcome,
    ExpressionDNA,
    MapMentalCandidate,
    MapResult,
    MentalModel,
    PersonaEvidence,
    PersonaSpec,
    ReduceMentalModel,
    ReduceResult,
    SourceInfo,
    SourceSegment,
    SourceUnit,
    StyleTags,
    TripleValidation,
)
from writing_factory.distill.service import DistillationService
from writing_factory.distill.sources import SourceCorpus
from writing_factory.distill.synthesis import CandidateBundleBuilder, PersonaSynthesizer
from writing_factory.llm.models import ChatResult
from writing_factory.store import Database
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.persona_repository import PersonaRepository


class FakeChatClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return ChatResult(content=self.responses.pop(0), model="fixture")


def _unit() -> SourceUnit:
    return SourceUnit(
        unit_id="unit_one",
        segments=[
            SourceSegment(
                chunk_id="chunk_a",
                doc_id="doc_a",
                document_title="A",
                filename="a.txt",
                text="source text",
            )
        ],
    )


def test_map_rejects_unknown_chunk_id() -> None:
    response = MapResult(
        unit_id="unit_one",
        mental_candidates=[
            MapMentalCandidate(
                name="candidate",
                description="description",
                evidence=[
                    {
                        "chunk_id": "invented_chunk",
                        "domain": "domain",
                        "summary": "summary",
                    }
                ],
                generative_rationale="generative",
                exclusivity_rationale="exclusive",
            )
        ],
    ).model_dump_json()
    extractor = PersonaMapExtractor(FakeChatClient([response]), max_attempts=1)

    with pytest.raises(StructuredDistillationError, match="unknown chunk_id"):
        extractor.extract("author", "person", _unit())


def test_map_discards_single_anchor_tension_as_an_information_gap() -> None:
    response = {
        "unit_id": "unit_one",
        "tensions": [
            {
                "side_a": "position A",
                "side_b": "position B",
                "tension_type": "domain",
                "evidence": [
                    {
                        "chunk_id": "chunk_a",
                        "domain": "domain",
                        "summary": "only one anchor",
                    }
                ],
            }
        ],
    }
    extractor = PersonaMapExtractor(FakeChatClient([json.dumps(response)]), max_attempts=1)

    result = extractor.extract("author", "person", _unit())

    assert result.tensions == []
    assert result.insufficient_dimensions == ["部分候选张力只有单一证据，未纳入档案"]


def _synthesis_sources():
    unit = SourceUnit(
        unit_id="unit_two",
        segments=[
            SourceSegment(
                chunk_id="chunk_a",
                doc_id="doc_a",
                document_title="A",
                filename="a.txt",
                text="source A",
            ),
            SourceSegment(
                chunk_id="chunk_b",
                doc_id="doc_b",
                document_title="B",
                filename="b.txt",
                text="source B",
            ),
        ],
    )
    candidate = MapMentalCandidate(
        name="candidate",
        description="description",
        evidence=[
            {
                "chunk_id": "chunk_a",
                "domain": "出版产业",
                "summary": "evidence A",
                "confidence": "high",
            },
            {
                "chunk_id": "chunk_b",
                "domain": "公共文化",
                "summary": "evidence B",
                "confidence": "high",
            },
        ],
        generative_rationale="predicts new positions",
        exclusivity_rationale="distinctive",
    )
    mapped = MapResult(unit_id=unit.unit_id, mental_candidates=[candidate])
    return unit, mapped


def test_reduce_builds_three_source_backed_models() -> None:
    unit, mapped = _synthesis_sources()
    registry, _bundle = CandidateBundleBuilder().build([mapped], (unit,))
    evidence_ids = list(registry)
    reduced = ReduceResult(
        mental_models=[
            ReduceMentalModel(
                name=f"model {index}",
                description="description",
                evidence_ids=evidence_ids,
                applicability="application",
                limits="limit",
                generative=True,
                exclusive=True,
                generative_rationale="generative",
                exclusivity_rationale="exclusive",
            )
            for index in range(3)
        ],
        style_tags=StyleTags(),
        declared_limits=["limit one", "limit two", "limit three"],
    )
    client = FakeChatClient([reduced.model_dump_json()])
    synthesizer = PersonaSynthesizer(client)
    expression = ExpressionAnalyzer().analyze(["我一定研究现实。然而也许会变化。"])

    persona = synthesizer.synthesize(
        persona_id="persona",
        name="author",
        mode="person",
        map_results=[mapped],
        units=(unit,),
        source_info=(
            SourceInfo(doc_id="doc_a", title="A", filename="a.txt", chunk_count=1),
            SourceInfo(doc_id="doc_b", title="B", filename="b.txt", chunk_count=1),
        ),
        expression=expression,
        research_date=date(2026, 7, 12),
    )

    assert len(persona.mental_models) == 3
    assert len(persona.evidence_registry) == 2
    assert all(model.validation.passed for model in persona.mental_models)
    assert client.calls[0]["reasoning_effort"] == "high"
    assert client.calls[0]["max_tokens"] == 8192


def test_topic_reduce_keeps_school_divergence_and_neutralizes_voice() -> None:
    unit, mapped = _synthesis_sources()
    registry, _bundle = CandidateBundleBuilder().build([mapped], (unit,))
    evidence_ids = list(registry)
    reduced = ReduceResult(
        mental_models=[
            ReduceMentalModel(
                name=f"framework {index}",
                description="description",
                evidence_ids=evidence_ids,
                applicability="application",
                limits="limit",
                generative=True,
                exclusive=True,
                generative_rationale="generative",
                exclusivity_rationale="exclusive",
            )
            for index in range(3)
        ],
        style_tags=StyleTags(formal_to_colloquial=0.8),
        tics=[{"text": "显然", "confidence": "inferred"}],
        school_divergences=[
            {
                "question": "出版应优先市场还是公共价值？",
                "positions": [
                    {
                        "label": "市场路径",
                        "position": "优先需求响应",
                        "evidence_ids": [evidence_ids[0]],
                    },
                    {
                        "label": "公共路径",
                        "position": "优先文化服务",
                        "evidence_ids": [evidence_ids[1]],
                    },
                ],
            }
        ],
        declared_limits=["limit one", "limit two", "limit three"],
    )
    synthesizer = PersonaSynthesizer(FakeChatClient([reduced.model_dump_json()]))
    expression = ExpressionAnalyzer().analyze(["主题材料。不同学派存在分歧。"])

    persona = synthesizer.synthesize(
        persona_id="topic_persona",
        name="出版研究",
        mode="topic",
        map_results=[mapped],
        units=(unit,),
        source_info=(
            SourceInfo(doc_id="doc_a", title="A", filename="a.txt", chunk_count=1),
            SourceInfo(doc_id="doc_b", title="B", filename="b.txt", chunk_count=1),
        ),
        expression=expression,
        research_date=date(2026, 7, 12),
    )

    assert len(persona.school_divergences) == 1
    assert persona.expression_dna.style_tags == StyleTags()
    assert persona.expression_dna.tics == []
    assert persona.expression_dna.style_rules[0] == "使用中性、专业表达，不模拟任何具体作者"


def _persona(persona_id: str) -> PersonaSpec:
    evidence = [
        PersonaEvidence(
            evidence_id="ev_a",
            chunk_id="chunk_a",
            doc_id="doc_a",
            domain="出版",
            summary="A",
            confidence="high",
        ),
        PersonaEvidence(
            evidence_id="ev_b",
            chunk_id="chunk_b",
            doc_id="doc_b",
            domain="文化",
            summary="B",
            confidence="high",
        ),
    ]
    validation = TripleValidation(
        cross_domain=True,
        generative=True,
        exclusive=True,
        generative_rationale="yes",
        exclusivity_rationale="yes",
    )
    fingerprint = ExpressionAnalyzer().analyze(["第一段。第二段。"])
    return PersonaSpec(
        id=persona_id,
        name="author",
        mode="person",
        mental_models=[
            MentalModel(
                name=f"model {index}",
                description="description",
                cross_domain_evidence=evidence,
                applicability="application",
                limits="limits",
                validation=validation,
            )
            for index in range(3)
        ],
        expression_dna=ExpressionDNA(
            sentence_fingerprint=fingerprint.fingerprint,
            style_tags=StyleTags(),
        ),
        evidence_registry=evidence,
        source_info=[SourceInfo(doc_id="doc_a", title="A", filename="a.txt", chunk_count=1)],
        research_date=date(2026, 7, 12),
        declared_limits=["one", "two", "three"],
    )


class FakeCorpusBuilder:
    def __init__(self, corpus: SourceCorpus) -> None:
        self.corpus = corpus

    def build(self, kb_id, *, doc_ids=None):
        return self.corpus


class FailingOnceExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.failed = False

    def extract(self, name, mode, unit):
        self.calls.append(unit.unit_id)
        if unit.unit_id == "unit_2" and not self.failed:
            self.failed = True
            raise RuntimeError("planned failure")
        return MapResult(unit_id=unit.unit_id)


class FakeSynthesizer:
    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, *, persona_id, **kwargs):
        self.calls += 1
        return _persona(persona_id)


def test_service_resumes_maps_and_reuses_ready_profile(tmp_path) -> None:
    database = Database(tmp_path / "distill.db")
    database.initialize()
    KnowledgeBaseRepository(database).ensure_default()
    repository = PersonaRepository(database)
    units = tuple(
        SourceUnit(
            unit_id=f"unit_{index}",
            segments=[
                SourceSegment(
                    chunk_id=f"chunk_{index}",
                    doc_id="doc_a",
                    document_title="A",
                    filename="a.txt",
                    text=f"source {index}",
                )
            ],
        )
        for index in (1, 2)
    )
    corpus = SourceCorpus(
        units=units,
        source_info=(SourceInfo(doc_id="doc_a", title="A", filename="a.txt", chunk_count=2),),
        source_hash="source_hash",
    )
    extractor = FailingOnceExtractor()
    synthesizer = FakeSynthesizer()
    service = DistillationService(
        repository,
        FakeCorpusBuilder(corpus),
        extractor,
        synthesizer,
    )

    with pytest.raises(RuntimeError, match="planned failure"):
        service.distill(kb_id="kb_default", name="author", mode="person")
    second = service.distill(kb_id="kb_default", name="author", mode="person")
    third = service.distill(kb_id="kb_default", name="author", mode="person")

    assert isinstance(second, DistillationOutcome)
    assert not second.reused
    assert third.reused
    assert extractor.calls.count("unit_1") == 1
    assert extractor.calls.count("unit_2") == 2
    assert synthesizer.calls == 1
