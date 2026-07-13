"""Source-backed map/reduce and recoverable distillation service tests."""

from __future__ import annotations

import json
import threading
import time
from datetime import date

import pytest

from writing_factory.distill.expression import ExpressionAnalyzer
from writing_factory.distill.extraction import (
    PersonaMapExtractor,
    StructuredDistillationError,
)
from writing_factory.distill.language import OutputLanguageError, validate_map_language
from writing_factory.distill.models import (
    DistillationOutcome,
    ExpressionDNA,
    LexicalMarker,
    MapInformationGap,
    MapMentalCandidate,
    MapResult,
    MentalModel,
    PersonaEvidence,
    PersonaSpec,
    ReduceInformationGap,
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

    with pytest.raises(StructuredDistillationError, match="未知 chunk_id") as exc_info:
        extractor.extract("author", "person", _unit())

    assert "chunk_a" in str(exc_info.value)


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
    assert result.information_gaps == [
        MapInformationGap(
            dimension="核心张力",
            description="部分候选张力只有单一证据，不能纳入全局档案",
            reason="当前单元缺少分别支持张力两侧的至少两条证据",
            resolvable_by_more_sources=True,
            confidence="high",
        )
    ]


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
        name="制度与需求协同",
        description="同时考察制度约束与社会需求如何共同塑造出版实践",
        evidence=[
            {
                "chunk_id": "chunk_a",
                "domain": "出版产业",
                "summary": "出版产业的运行需要在制度边界内回应市场变化",
                "confidence": "high",
            },
            {
                "chunk_id": "chunk_b",
                "domain": "公共文化",
                "summary": "公共文化服务需要根据社会需求调整资源配置",
                "confidence": "high",
            },
        ],
        generative_rationale="可用于推断作者面对新型出版问题时如何权衡制度与需求",
        exclusivity_rationale="强调制度边界与现实需求的联动，而非一般性的平衡原则",
    )
    mapped = MapResult(unit_id=unit.unit_id, mental_candidates=[candidate])
    return unit, mapped


def test_reduce_builds_three_source_backed_models() -> None:
    unit, mapped = _synthesis_sources()
    registry, _gap_registry, _bundle = CandidateBundleBuilder().build([mapped], (unit,))
    evidence_ids = list(registry)
    reduced = ReduceResult(
        mental_models=[
            ReduceMentalModel(
                name=f"协同分析模型{index}",
                description="从制度条件与现实需求的互动关系解释出版实践",
                evidence_ids=evidence_ids,
                applicability="适用于制度约束与社会需求同时存在的出版问题",
                limits="不适用于缺少制度背景或现实需求证据的情形",
                generative=True,
                exclusive=True,
                generative_rationale="能够推断作者对新兴出版治理问题的可能判断",
                exclusivity_rationale="体现制度与需求联动的稳定分析偏好",
            )
            for index in range(3)
        ],
        style_tags=StyleTags(),
        declared_limits=[f"{item}。" for item in PersonaSynthesizer.REQUIRED_LIMITS],
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
    assert len(persona.declared_limits) == 3
    assert client.calls[0]["reasoning_effort"] == "high"
    assert client.calls[0]["max_tokens"] == 8192
    assert client.calls[0]["request_timeout_seconds"] == 1200.0
    assert client.calls[0]["request_attempts"] == 2


def test_topic_reduce_keeps_school_divergence_and_neutralizes_voice() -> None:
    unit, mapped = _synthesis_sources()
    registry, _gap_registry, _bundle = CandidateBundleBuilder().build([mapped], (unit,))
    evidence_ids = list(registry)
    reduced = ReduceResult(
        mental_models=[
            ReduceMentalModel(
                name=f"出版研究框架{index}",
                description="从制度安排与公共价值的互动关系解释出版现象",
                evidence_ids=evidence_ids,
                applicability="适用于比较不同出版治理路径及其社会效果",
                limits="不适用于缺少制度条件和公共价值证据的问题",
                generative=True,
                exclusive=True,
                generative_rationale="能够推导新型出版现象中不同治理路径的分歧",
                exclusivity_rationale="保留出版研究中特有的制度与公共价值关系",
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
        declared_limits=["无法覆盖所有研究传统", "档案只是语料快照", "公开材料不等于完整学术立场"],
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
            summary="出版活动需要回应产业结构与制度安排的共同变化",
            confidence="high",
        ),
        PersonaEvidence(
            evidence_id="ev_b",
            chunk_id="chunk_b",
            doc_id="doc_b",
            domain="文化",
            summary="文化服务需要兼顾公共价值与社会需求的持续变化",
            confidence="high",
        ),
    ]
    validation = TripleValidation(
        cross_domain=True,
        generative=True,
        exclusive=True,
        generative_rationale="能够推断作者面对新问题时的分析路径",
        exclusivity_rationale="体现作者区别于通用原则的稳定判断方式",
    )
    fingerprint = ExpressionAnalyzer().analyze(["第一段。第二段。"])
    return PersonaSpec(
        id=persona_id,
        name="测试作者",
        mode="person",
        mental_models=[
            MentalModel(
                name=f"制度需求联动模型{index}",
                description="通过制度条件与社会需求的互动解释出版和文化现象",
                cross_domain_evidence=evidence,
                applicability="适用于同时涉及制度约束和社会需求的问题",
                limits="缺少现实材料时不能据此推断具体立场",
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
        declared_limits=["无法捕捉作者的直觉与灵感", "档案只是语料快照", "公开表达不等于真实想法"],
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
        self.map_orders: list[list[str]] = []

    def synthesize(self, *, persona_id, map_results, **kwargs):
        self.calls += 1
        self.map_orders.append([result.unit_id for result in map_results])
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
        map_concurrency=1,
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


class ConcurrentFailingOnceExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.failed = False
        self.lock = threading.Lock()

    def extract(self, name, mode, unit):
        with self.lock:
            self.calls.append(unit.unit_id)
            should_fail = unit.unit_id == "unit_0" and not self.failed
            if should_fail:
                self.failed = True
        time.sleep(0.01 if should_fail else 0.05)
        if should_fail:
            raise RuntimeError("planned concurrent failure")
        return MapResult(unit_id=unit.unit_id)


def test_service_saves_successful_in_flight_maps_before_raising(tmp_path) -> None:
    database = Database(tmp_path / "concurrent_resume.db")
    database.initialize()
    KnowledgeBaseRepository(database).ensure_default()
    extractor = ConcurrentFailingOnceExtractor()
    synthesizer = FakeSynthesizer()
    service = DistillationService(
        PersonaRepository(database),
        FakeCorpusBuilder(_service_corpus(unit_count=3)),
        extractor,
        synthesizer,
        map_concurrency=2,
    )

    with pytest.raises(RuntimeError, match="planned concurrent failure"):
        service.distill(kb_id="kb_default", name="测试作者", mode="person")
    service.distill(kb_id="kb_default", name="测试作者", mode="person")

    assert extractor.calls.count("unit_0") == 2
    assert extractor.calls.count("unit_1") == 1
    assert extractor.calls.count("unit_2") == 1
    assert synthesizer.calls == 1


def test_map_schema_uses_english_keys_and_chinese_descriptions() -> None:
    schema = MapResult.model_json_schema()

    assert "mental_candidates" in schema["properties"]
    assert "information_gaps" in schema["properties"]
    assert "心智模型" in schema["properties"]["mental_candidates"]["description"]
    gap_schema = schema["$defs"]["MapInformationGap"]
    assert "dimension" in gap_schema["properties"]
    assert "分析维度" in gap_schema["properties"]["dimension"]["description"]


def test_reduce_schema_uses_english_keys_and_chinese_descriptions() -> None:
    schema = ReduceResult.model_json_schema()

    assert "mental_models" in schema["properties"]
    assert "information_gaps" in schema["properties"]
    assert "全语料复核" in schema["properties"]["information_gaps"]["description"]
    model_schema = schema["$defs"]["ReduceMentalModel"]
    assert "generative_rationale" in model_schema["properties"]
    assert "生成力判断理由" in model_schema["properties"]["generative_rationale"]["description"]


def test_chinese_quality_gate_rejects_english_map_output() -> None:
    result = MapResult(
        unit_id="unit_one",
        mental_candidates=[
            MapMentalCandidate(
                name="Market adaptation",
                description="The author adapts institutions to market demand.",
                evidence=[
                    {
                        "chunk_id": "chunk_a",
                        "domain": "Publishing",
                        "summary": "Institutions respond to changing demand.",
                    }
                ],
                generative_rationale="This predicts future positions.",
                exclusivity_rationale="This is distinctive.",
            )
        ],
    )

    with pytest.raises(OutputLanguageError, match="简体中文"):
        validate_map_language(result, "zh-CN")


def test_reduce_rechecks_local_information_gap_against_full_corpus() -> None:
    unit, mapped = _synthesis_sources()
    mapped = mapped.model_copy(
        update={
            "information_gaps": [
                MapInformationGap(
                    dimension="时间演变",
                    description="当前单元缺少早期研究材料",
                    reason="本单元只包含近期论文切片",
                    resolvable_by_more_sources=True,
                    confidence="medium",
                )
            ]
        }
    )
    registry, gap_registry, _bundle = CandidateBundleBuilder().build([mapped], (unit,))
    reduced = _reduced_fixture(list(registry))
    gap_id = next(iter(gap_registry))
    reduced = reduced.model_copy(
        update={
            "information_gaps": [
                ReduceInformationGap(
                    dimension="时间演变",
                    description="全部语料仍缺少作者早期研究阶段的连续材料",
                    supporting_gap_ids=[gap_id],
                    reviewed_document_count=2,
                    unresolved_reason="完整文档清单中的两篇材料均集中于近期，无法验证早期变化",
                    confidence="medium",
                )
            ]
        }
    )
    synthesizer = PersonaSynthesizer(FakeChatClient([reduced.model_dump_json()]))

    persona = synthesizer.synthesize(
        persona_id="persona_gap",
        name="测试作者",
        mode="person",
        map_results=[mapped],
        units=(unit,),
        source_info=(
            SourceInfo(doc_id="doc_a", title="甲", filename="a.txt", chunk_count=1),
            SourceInfo(doc_id="doc_b", title="乙", filename="b.txt", chunk_count=1),
        ),
        expression=ExpressionAnalyzer().analyze(["这些材料用于分析作者研究路径的变化。"]),
        research_date=date(2026, 7, 13),
    )

    assert persona.information_gaps[0].supporting_gap_ids == [gap_id]
    assert persona.information_gaps[0].reviewed_document_count == 2
    assert persona.information_gaps[0].source_doc_ids == ["doc_a", "doc_b"]


def test_reduce_rejects_gap_not_rechecked_against_every_document() -> None:
    unit, mapped = _synthesis_sources()
    mapped = mapped.model_copy(
        update={
            "information_gaps": [
                MapInformationGap(
                    dimension="研究阶段",
                    description="当前单元缺少早期材料",
                    reason="这里只包含近期论文",
                    resolvable_by_more_sources=True,
                    confidence="medium",
                )
            ]
        }
    )
    registry, gap_registry, _bundle = CandidateBundleBuilder().build([mapped], (unit,))
    reduced = _reduced_fixture(list(registry)).model_copy(
        update={
            "information_gaps": [
                ReduceInformationGap(
                    dimension="研究阶段",
                    description="全局材料可能仍缺少早期研究内容",
                    supporting_gap_ids=[next(iter(gap_registry))],
                    reviewed_document_count=1,
                    unresolved_reason="尚未检查完整的两篇文档清单",
                    confidence="low",
                )
            ]
        }
    )
    synthesizer = PersonaSynthesizer(
        FakeChatClient([reduced.model_dump_json()]),
        max_attempts=1,
    )

    with pytest.raises(StructuredDistillationError, match="完整语料清单"):
        synthesizer.synthesize(
            persona_id="persona_bad_gap",
            name="测试作者",
            mode="person",
            map_results=[mapped],
            units=(unit,),
            source_info=(
                SourceInfo(doc_id="doc_a", title="甲", filename="a.txt", chunk_count=1),
                SourceInfo(doc_id="doc_b", title="乙", filename="b.txt", chunk_count=1),
            ),
            expression=ExpressionAnalyzer().analyze(["完整语料应当接受全局复核。"]),
            research_date=date(2026, 7, 13),
        )


def test_reduce_reports_allowed_gap_ids_for_repair() -> None:
    unit, mapped = _synthesis_sources()
    mapped = mapped.model_copy(
        update={
            "information_gaps": [
                MapInformationGap(
                    dimension="研究阶段",
                    description="当前单元缺少早期材料",
                    reason="这里只包含近期论文",
                    resolvable_by_more_sources=True,
                    confidence="medium",
                )
            ]
        }
    )
    registry, gap_registry, _bundle = CandidateBundleBuilder().build([mapped], (unit,))
    reduced = _reduced_fixture(list(registry)).model_copy(
        update={
            "information_gaps": [
                ReduceInformationGap(
                    dimension="研究阶段",
                    description="全局材料仍缺少早期研究内容",
                    supporting_gap_ids=["gap_invented"],
                    reviewed_document_count=2,
                    unresolved_reason="完整语料无法验证作者早期研究阶段",
                    confidence="medium",
                )
            ]
        }
    )

    with pytest.raises(StructuredDistillationError, match="未知 gap_id") as exc_info:
        PersonaSynthesizer._validate_all_references(
            reduced,
            registry,
            gap_registry,
            (
                SourceInfo(doc_id="doc_a", title="甲", filename="a.txt", chunk_count=1),
                SourceInfo(doc_id="doc_b", title="乙", filename="b.txt", chunk_count=1),
            ),
        )

    assert next(iter(gap_registry)) in str(exc_info.value)


def test_reduce_can_repair_sequential_independent_contract_failures() -> None:
    unit, mapped = _synthesis_sources()
    mapped = mapped.model_copy(
        update={
            "information_gaps": [
                MapInformationGap(
                    dimension="研究阶段",
                    description="当前单元缺少早期材料",
                    reason="这里只包含近期论文",
                    resolvable_by_more_sources=True,
                    confidence="medium",
                )
            ]
        }
    )
    registry, _gap_registry, _bundle = CandidateBundleBuilder().build([mapped], (unit,))
    unknown_gap = _reduced_fixture(list(registry)).model_copy(
        update={
            "information_gaps": [
                ReduceInformationGap(
                    dimension="研究阶段",
                    description="全局材料仍缺少早期研究内容",
                    supporting_gap_ids=["gap_invented"],
                    reviewed_document_count=2,
                    unresolved_reason="完整语料无法验证作者早期研究阶段",
                    confidence="medium",
                )
            ]
        }
    )
    ungrounded_marker = _reduced_fixture(list(registry)).model_copy(
        update={"tics": [LexicalMarker(text="显然", confidence="high", evidence_ids=[])]}
    )
    repaired = _reduced_fixture(list(registry)).model_copy(
        update={"tics": [LexicalMarker(text="显然", confidence="inferred", evidence_ids=[])]}
    )
    client = FakeChatClient(
        [
            unknown_gap.model_dump_json(),
            ungrounded_marker.model_dump_json(),
            repaired.model_dump_json(),
        ]
    )

    persona = PersonaSynthesizer(client).synthesize(
        persona_id="persona_repairs",
        name="测试作者",
        mode="person",
        map_results=[mapped],
        units=(unit,),
        source_info=(
            SourceInfo(doc_id="doc_a", title="甲", filename="a.txt", chunk_count=1),
            SourceInfo(doc_id="doc_b", title="乙", filename="b.txt", chunk_count=1),
        ),
        expression=ExpressionAnalyzer().analyze(["完整语料需要经过多轮契约校验。"]),
        research_date=date(2026, 7, 13),
    )

    assert len(client.calls) == 3
    assert persona.expression_dna.tics[0].confidence == "inferred"


def _reduced_fixture(evidence_ids: list[str]) -> ReduceResult:
    return ReduceResult(
        mental_models=[
            ReduceMentalModel(
                name=f"全局协同模型{index}",
                description="从制度条件与公共需求的互动关系解释出版实践",
                evidence_ids=evidence_ids,
                applicability="适用于制度约束和公共需求同时存在的研究问题",
                limits="缺少制度背景或公共需求证据时不能直接使用",
                generative=True,
                exclusive=True,
                generative_rationale="能够推断作者对新型出版问题的分析路径",
                exclusivity_rationale="体现制度条件与公共需求联动的稳定偏好",
            )
            for index in range(3)
        ],
        style_tags=StyleTags(),
        declared_limits=["无法捕捉作者的直觉与灵感", "档案只是语料快照", "公开表达不等于真实想法"],
    )


class TrackingExtractor:
    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.calls: list[str] = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def extract(self, name, mode, unit):
        with self.lock:
            self.calls.append(unit.unit_id)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            return MapResult(unit_id=unit.unit_id)
        finally:
            with self.lock:
                self.active -= 1


def _service_corpus(unit_count: int = 6) -> SourceCorpus:
    units = tuple(
        SourceUnit(
            unit_id=f"unit_{index}",
            segments=[
                SourceSegment(
                    chunk_id=f"chunk_{index}",
                    doc_id="doc_a",
                    document_title="测试文档",
                    filename="source.txt",
                    text=f"测试语料{index}",
                )
            ],
        )
        for index in range(unit_count)
    )
    return SourceCorpus(
        units=units,
        source_info=(
            SourceInfo(
                doc_id="doc_a", title="测试文档", filename="source.txt", chunk_count=unit_count
            ),
        ),
        source_hash="concurrency_source_hash",
    )


def test_service_runs_three_maps_concurrently_and_preserves_reduce_order(tmp_path) -> None:
    database = Database(tmp_path / "concurrency.db")
    database.initialize()
    KnowledgeBaseRepository(database).ensure_default()
    extractor = TrackingExtractor()
    synthesizer = FakeSynthesizer()
    corpus = _service_corpus()
    service = DistillationService(
        PersonaRepository(database),
        FakeCorpusBuilder(corpus),
        extractor,
        synthesizer,
        map_concurrency=3,
    )

    service.distill(kb_id="kb_default", name="测试作者", mode="person")

    assert extractor.peak == 3
    assert synthesizer.map_orders == [[unit.unit_id for unit in corpus.units]]


def test_reduce_version_change_reuses_compatible_maps_across_runs(tmp_path) -> None:
    class NextReduceService(DistillationService):
        REDUCE_PIPELINE_VERSION = "persona-reduce-test-next"

    database = Database(tmp_path / "reuse.db")
    database.initialize()
    KnowledgeBaseRepository(database).ensure_default()
    repository = PersonaRepository(database)
    corpus = _service_corpus(unit_count=2)
    first_extractor = TrackingExtractor(delay=0.0)
    DistillationService(
        repository,
        FakeCorpusBuilder(corpus),
        first_extractor,
        FakeSynthesizer(),
        map_concurrency=1,
    ).distill(kb_id="kb_default", name="测试作者", mode="person")
    second_extractor = TrackingExtractor(delay=0.0)
    progress_messages: list[str] = []

    outcome = NextReduceService(
        repository,
        FakeCorpusBuilder(corpus),
        second_extractor,
        FakeSynthesizer(),
        map_concurrency=1,
    ).distill(
        kb_id="kb_default",
        name="测试作者",
        mode="person",
        progress=lambda _percent, message: progress_messages.append(message),
    )

    assert not outcome.reused
    assert first_extractor.calls == ["unit_0", "unit_1"]
    assert second_extractor.calls == []
    assert "复用思维候选" in progress_messages
