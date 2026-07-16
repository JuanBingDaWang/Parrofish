"""学术蒸馏 v2 的留出、选模与运行时隔离测试。"""

from __future__ import annotations

import json
import threading
import time
from datetime import date

import pytest

from writing_factory.distill.academic import (
    AcademicModelValidation,
    CandidateAssessment,
    CandidateCluster,
    CandidateRecord,
    CandidateRegistry,
    ExclusivityAssessment,
    PaperMentalCandidate,
    PaperProfile,
)
from writing_factory.distill.academic_pipeline import (
    AcademicDistillationEngine,
    choose_holdout_doc_ids,
)
from writing_factory.distill.expression import ExpressionAnalyzer
from writing_factory.distill.models import (
    ExpressionDNA,
    ExtractedEvidence,
    MapMentalCandidate,
    MapResult,
    MentalModel,
    PersonaEvidence,
    PersonaSpec,
    SourceInfo,
    SourceSegment,
    SourceUnit,
    StyleTags,
    TripleValidation,
)
from writing_factory.distill.prompts import academic_supplement_messages
from writing_factory.distill.runtime import build_runtime_persona
from writing_factory.distill.selection import select_academic_candidates
from writing_factory.distill.synthesis import CandidateBundleBuilder, PersonaSynthesizer
from writing_factory.generate.source_policy import (
    build_generation_source_policy,
    find_suspicious_source_overlap,
)
from writing_factory.llm.models import ChatResult
from writing_factory.store import Database
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.persona_repository import PersonaRepository


def _paper_candidate(doc: int, candidate: int) -> PaperMentalCandidate:
    return PaperMentalCandidate(
        paper_candidate_id=f"paper_{doc}_{candidate}",
        map_candidate_ids=[f"map_{doc}_{candidate}"],
        operation="argument_structure",
        name=f"候选{candidate}",
        description="通过明确的问题层次推进学术论证。",
        evidence_ids=[f"ev_{doc}_{candidate}"],
        applicability="需要组织多层论证时使用。",
        limits="材料不足时不能强行建立层次。",
        research_context=f"论文{doc}的论证结构",
    )


def _cluster(candidate: int, docs: tuple[int, ...]) -> CandidateCluster:
    return CandidateCluster(
        candidate_id=f"candidate_{candidate}",
        operation="argument_structure",
        name=f"模型{candidate}",
        description="从问题层次组织证据和推论。",
        paper_candidate_ids=[f"paper_{doc}_{candidate}" for doc in docs],
        evidence_ids=[f"ev_{doc}_{candidate}" for doc in docs],
        applicability="多层学术问题。",
        limits="不适用于单一事实说明。",
        attribution_scope="author_specific",
        attribution_rationale="在不同论文中反复出现。",
    )


def _profiles(candidate_count: int, docs: tuple[int, ...] = (1, 2, 3)) -> list[PaperProfile]:
    return [
        PaperProfile(
            doc_id=f"doc_{doc}",
            candidates=[_paper_candidate(doc, candidate) for candidate in range(candidate_count)],
        )
        for doc in docs
    ]


def test_holdout_count_is_adaptive_and_never_exceeds_two() -> None:
    def sources(count: int) -> tuple[SourceInfo, ...]:
        return tuple(
            SourceInfo(
                doc_id=f"doc_{index}",
                title=f"论文{index:02d}",
                filename=f"{index}.pdf",
                chunk_count=1,
            )
            for index in range(count)
        )

    assert choose_holdout_doc_ids(sources(3)) == []
    assert len(choose_holdout_doc_ids(sources(4))) == 1
    assert len(choose_holdout_doc_ids(sources(7))) == 1
    assert len(choose_holdout_doc_ids(sources(8))) == 2
    assert len(choose_holdout_doc_ids(sources(20))) == 2


def test_personal_models_exclude_generic_conventions_after_reaching_three() -> None:
    profiles = _profiles(4)
    clusters = [_cluster(index, (1, 2, 3)) for index in range(4)]
    exclusivity = [
        ExclusivityAssessment(
            candidate_id=f"candidate_{index}",
            specificity="author_distinctive" if index < 3 else "field_conventional",
            rationale="目标与对照语料呈现出不同的稳定程度。",
        )
        for index in range(4)
    ]

    registry = select_academic_candidates(
        clusters=clusters,
        target_profiles=profiles,
        target_doc_ids=[profile.doc_id for profile in profiles],
        holdout_doc_ids=[],
        control_doc_ids=["control_1"],
        domain="出版学",
        generative=[],
        exclusivity=exclusivity,
    )

    core = [item for item in registry.records if item.selected_as == "core"]
    conventions = [item for item in registry.records if item.selected_as == "convention"]
    assert len(core) == 3
    assert all(item.validation.specificity == "author_distinctive" for item in core)
    assert [item.candidate.candidate_id for item in conventions] == ["candidate_3"]


def test_generic_model_only_fills_core_when_personal_models_are_fewer_than_three() -> None:
    profiles = _profiles(4, docs=(1, 2))
    clusters = [_cluster(index, (1, 2)) for index in range(4)]
    exclusivity = [
        ExclusivityAssessment(
            candidate_id=f"candidate_{index}",
            specificity="author_distinctive" if index < 2 else "field_conventional",
            rationale="根据同领域对照语料进行分层。",
        )
        for index in range(4)
    ]

    registry = select_academic_candidates(
        clusters=clusters,
        target_profiles=profiles,
        target_doc_ids=[profile.doc_id for profile in profiles],
        holdout_doc_ids=[],
        control_doc_ids=["control_1"],
        domain="出版学",
        generative=[],
        exclusivity=exclusivity,
    )

    core = sorted(
        (item for item in registry.records if item.selected_as == "core"),
        key=lambda item: item.selection_rank or 99,
    )
    assert [item.validation.specificity for item in core] == [
        "author_distinctive",
        "author_distinctive",
        "field_conventional",
    ]


def test_selector_refuses_to_fabricate_three_models() -> None:
    profiles = _profiles(2, docs=(1, 2))
    with pytest.raises(ValueError, match="不足 3 个"):
        select_academic_candidates(
            clusters=[_cluster(index, (1, 2)) for index in range(2)],
            target_profiles=profiles,
            target_doc_ids=[profile.doc_id for profile in profiles],
            holdout_doc_ids=[],
            control_doc_ids=[],
            domain="",
            generative=[],
            exclusivity=[],
        )


def test_failed_distinctive_candidate_is_downgraded_to_heuristic() -> None:
    profiles = _profiles(4)
    clusters = [_cluster(index, (1, 2, 3)) for index in range(4)]
    exclusivity = [
        ExclusivityAssessment(
            candidate_id=f"candidate_{index}",
            specificity="author_distinctive",
            rationale="目标语料中的稳定程度高于同领域对照语料。",
        )
        for index in range(4)
    ]
    failed = CandidateAssessment(
        candidate_id="candidate_3",
        status="failed",
        rationale="留出论文没有复现该写作操作。",
    )

    registry = select_academic_candidates(
        clusters=clusters,
        target_profiles=profiles,
        target_doc_ids=[profile.doc_id for profile in profiles],
        holdout_doc_ids=["doc_3"],
        control_doc_ids=["control_1"],
        domain="出版学",
        generative=[failed],
        exclusivity=exclusivity,
    )

    downgraded = next(
        item for item in registry.records if item.candidate.candidate_id == "candidate_3"
    )
    assert downgraded.selected_as == "heuristic"

    messages = academic_supplement_messages(
        candidate_bundle={"mental_candidates": []},
        expression=ExpressionAnalyzer().analyze(["先提出问题，再组织证据。"]),
        source_info=tuple(
            SourceInfo(
                doc_id=profile.doc_id,
                title=profile.doc_id,
                filename=f"{profile.doc_id}.pdf",
                chunk_count=1,
            )
            for profile in profiles
        ),
        output_language="zh-CN",
        academic_registry=registry,
    )
    request = json.loads(messages[1]["content"])
    assert [
        item["candidate"]["candidate_id"] for item in request["downgraded_model_candidates"]
    ] == ["candidate_3"]


def test_runtime_persona_contains_no_evidence_or_source_anchors() -> None:
    evidence = [
        PersonaEvidence(
            evidence_id=f"ev_{index}",
            chunk_id=f"chunk_{index}",
            doc_id=f"doc_{index}",
            domain="出版学",
            summary="这里只用于证明作者模型。",
            confidence="high",
        )
        for index in range(2)
    ]
    models = [
        MentalModel(
            name=f"模型{index}",
            description="优先建立问题层次再安排证据。",
            cross_domain_evidence=evidence,
            applicability="学术论证。",
            limits="不替代当前任务事实。",
            validation=TripleValidation(
                cross_domain=True,
                generative=True,
                exclusive=True,
                generative_rationale="可迁移到新问题。",
                exclusivity_rationale="具有作者区分度。",
            ),
        )
        for index in range(3)
    ]
    statistics = ExpressionAnalyzer().analyze(["我们需要先提出问题，然后组织证据。"])
    persona = PersonaSpec(
        id="persona_test",
        name="测试作者",
        mode="person",
        mental_models=models,
        expression_dna=ExpressionDNA(
            sentence_fingerprint=statistics.fingerprint,
            style_tags=StyleTags(),
        ),
        evidence_registry=evidence,
        source_info=[
            SourceInfo(
                doc_id="doc_0",
                title="不应进入运行时的旧论文",
                filename="old.pdf",
                chunk_count=1,
            )
        ],
        research_date=date.today(),
        declared_limits=["边界一", "边界二", "边界三"],
    )

    payload = json.dumps(build_runtime_persona(persona).model_dump(mode="json"), ensure_ascii=False)

    assert "evidence_id" not in payload
    assert "chunk_id" not in payload
    assert "不应进入运行时的旧论文" not in payload
    assert "这里只用于证明作者模型" not in payload


def test_repeated_distillation_is_one_profile_with_version_history(tmp_path) -> None:
    database = Database(tmp_path / "versions.db")
    database.initialize()
    KnowledgeBaseRepository(database).ensure_default()
    repository = PersonaRepository(database)
    first = repository.begin_or_resume(
        name="叶芃",
        mode="person",
        kb_id="kb_default",
        source_hash="source_v1",
        input_hash="input_v1",
        source_doc_ids=["doc_1"],
        map_total=1,
    )
    repository.mark_failed(first.run_id, first.persona_id, "TestFailure")
    second = repository.begin_or_resume(
        name="叶芃",
        mode="person",
        kb_id="kb_default",
        source_hash="source_v2",
        input_hash="input_v2",
        source_doc_ids=["doc_1", "doc_2"],
        map_total=2,
    )

    profiles = repository.list_personas("kb_default")
    versions = repository.list_versions(second.persona_id)

    assert len(profiles) == 1
    assert profiles[0]["version_count"] == 2
    assert [item["version_number"] for item in versions] == [2, 1]


def test_generation_sources_exclude_persona_corpus_unless_explicitly_allowed() -> None:
    evidence = [
        PersonaEvidence(
            evidence_id=f"ev_{index}",
            chunk_id=f"chunk_{index}",
            doc_id=f"doc_{index}",
            domain="出版学",
            summary="作者模型证据。",
            confidence="high",
        )
        for index in range(2)
    ]
    statistics = ExpressionAnalyzer().analyze(["先提出问题，再组织证据。"])
    model = MentalModel(
        name="问题驱动",
        description="先提出问题，再组织证据。",
        cross_domain_evidence=evidence,
        applicability="学术写作。",
        limits="不提供事实。",
        validation=TripleValidation(
            cross_domain=True,
            generative=True,
            exclusive=True,
            generative_rationale="可迁移。",
            exclusivity_rationale="有区分度。",
        ),
    )
    persona = PersonaSpec(
        id="persona_policy",
        name="测试作者",
        mode="person",
        mental_models=[
            model,
            model.model_copy(update={"name": "模型二"}),
            model.model_copy(update={"name": "模型三"}),
        ],
        expression_dna=ExpressionDNA(
            sentence_fingerprint=statistics.fingerprint,
            style_tags=StyleTags(),
        ),
        evidence_registry=evidence,
        source_info=[
            SourceInfo(doc_id="doc_0", title="旧论文", filename="old.pdf", chunk_count=1),
            SourceInfo(
                doc_id="control_doc",
                title="对照论文",
                filename="control.pdf",
                chunk_count=1,
            ),
        ],
        research_date=date.today(),
        declared_limits=["边界一", "边界二", "边界三"],
    )

    default_policy = build_generation_source_policy(
        persona=persona,
        selected_task_doc_ids={"doc_0", "new_doc"},
    )
    explicit_policy = build_generation_source_policy(
        persona=persona,
        selected_task_doc_ids={"doc_0", "control_doc", "new_doc"},
        explicitly_allowed_persona_doc_ids={"doc_0"},
        target_persona_doc_ids={"doc_0"},
    )
    role_aware_policy = build_generation_source_policy(
        persona=persona,
        selected_task_doc_ids={"doc_0", "control_doc", "new_doc"},
        target_persona_doc_ids={"doc_0"},
    )

    assert default_policy.allowed_task_doc_ids == {"new_doc"}
    assert role_aware_policy.allowed_task_doc_ids == {"control_doc", "new_doc"}
    assert explicit_policy.allowed_task_doc_ids == {"doc_0", "control_doc", "new_doc"}


def test_similarity_guard_detects_long_copy_from_persona_source() -> None:
    copied = "这一研究路径强调从媒介制度变迁解释出版主体之间的关系"

    matches = find_suspicious_source_overlap(
        f"新稿开头。{copied}，并据此展开讨论。",
        [f"旧论文中写道：{copied}。这是后续内容。"],
        minimum_characters=20,
    )

    assert matches


class _AcademicFakeClient:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def chat(self, messages, **_kwargs):
        request = json.loads(messages[1]["content"])
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.01)
        try:
            task = request["task"]
            if task.startswith("归并同一篇文档"):
                result = self._paper(request)
            elif task.startswith("列出并聚类"):
                result = self._clusters(request)
            elif task.startswith("逐个检验"):
                result = self._generative(request)
            else:
                raise AssertionError(task)
            return ChatResult(content=json.dumps(result, ensure_ascii=False), model="fake")
        finally:
            with self.lock:
                self.active -= 1

    @staticmethod
    def _paper(request):
        candidates = []
        for item in request["map_candidates"]:
            candidates.append(
                {
                    "paper_candidate_id": item["map_candidate_id"],
                    "map_candidate_ids": [item["map_candidate_id"]],
                    "operation": "argument_structure",
                    "name": item["name"],
                    "description": "通过稳定的问题层次组织证据并推进学术论证。",
                    "evidence_ids": item["evidence_ids"],
                    "applicability": "适用于需要解释多层关系的学术问题。",
                    "limits": "材料不足或问题单一时不能机械套用。",
                    "research_context": "该论文通过分层问题组织材料和推论。",
                }
            )
        return {"doc_id": request["doc_id"], "candidates": candidates}

    @staticmethod
    def _clusters(request):
        grouped: dict[str, list[dict[str, object]]] = {}
        for profile in request["paper_profiles"]:
            for item in profile["candidates"]:
                grouped.setdefault(item["name"], []).append(item)
        return {
            "candidates": [
                {
                    "candidate_id": members[0]["paper_candidate_id"],
                    "operation": "argument_structure",
                    "name": name,
                    "description": "该作者反复通过问题层次安排证据和推论。",
                    "paper_candidate_ids": [item["paper_candidate_id"] for item in members],
                    "evidence_ids": [item["evidence_ids"][0] for item in members],
                    "applicability": "需要解释复杂关系的学术论证。",
                    "limits": "简单事实说明中不需要多层展开。",
                    "attribution_scope": "author_specific",
                    "attribution_rationale": "该操作在多篇目标论文中稳定复现。",
                }
                for name, members in grouped.items()
            ]
        }

    @staticmethod
    def _generative(request):
        holdout = request["holdout_paper_profiles"][0]["candidates"]
        by_name = {item["name"]: item for item in holdout}
        return {
            "assessments": [
                {
                    "candidate_id": item["candidate_id"],
                    "status": "passed",
                    "rationale": "留出论文独立呈现了相同的问题分层和论证推进操作。",
                    "matched_paper_candidate_ids": [by_name[item["name"]]["paper_candidate_id"]],
                }
                for item in request["candidates"]
            ]
        }


def test_academic_engine_runs_paper_consolidation_and_neutral_holdout(tmp_path) -> None:
    database = Database(tmp_path / "academic_engine.db")
    database.initialize()
    KnowledgeBaseRepository(database).ensure_default()
    repository = PersonaRepository(database)
    run = repository.begin_or_resume(
        name="测试作者",
        mode="person",
        kb_id="kb_default",
        source_hash="target_hash",
        input_hash="input_hash",
        source_doc_ids=[f"doc_{index}" for index in range(4)],
        map_total=4,
    )
    evidence_registry = []
    mental_candidates = []
    for doc in range(4):
        for candidate in range(3):
            evidence_id = f"ev_{doc}_{candidate}"
            evidence_registry.append(
                {
                    "evidence_id": evidence_id,
                    "chunk_id": f"chunk_{doc}_{candidate}",
                    "doc_id": f"doc_{doc}",
                    "domain": "出版学",
                    "summary": "该片段展示了问题分层和论证组织方式。",
                    "confidence": "high",
                }
            )
            mental_candidates.append(
                {
                    "map_candidate_id": f"map_{doc}_{candidate}",
                    "unit_id": f"unit_{doc}",
                    "name": f"模型{candidate}",
                    "description": "通过问题层次组织证据和推论。",
                    "evidence_ids": [evidence_id],
                    "source_doc_ids": [f"doc_{doc}"],
                    "generative_rationale": "能够迁移到新的研究问题。",
                    "exclusivity_rationale": "需要后续对照语料验证。",
                }
            )
    source_info = tuple(
        SourceInfo(
            doc_id=f"doc_{index}",
            title=f"测试论文{index}",
            filename=f"{index}.pdf",
            chunk_count=1,
        )
        for index in range(4)
    )
    client = _AcademicFakeClient()
    engine = AcademicDistillationEngine(client, repository, parallelism=lambda: 3)

    registry = engine.build_registry(
        run_id=run.run_id,
        target_label="测试作者",
        domain="出版学",
        target_bundle={
            "mental_candidates": mental_candidates,
            "evidence_registry": evidence_registry,
        },
        target_source_info=source_info,
        target_hash="target_hash",
        control_bundle=None,
        control_source_info=(),
        control_hash=None,
        progress=lambda _percent, _message: None,
        check_cancelled=lambda: None,
    )

    assert len([item for item in registry.records if item.selected_as == "core"]) == 3
    assert len(registry.holdout_doc_ids) == 1
    assert all(item.validation.generative_status == "passed" for item in registry.records)
    assert client.peak == 3


class _SupplementFakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, _messages, **kwargs):
        self.calls.append(kwargs)
        result = ChatResult(
            content=json.dumps(
                {
                    "decision_heuristics": [],
                    "style_tags": {},
                    "declared_limits": [
                        "不能代替当前任务的事实证据。",
                        "不能推断作者未公开的真实想法。",
                        "只能反映当前语料范围内的写作操作。",
                    ],
                },
                ensure_ascii=False,
            ),
            model="fake",
        )
        validator = kwargs.get("result_validator")
        if callable(validator):
            validator(result)
        return result


def test_academic_supplement_uses_direct_non_thinking_assembly() -> None:
    units: list[SourceUnit] = []
    maps: list[MapResult] = []
    source_info: list[SourceInfo] = []
    for index in range(3):
        doc_id = f"doc_{index}"
        unit_id = f"unit_{index}"
        chunk_id = f"chunk_{index}"
        units.append(
            SourceUnit(
                unit_id=unit_id,
                segments=[
                    SourceSegment(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        document_title=f"论文{index}",
                        filename=f"{index}.pdf",
                        text="先界定问题，再组织证据并说明适用边界。",
                    )
                ],
            )
        )
        maps.append(
            MapResult(
                unit_id=unit_id,
                mental_candidates=[
                    MapMentalCandidate(
                        name=f"局部候选{index}",
                        description="先界定问题，再组织证据。",
                        evidence=[
                            ExtractedEvidence(
                                chunk_id=chunk_id,
                                domain="出版学",
                                summary="该片段展示了问题界定与证据组织。",
                            )
                        ],
                        generative_rationale="能够迁移到新的研究问题。",
                        exclusivity_rationale="需要结合对照语料判断。",
                    )
                ],
            )
        )
        source_info.append(
            SourceInfo(
                doc_id=doc_id,
                title=f"论文{index}",
                filename=f"{index}.pdf",
                domain="出版学",
                chunk_count=1,
            )
        )
    evidence, _gaps, _bundle = CandidateBundleBuilder().build(maps, tuple(units))
    evidence_ids = list(evidence)
    validation = AcademicModelValidation(
        supporting_doc_ids=["doc_0", "doc_1"],
        recurrence_document_count=2,
        recurrence_level="basic",
        generative_status="passed",
        generative_rationale="留出论文复现了相同操作。",
        specificity="author_distinctive",
        exclusivity_rationale="对照语料中没有同等稳定的操作。",
        control_corpus_used=True,
    )

    def record(index: int, selected_as: str, *, failed: bool = False) -> CandidateRecord:
        item_validation = (
            validation.model_copy(
                update={
                    "generative_status": "failed",
                    "generative_rationale": "留出论文未复现该操作。",
                }
            )
            if failed
            else validation
        )
        return CandidateRecord(
            candidate=CandidateCluster(
                candidate_id=f"candidate_{index}",
                operation="argument_structure",
                name=f"模型{index}",
                description="先界定问题层次，再安排证据与推论。",
                paper_candidate_ids=[f"paper_0_{index}", f"paper_1_{index}"],
                evidence_ids=evidence_ids[:2],
                applicability="需要解释复杂关系的学术论证。",
                limits="简单事实说明中不必使用。",
                attribution_scope="author_specific",
                attribution_rationale="在不同论文中反复出现。",
            ),
            validation=item_validation,
            selected_as=selected_as,
            selection_rank=index + 1 if selected_as == "core" else None,
        )

    registry = CandidateRegistry(
        target_doc_ids=[item.doc_id for item in source_info],
        control_doc_ids=["control_1"],
        domain="出版学",
        records=[
            record(0, "core"),
            record(1, "core"),
            record(2, "core"),
            record(3, "heuristic", failed=True),
        ],
    )
    client = _SupplementFakeClient()
    synthesizer = PersonaSynthesizer(client)

    persona = synthesizer.synthesize(
        persona_id="persona_academic",
        name="测试作者",
        mode="person",
        map_results=maps,
        units=tuple(units),
        source_info=tuple(source_info),
        expression=ExpressionAnalyzer().analyze([unit.segments[0].text for unit in units]),
        research_date=date.today(),
        academic_registry=registry,
    )

    assert client.calls[0]["thinking"] is False
    assert client.calls[0]["stream"] is False
    assert callable(client.calls[0]["result_validator"])
    assert client.calls[0]["use_cache"] is True
    assert client.calls[0]["report_stream_error"] is False
    assert [item.candidate_id for item in persona.mental_models] == [
        "candidate_0",
        "candidate_1",
        "candidate_2",
    ]
    assert persona.decision_heuristics[0].rule.startswith("采用“模型3”")
