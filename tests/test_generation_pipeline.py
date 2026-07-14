"""Offline end-to-end coverage for the Stage 4-6 writing graph."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from writing_factory.generate.framework import FrameworkOutputError, _validate_outline_budget
from writing_factory.generate.models import (
    AnnotatedOutline,
    Claim,
    EvidenceItem,
    EvidencePack,
    GenerationContext,
    GenerationOptions,
    OutlineNode,
    ReferenceList,
    SectionDraft,
    StructureReview,
    TermConsistencyReport,
    ThesisStatement,
    VerifiedClaim,
    VerifiedDraft,
)
from writing_factory.generate.prompts import drafting_messages, verification_messages
from writing_factory.generate.verification import verify_section
from writing_factory.kb.models import Bibliography, FusedHit, RetrievalResult
from writing_factory.llm.models import ChatResult
from writing_factory.orchestration.errors import PipelineNodeError
from writing_factory.orchestration.nodes import (
    WritingPipeline,
    prepare_revise_section,
    should_continue_after_verify,
)
from writing_factory.orchestration.pipeline_runner import (
    _checkpoint_progress,
    _legacy_resume_config,
    _verification_recovery_config,
    run_writing_pipeline_with_progress,
)
from writing_factory.orchestration.reference_assembler import assemble_reference_list


class FakeTaskContext:
    def __init__(self) -> None:
        self.progress: list[tuple[int, str]] = []

    def report_progress(self, percent: int, message: str) -> None:
        self.progress.append((percent, message))

    def check_cancelled(self) -> None:
        return None

    @property
    def is_cancelled(self) -> bool:
        return False


class FakePersonaRepository:
    def load_ready(self, _persona_id: str):
        return SimpleNamespace(source_info=[]), ""

    def load_runtime(self, persona_id: str):
        return SimpleNamespace(
            name="测试作者",
            model_dump=lambda **_kwargs: {
                "persona_id": persona_id,
                "name": "测试作者",
                "mental_models": [],
                "expression_dna": {},
            },
        )


class FakeKnowledgeRepository:
    def ready_child_chunks_by_ids(self, _kb_id: str, _chunk_ids: set[str]):
        return []

    def get_bibliographies(self, _kb_id: str, doc_ids: set[str]):
        return {
            doc_id: Bibliography(
                author="张三",
                title="数字人文研究",
                year=2024,
                publisher_or_journal="出版研究",
                document_type="article",
                extra={"issue": "2", "pages": "10-20"},
            )
            for doc_id in doc_ids
        }


class FakeRetriever:
    def __init__(self, repository: FakeKnowledgeRepository) -> None:
        self.repository = repository
        self.requests = []

    def search(self, request, **_kwargs):
        self.requests.append(request)
        return RetrievalResult(
            query=request.query,
            hits=(
                FusedHit(
                    chunk_id="chunk_1",
                    doc_id="doc_task",
                    text="数字人文方法能够扩展传统文献研究的证据处理能力。",
                    source="hybrid",
                    final_rank=1,
                    rrf_score=0.9,
                    page_start=3,
                    page_end=3,
                    section_heading="研究方法",
                    matched_child_ids=("chunk_1",),
                ),
            ),
        )


class FakeSiliconFlow:
    def __init__(self) -> None:
        self.responses = [
            {
                "thesis_text": "数字人文方法扩展了传统文献研究的证据能力。",
                "angle": "证据方法转型",
                "kb_support_assessment": "任务语料提供直接支持。",
                "persona_id": "persona_1",
            },
            {
                "thesis": {
                    "thesis_text": "数字人文方法扩展了传统文献研究的证据能力。",
                    "angle": "证据方法转型",
                    "kb_support_assessment": "任务语料提供直接支持。",
                    "persona_id": "persona_1",
                },
                "root_nodes": [
                    {
                        "node_id": "1",
                        "heading": "证据方法的变化",
                        "rhetorical_purpose": "论证核心主张",
                        "candidate_source_keys": ["S1"],
                        "children": [],
                    }
                ],
                "term_registry": {},
                "kb_id": "kb",
            },
            {
                "section_id": "1",
                "heading": "证据方法的变化",
                "paragraphs": ["数字人文方法能够扩展传统文献研究的证据处理能力。[S1]"],
                "claims": [
                    {
                        "claim_id": "1_c1",
                        "text": "数字人文方法能够扩展传统文献研究的证据处理能力。",
                        "claim_type": "fact",
                        "source_keys": ["S1"],
                        "paragraph_index": 0,
                    }
                ],
                "evidence_pack": {
                    "section_id": "1",
                    "items": [
                        {
                            "source_key": "S1",
                            "chunk_id": "forged_chunk",
                            "doc_id": "forged_doc",
                            "verbatim_excerpt": "模型擅自回传的证据不得进入冻结证据包。",
                        }
                    ],
                },
            },
            {
                "section_id": "1",
                "verified_claims": [
                    {
                        "claim_id": "1_c1",
                        "verdict": "supported",
                        "verifier_rationale": "证据原文直接支持。",
                    }
                ]
            },
            "数字人文方法能够扩展传统文献研究的证据处理能力。[S1]",
            {"fact_drift_detected": False},
            {"issues": [], "overall_assessment": "结构清晰。"},
            {
                "sections": [
                    {
                        "section_id": "1",
                        "polished_text": "数字人文方法能够扩展传统文献研究的证据处理能力。[S1]",
                    }
                ],
                "transitions_added": [],
                "global_consistency_notes": "无需调整。",
            },
            {"fact_drift_detected": False},
        ]
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        response = self.responses.pop(0)
        content = (
            response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
        )
        return ChatResult(content=content, model="fake", finish_reason="stop")


def test_full_writing_graph_runs_offline(tmp_path: Path) -> None:
    kb_repository = FakeKnowledgeRepository()
    retriever = FakeRetriever(kb_repository)
    context = FakeTaskContext()
    client = FakeSiliconFlow()
    snapshots: list[dict] = []
    result = run_writing_pipeline_with_progress(
        persona_id="persona_1",
        task_description="讨论数字人文方法",
        domain="数字人文",
        context=context,
        siliconflow=client,
        retriever=retriever,
        persona_repository=FakePersonaRepository(),
        kb_repository=kb_repository,
        checkpoint_dir=tmp_path,
        kb_id="kb",
        task_id="task_1",
        selected_doc_ids={"doc_task"},
        state_callback=snapshots.append,
    )

    assert result["status"] == "done"
    assert result["task_id"] == "task_1"
    assert "数字人文" in result["final_draft_json"]
    final_draft = json.loads(result["final_draft_json"])
    assert "[1]" in final_draft["sections"][0]["polished_text"]
    assert "[S1]" not in final_draft["sections"][0]["polished_text"]
    outline = json.loads(result["outline_json"])
    assert outline["root_nodes"][0]["candidate_evidence"][0]["chunk_id"] == "chunk_1"
    assert outline["root_nodes"][0]["candidate_source_keys"] == ["S1"]
    frozen_pack = EvidencePack.model_validate_json(result["sections"][0]["evidence_pack_json"])
    assert frozen_pack.section_id == "1"
    assert frozen_pack.items[0].chunk_id == "chunk_1"
    assert frozen_pack.items[0].doc_id == "doc_task"
    assert "模型擅自回传" not in frozen_pack.items[0].verbatim_excerpt
    assert result["sections"][0]["source_key_offset"] == 1
    assert len(retriever.requests) == 4
    assert all(request.filters.doc_ids == {"doc_task"} for request in retriever.requests)
    assert context.progress[-1][0] == 100
    assert [percent for percent, _message in context.progress] == sorted(
        percent for percent, _message in context.progress
    )
    assert any(snapshot.get("sections") for snapshot in snapshots)
    assert snapshots[-1]["status"] == "done"
    assert "request_timeout_seconds" not in client.calls[1][1]
    assert "request_total_timeout_seconds" not in client.calls[1][1]
    assert client.calls[1][1]["max_tokens"] == 8192
    assert client.calls[1][1]["stream"] is True

    resumed_client = FakeSiliconFlow()
    resumed_client.responses.clear()
    resumed_context = FakeTaskContext()
    resumed = run_writing_pipeline_with_progress(
        persona_id="persona_1",
        task_description="讨论数字人文方法",
        domain="数字人文",
        context=resumed_context,
        siliconflow=resumed_client,
        retriever=FakeRetriever(kb_repository),
        persona_repository=FakePersonaRepository(),
        kb_repository=kb_repository,
        checkpoint_dir=tmp_path,
        kb_id="kb",
        task_id="task_1",
        selected_doc_ids={"doc_task"},
        resume=True,
    )
    assert resumed["status"] == "done"
    assert resumed_client.calls == []
    assert [percent for percent, _message in resumed_context.progress] == sorted(
        percent for percent, _message in resumed_context.progress
    )
    assert resumed_context.progress[0][1] == "读取写作断点"


def test_evidence_prefetch_is_concurrent_and_keeps_outline_order() -> None:
    class ConcurrentRetriever(FakeRetriever):
        def __init__(self, repository) -> None:
            super().__init__(repository)
            self._lock = threading.Lock()
            self.active = 0
            self.peak = 0

        def search(self, request, **_kwargs):
            with self._lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                time.sleep(0.05)
                section_id = request.query.splitlines()[1]
                return RetrievalResult(
                    query=request.query,
                    hits=(
                        FusedHit(
                            chunk_id=f"chunk_{section_id}",
                            doc_id="doc_task",
                            text=f"{section_id}的证据。",
                            source="hybrid",
                            final_rank=1,
                            rrf_score=0.9,
                        ),
                    ),
                )
            finally:
                with self._lock:
                    self.active -= 1

    thesis = ThesisStatement(
        thesis_text="核心论点",
        angle="测试角度",
        kb_support_assessment="证据充足",
        persona_id="persona_1",
    )
    nodes = [
        OutlineNode(node_id=str(index), heading=f"第{index}节", rhetorical_purpose="论证")
        for index in range(1, 4)
    ]
    outline = AnnotatedOutline(thesis=thesis, root_nodes=nodes, kb_id="kb")
    context = GenerationContext(
        kb_id="kb",
        task_description="测试并发预取",
        persona_id="persona_1",
        allowed_doc_ids=("doc_task",),
    )
    repository = FakeKnowledgeRepository()
    retriever = ConcurrentRetriever(repository)
    pipeline = WritingPipeline(
        persona_repository=FakePersonaRepository(),
        retriever=retriever,
        siliconflow=SimpleNamespace(),
        kb_repository=repository,
    )
    state = {
        "context_json": context.model_dump_json(),
        "thesis_json": thesis.model_dump_json(),
        "outline_json": outline.model_dump_json(),
        "source_key_counter": 4,
        "sections": [
            {"section_id": node.node_id, "heading": node.heading, "source_key_offset": 0}
            for node in nodes
        ],
    }

    result = pipeline.prefetch_evidence_node(state)

    assert retriever.peak >= 2
    assert [section["heading"] for section in result["sections"]] == [
        "第1节",
        "第2节",
        "第3节",
    ]
    assert [section["source_key_offset"] for section in result["sections"]] == [4, 12, 20]
    packs = [
        EvidencePack.model_validate_json(section["evidence_pack_json"])
        for section in result["sections"]
    ]
    assert [pack.items[0].source_key for pack in packs] == ["S5", "S13", "S21"]
    assert result["source_key_counter"] == 28


def test_outline_budget_rejects_too_many_body_units_for_short_article() -> None:
    thesis = ThesisStatement(
        thesis_text="核心论点",
        angle="测试角度",
        kb_support_assessment="证据充足",
        persona_id="persona_1",
    )
    outline = AnnotatedOutline(
        thesis=thesis,
        root_nodes=[
            OutlineNode(
                node_id=str(index),
                heading=f"第{index}节",
                rhetorical_purpose="论证",
            )
            for index in range(1, 7)
        ],
        kb_id="kb",
    )

    with pytest.raises(FrameworkOutputError, match="当前提纲有 6 个"):
        _validate_outline_budget(outline, target_length_chars=1500)


def test_parent_outline_nodes_are_containers_not_duplicate_body_units(monkeypatch) -> None:
    thesis = ThesisStatement(
        thesis_text="核心论点",
        angle="测试角度",
        kb_support_assessment="证据充足",
        persona_id="persona_1",
    )
    outline = AnnotatedOutline(
        thesis=thesis,
        root_nodes=[
            OutlineNode(
                node_id="1",
                heading="父级标题",
                rhetorical_purpose="组织层级",
                children=[
                    OutlineNode(node_id="1.1", heading="子节甲", rhetorical_purpose="论证甲"),
                    OutlineNode(node_id="1.2", heading="子节乙", rhetorical_purpose="论证乙"),
                ],
            ),
            OutlineNode(node_id="2", heading="独立结论", rhetorical_purpose="总结"),
        ],
        kb_id="kb",
    )
    monkeypatch.setattr(
        "writing_factory.generate.framework.build_framework",
        lambda **_kwargs: outline,
    )
    context = GenerationContext(
        kb_id="kb",
        task_description="1500字短文",
        persona_id="persona_1",
        generation_options=GenerationOptions(target_length_chars=1500),
    )
    pipeline = WritingPipeline(
        persona_repository=FakePersonaRepository(),
        retriever=SimpleNamespace(),
        siliconflow=SimpleNamespace(),
        kb_repository=FakeKnowledgeRepository(),
    )

    result = pipeline.build_framework_node(
        {
            "context_json": context.model_dump_json(),
            "thesis_json": thesis.model_dump_json(),
        }
    )

    assert [section["section_id"] for section in result["sections"]] == ["1.1", "1.2", "2"]
    assert [section["target_length_chars"] for section in result["sections"]] == [500] * 3


def test_revision_recovers_frozen_evidence_and_strips_prior_source_keys(monkeypatch) -> None:
    thesis = ThesisStatement(
        thesis_text="核心论点",
        angle="测试角度",
        kb_support_assessment="证据充足",
        persona_id="persona_1",
    )
    outline = AnnotatedOutline(
        thesis=thesis,
        root_nodes=[OutlineNode(node_id="3", heading="第三节", rhetorical_purpose="论证")],
        kb_id="kb",
    )
    frozen = EvidencePack(
        section_id="3",
        items=[
            EvidenceItem(
                source_key="S96",
                chunk_id="chunk_96",
                doc_id="doc_task",
                verbatim_excerpt="冻结证据",
            )
        ],
    )
    old_draft = SectionDraft(
        section_id="3",
        heading="第三节",
        paragraphs=["冻结事实。[S96]"],
        claims=[
            Claim(
                claim_id="c1",
                text="冻结事实。",
                claim_type="fact",
                source_keys=["S96"],
                paragraph_index=0,
            )
        ],
        evidence_pack=frozen,
    )
    failed_verification = VerifiedDraft(
        section_id="3",
        verified_claims=[
            VerifiedClaim(
                claim=old_draft.claims[0],
                verdict="unsupported",
                verifier_rationale="S96 没有支持这项事实。",
            )
        ],
        unsupported_count=1,
    )
    captured: dict[str, object] = {}

    def fake_draft_section(**kwargs):
        captured.update(kwargs)
        return old_draft

    monkeypatch.setattr(
        "writing_factory.generate.drafting.draft_section",
        fake_draft_section,
    )
    context = GenerationContext(
        kb_id="kb",
        task_description="测试旧断点",
        persona_id="persona_1",
    )
    pipeline = WritingPipeline(
        persona_repository=FakePersonaRepository(),
        retriever=SimpleNamespace(),
        siliconflow=SimpleNamespace(),
        kb_repository=FakeKnowledgeRepository(),
    )
    result = pipeline.draft_section_node(
        {
            "context_json": context.model_dump_json(),
            "thesis_json": thesis.model_dump_json(),
            "outline_json": outline.model_dump_json(),
            "term_registry_json": "{}",
            "sections": [
                {
                    "section_id": "3",
                    "heading": "第三节",
                    "status": "revising",
                    "revision_count": 1,
                    "source_key_offset": 92,
                    "draft_json": old_draft.model_dump_json(),
                    "verified_draft_json": failed_verification.model_dump_json(),
                }
            ],
            "current_section_index": 0,
            "source_key_counter": 100,
            "claims_made_json": '["前文事实。[S85]"]',
        }
    )

    assert captured["evidence_pack"] == frozen
    assert captured["prior_claims"] == ["前文事实。"]
    assert captured["revision_feedback"] == [
        {
            "claim_id": "c1",
            "claim_text": "冻结事实。",
            "claim_type": "fact",
            "source_keys": ["S96"],
            "verdict": "unsupported",
            "rationale": "S96 没有支持这项事实。",
            "required_action": "删除该事实论断，或缩写到冻结证据明确支持的范围；不得只更换引用键。",
        }
    ]
    assert EvidencePack.model_validate_json(
        result["sections"][0]["evidence_pack_json"]
    ) == frozen


def test_quality_presets_reduce_calls_without_mislabeling_fast_draft(tmp_path: Path) -> None:
    repository = FakeKnowledgeRepository()
    fast_client = FakeSiliconFlow()
    fast = run_writing_pipeline_with_progress(
        persona_id="persona_1",
        task_description="讨论数字人文方法",
        domain="数字人文",
        context=FakeTaskContext(),
        siliconflow=fast_client,
        retriever=FakeRetriever(repository),
        persona_repository=FakePersonaRepository(),
        kb_repository=repository,
        checkpoint_dir=tmp_path / "fast",
        kb_id="kb",
        task_id="task_fast",
        selected_doc_ids={"doc_task"},
        generation_options=GenerationOptions.from_preset(
            "fast_draft",
            target_length_chars=1500,
        ),
    )
    balanced_client = FakeSiliconFlow()
    balanced = run_writing_pipeline_with_progress(
        persona_id="persona_1",
        task_description="讨论数字人文方法",
        domain="数字人文",
        context=FakeTaskContext(),
        siliconflow=balanced_client,
        retriever=FakeRetriever(repository),
        persona_repository=FakePersonaRepository(),
        kb_repository=repository,
        checkpoint_dir=tmp_path / "balanced",
        kb_id="kb",
        task_id="task_balanced",
        selected_doc_ids={"doc_task"},
        generation_options=GenerationOptions.from_preset(
            "balanced",
            target_length_chars=1500,
        ),
    )

    assert len(fast_client.calls) == 3
    assert json.loads(fast["final_draft_json"])["quality_status"] == "unverified_draft"
    assert len(balanced_client.calls) == 4
    assert json.loads(balanced["final_draft_json"])["quality_status"] == "verified_final"


def test_checkpoint_progress_uses_real_sections() -> None:
    state = {
        "sections": [
            {"status": "polished"},
            {"status": "verified"},
            {"status": "pending"},
        ]
    }
    assert _checkpoint_progress("verify_section", state) == 52
    assert _checkpoint_progress("parallel_reviews", state) == 88


def test_term_and_structure_reviews_run_in_parallel(monkeypatch) -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    def enter(result):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.05)
            return result
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(
        "writing_factory.orchestration.nodes.review_term_consistency",
        lambda **_kwargs: enter(TermConsistencyReport(reviewer_note="术语完成")),
    )
    monkeypatch.setattr(
        "writing_factory.orchestration.nodes.review_structure",
        lambda **_kwargs: enter(StructureReview(overall_assessment="结构完成")),
    )
    thesis = ThesisStatement(
        thesis_text="核心论点",
        angle="测试角度",
        kb_support_assessment="证据充足",
        persona_id="persona_1",
    )
    outline = AnnotatedOutline(
        thesis=thesis,
        root_nodes=[OutlineNode(node_id="1", heading="第一节", rhetorical_purpose="论证")],
        term_registry={"术语": "定义"},
        kb_id="kb",
    )
    pipeline = WritingPipeline(
        persona_repository=FakePersonaRepository(),
        retriever=SimpleNamespace(),
        siliconflow=SimpleNamespace(),
        kb_repository=FakeKnowledgeRepository(),
    )

    result = pipeline.parallel_reviews_node(
        {
            "thesis_json": thesis.model_dump_json(),
            "outline_json": outline.model_dump_json(),
            "term_registry_json": json.dumps(outline.term_registry, ensure_ascii=False),
            "sections": [],
        }
    )

    assert peak == 2
    assert TermConsistencyReport.model_validate_json(
        result["term_consistency_json"]
    ).reviewer_note == "术语完成"
    assert StructureReview.model_validate_json(
        result["structure_review_json"]
    ).overall_assessment == "结构完成"


def test_framework_regenerates_with_doubled_output_limits(tmp_path: Path) -> None:
    class TruncatedFrameworkClient(FakeSiliconFlow):
        def chat(self, messages, **kwargs):
            if len(self.calls) == 1:
                self.calls.append((messages, kwargs))
                return ChatResult(
                    content='{"thesis": {"thesis_text": "被截断',
                    model="fake",
                    finish_reason="length",
                )
            if len(self.calls) == 2:
                self.calls.append((messages, kwargs))
                return ChatResult(
                    content='{"thesis": {"thesis_text": "仍然截断',
                    model="fake",
                    finish_reason="stop",
                )
            return super().chat(messages, **kwargs)

    kb_repository = FakeKnowledgeRepository()
    client = TruncatedFrameworkClient()
    result = run_writing_pipeline_with_progress(
        persona_id="persona_1",
        task_description="讨论数字人文方法",
        domain="数字人文",
        context=FakeTaskContext(),
        siliconflow=client,
        retriever=FakeRetriever(kb_repository),
        persona_repository=FakePersonaRepository(),
        kb_repository=kb_repository,
        checkpoint_dir=tmp_path,
        kb_id="kb",
        task_id="task_framework_regeneration",
        selected_doc_ids={"doc_task"},
    )

    assert result["status"] == "done"
    assert [client.calls[index][1]["max_tokens"] for index in (1, 2, 3)] == [
        8192,
        16384,
        32768,
    ]
    assert all(
        "request_total_timeout_seconds" not in client.calls[index][1]
        for index in (1, 2, 3)
    )
    assert "从头重新生成" in client.calls[2][0][-1]["content"]


def test_framework_failure_stops_and_resume_retries_failed_node(tmp_path: Path) -> None:
    class FrameworkFailingClient(FakeSiliconFlow):
        def chat(self, messages, **kwargs):
            if len(self.calls) == 1:
                self.calls.append((messages, kwargs))
                raise TimeoutError("planned timeout")
            return super().chat(messages, **kwargs)

    kb_repository = FakeKnowledgeRepository()
    failed_context = FakeTaskContext()
    with pytest.raises(PipelineNodeError, match="框架生成失败.*planned timeout"):
        run_writing_pipeline_with_progress(
            persona_id="persona_1",
            task_description="讨论数字人文方法",
            domain="数字人文",
            context=failed_context,
            siliconflow=FrameworkFailingClient(),
            retriever=FakeRetriever(kb_repository),
            persona_repository=FakePersonaRepository(),
            kb_repository=kb_repository,
            checkpoint_dir=tmp_path,
            kb_id="kb",
            task_id="task_resume_framework",
            selected_doc_ids={"doc_task"},
        )

    assert failed_context.progress[-1][0] < 100
    assert "框架生成失败" in failed_context.progress[-1][1]

    resumed_client = FakeSiliconFlow()
    resumed_client.responses = resumed_client.responses[1:]
    resumed = run_writing_pipeline_with_progress(
        persona_id="persona_1",
        task_description="讨论数字人文方法",
        domain="数字人文",
        context=FakeTaskContext(),
        siliconflow=resumed_client,
        retriever=FakeRetriever(kb_repository),
        persona_repository=FakePersonaRepository(),
        kb_repository=kb_repository,
        checkpoint_dir=tmp_path,
        kb_id="kb",
        task_id="task_resume_framework",
        selected_doc_ids={"doc_task"},
        resume=True,
    )

    assert resumed["status"] == "done"
    assert len(resumed_client.calls) == 8


def test_legacy_terminal_error_rewinds_to_latest_healthy_pending_checkpoint() -> None:
    terminal = SimpleNamespace(
        values={"status": "error"},
        next=(),
        config={"configurable": {"checkpoint_id": "terminal"}},
        metadata={"step": 5},
    )
    still_error = SimpleNamespace(
        values={"status": "error"},
        next=("draft_section",),
        config={"configurable": {"checkpoint_id": "error"}},
        metadata={"step": 2},
    )
    healthy = SimpleNamespace(
        values={"status": "framework_building"},
        next=("build_framework",),
        config={"configurable": {"checkpoint_id": "healthy"}},
        metadata={"step": 1},
    )
    graph = SimpleNamespace(
        get_state=lambda _config: terminal,
        get_state_history=lambda _config: [terminal, still_error, healthy],
    )
    thread_config = {"configurable": {"thread_id": "legacy"}}

    selected = _legacy_resume_config(graph, thread_config)

    assert selected == healthy.config


def test_pending_verification_gate_becomes_bounded_recovery_revision() -> None:
    claim = Claim(
        claim_id="sec3_clm3",
        text="没有证据支持的事实。[S95]",
        claim_type="fact",
        source_keys=["S95"],
        paragraph_index=0,
    )
    verified = VerifiedDraft(
        section_id="3",
        verified_claims=[
            VerifiedClaim(
                claim=claim,
                verdict="unsupported",
                verifier_rationale="S95 与该事实无关。",
            )
        ],
        unsupported_count=1,
    )
    state = {
        "status": "polishing",
        "current_section_index": 0,
        "sections": [
            {
                "section_id": "3",
                "status": "verified",
                "revision_count": 3,
                "verified_draft_json": verified.model_dump_json(),
            }
        ],
    }
    captured: dict[str, object] = {}

    class RecoveryGraph:
        def get_state(self, _config):
            return SimpleNamespace(values=state, next=("fail_verification",))

        def update_state(self, config, values, *, as_node):
            captured.update(config=config, values=values, as_node=as_node)
            return {"configurable": {"thread_id": "task", "checkpoint_id": "recovery"}}

    config = {"configurable": {"thread_id": "task"}}

    selected, recovered = _verification_recovery_config(RecoveryGraph(), config)

    assert recovered
    assert selected["configurable"]["checkpoint_id"] == "recovery"
    assert captured["as_node"] == "prepare_revise_section"
    values = captured["values"]
    assert values["status"] == "drafting"
    assert values["sections"][0]["revision_count"] == 3
    assert values["sections"][0]["recovery_revision_count"] == 1
    assert values["sections"][0]["status"] == "revising"


def test_recovery_revision_has_separate_two_attempt_limit() -> None:
    claim = Claim(
        claim_id="c1",
        text="事实。[S1]",
        claim_type="fact",
        source_keys=["S1"],
        paragraph_index=0,
    )
    verified = VerifiedDraft(
        section_id="3",
        verified_claims=[
            VerifiedClaim(
                claim=claim,
                verdict="unsupported",
                verifier_rationale="证据不支持。",
            )
        ],
        unsupported_count=1,
    )
    section = {
        "section_id": "3",
        "status": "verified",
        "revision_count": 3,
        "recovery_revision_count": 1,
        "verified_draft_json": verified.model_dump_json(),
    }
    state = {"current_section_index": 0, "sections": [section]}

    assert should_continue_after_verify(state) == "revise"
    prepared = prepare_revise_section(state)
    assert prepared["sections"][0]["revision_count"] == 3
    assert prepared["sections"][0]["recovery_revision_count"] == 2

    exhausted = {**state, "sections": [{**section, "recovery_revision_count": 2}]}
    assert should_continue_after_verify(exhausted) == "error"

    graph = SimpleNamespace(
        get_state=lambda _config: SimpleNamespace(
            values=exhausted,
            next=("fail_verification",),
        )
    )
    with pytest.raises(PipelineNodeError, match="额外恢复修订已用尽.*c1.*证据不支持"):
        _verification_recovery_config(
            graph,
            {"configurable": {"thread_id": "task"}},
        )


def test_fact_claim_requires_source_and_inline_marker() -> None:
    with pytest.raises(ValueError, match="至少一个 source_key"):
        Claim(
            claim_id="c1",
            text="事实",
            claim_type="fact",
            paragraph_index=0,
        )


def test_drafting_response_schema_excludes_frozen_evidence_pack() -> None:
    thesis = ThesisStatement(
        thesis_text="核心论点",
        angle="测试角度",
        kb_support_assessment="证据充足",
        persona_id="persona_1",
    )
    node = OutlineNode(node_id="1", heading="第一节", rhetorical_purpose="提出问题")
    evidence = EvidencePack(
        section_id="1",
        items=[
            EvidenceItem(
                source_key="S1",
                chunk_id="chunk_1",
                doc_id="doc_1",
                verbatim_excerpt="冻结证据原文",
            )
        ],
    )

    messages = drafting_messages(
        persona_spec_json={"name": "测试作者"},
        thesis=thesis,
        outline_node=node,
        evidence_pack=evidence,
        term_registry={},
    )
    request_text = messages[-1]["content"].split("任务要求_JSON\n", 1)[1].split(
        "\n来源数据_JSON_开始",
        1,
    )[0]
    request = json.loads(request_text)

    assert "evidence_pack" not in request["response_schema"]["properties"]
    assert "冻结证据原文" in messages[-1]["content"]


def test_verification_contract_is_flat_and_accepts_legacy_nested_response() -> None:
    draft = SectionDraft(
        section_id="1",
        heading="第一节",
        paragraphs=["事实陈述。[S1]", "作者解释。"],
        claims=[
            Claim(
                claim_id="fact_1",
                text="事实陈述。",
                claim_type="fact",
                source_keys=["S1"],
                paragraph_index=0,
            ),
            Claim(
                claim_id="interpretation_1",
                text="作者解释。",
                claim_type="interpretation",
                paragraph_index=1,
            ),
        ],
        evidence_pack=EvidencePack(
            section_id="1",
            items=[
                EvidenceItem(
                    source_key="S1",
                    chunk_id="chunk_1",
                    doc_id="doc_1",
                    verbatim_excerpt="事实陈述。",
                )
            ],
        ),
    )
    messages = verification_messages(section_draft=draft)
    request_text = messages[-1]["content"].split("任务要求_JSON\n", 1)[1].split(
        "\n来源数据_JSON_开始",
        1,
    )[0]
    request = json.loads(request_text)
    decision_schema = request["response_schema"]["$defs"]["VerificationDecision"]
    assert "claim_id" in decision_schema["properties"]
    assert "claim" not in decision_schema["properties"]
    assert "non_fact_claims" not in messages[-1]["content"]

    class LegacyClient:
        def chat(self, messages, **_kwargs):
            return ChatResult(
                content=json.dumps(
                    {
                        "section_id": "1",
                        "verified_claims": [
                            {
                                "claim": draft.claims[0].model_dump(),
                                "verdict": "supported",
                                "verifier_rationale": "原文直接支持。",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                model="fake",
            )

    verified = verify_section(section_draft=draft, siliconflow=LegacyClient())

    assert verified.unsupported_count == 0
    assert verified.supported_count == 2
    assert {item.claim.claim_id for item in verified.verified_claims} == {
        "fact_1",
        "interpretation_1",
    }

    with pytest.raises(ValueError, match="缺少引用标记"):
        SectionDraft.model_validate(
            {
                "section_id": "1",
                "heading": "标题",
                "paragraphs": ["有来源但没有标记。"],
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "有来源但没有标记。",
                        "claim_type": "fact",
                        "source_keys": ["S1"],
                        "paragraph_index": 0,
                    }
                ],
                "evidence_pack": {
                    "section_id": "1",
                    "items": [
                        {
                            "source_key": "S1",
                            "chunk_id": "chunk",
                            "doc_id": "doc",
                            "verbatim_excerpt": "原文",
                        }
                    ],
                },
            }
        )


def test_partial_verdict_never_routes_to_polish() -> None:
    claim = Claim(
        claim_id="c1",
        text="事实",
        claim_type="fact",
        source_keys=["S1"],
        paragraph_index=0,
    )
    verified = VerifiedDraft.model_validate(
        {
            "section_id": "1",
            "verified_claims": [
                {
                    "claim": claim.model_dump(),
                    "verdict": "partial",
                    "verifier_rationale": "仅部分支持",
                }
            ],
            "supported_count": 0,
            "partial_count": 1,
            "unsupported_count": 0,
        }
    )
    state = {
        "current_section_index": 0,
        "sections": [
            {
                "section_id": "1",
                "verified_draft_json": verified.model_dump_json(),
                "revision_count": 0,
            }
        ],
    }
    assert should_continue_after_verify(state) == "revise"


def test_reference_assembler_uses_citeproc() -> None:
    repository = FakeKnowledgeRepository()
    result: ReferenceList = assemble_reference_list(
        [
            EvidenceItem(
                source_key="S1",
                chunk_id="chunk_1",
                doc_id="doc_task",
                verbatim_excerpt="原文",
            )
        ],
        citation_style="gb-t-7714",
        kb_repository=repository,
        kb_id="kb",
    )
    assert result.items[0].citation_text == "张三. 数字人文研究. 出版研究, 2024(2): 10-20."
