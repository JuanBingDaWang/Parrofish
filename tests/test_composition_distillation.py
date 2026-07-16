"""Whole-document composition-DNA extraction, caching, and runtime isolation tests."""

from __future__ import annotations

import json
import threading
import time

import pytest

from tests.test_distill_pipeline import _persona
from writing_factory.distill.composition import CompositionDistiller
from writing_factory.distill.composition_models import (
    CompositionReduceResult,
    DocumentCompositionProfile,
    DocumentPatternCandidate,
    ReducedCompositionPattern,
    ReducedGenreCompositionProfile,
)
from writing_factory.distill.composition_validation import validate_composition_reduce
from writing_factory.distill.extraction import StructuredDistillationError
from writing_factory.distill.models import SourceInfo, SourceSegment, SourceUnit
from writing_factory.distill.runtime import build_runtime_persona
from writing_factory.generate.persona_context import persona_context_for_genre
from writing_factory.llm.models import ChatResult
from writing_factory.store import Database
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.persona_repository import PersonaRepository


class CompositionFakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.first_map_barrier = threading.Barrier(2)

    def chat(self, messages, **kwargs):
        step_id = str(kwargs["step_id"])
        with self.lock:
            self.calls.append(step_id)
            call_number = self.calls.count("distill.structure_map")
            self.active += 1
            self.peak = max(self.peak, self.active)
        if step_id == "distill.structure_map" and call_number <= 2:
            self.first_map_barrier.wait(timeout=1)
        time.sleep(0.01)
        try:
            payload = _source_payload(messages[-1]["content"])
            if step_id == "distill.structure_map":
                segment = payload["ordered_source_segments"][0]
                result = DocumentCompositionProfile(
                    doc_id=payload["doc_id"],
                    genre="commentary",
                    genre_label="评论 / 观点文章",
                    purpose="围绕公共问题形成有边界的判断",
                    audience="关注现实议题的普通读者",
                    heading_strategy="用功能明确的小标题推动观点",
                    paragraph_strategy="先判断，再给依据，随后解释并限定",
                    patterns=[
                        DocumentPatternCandidate(
                            name="判断到限定的推进",
                            scope="document",
                            description="先提出判断，再以依据和解释推进，最后说明适用边界",
                            sequence=["提出判断", "给出依据", "解释意义", "限定边界"],
                            relations=["递进", "因果", "限定"],
                            applicability="评论现实问题时",
                            variability="依据和解释可以交替出现",
                            evidence_chunk_ids=[segment["chunk_id"]],
                        )
                    ],
                )
            else:
                target_profiles = payload["target_document_profiles"]
                doc_ids = [item["doc_id"] for item in target_profiles]
                chunk_ids = [
                    item["patterns"][0]["evidence_chunk_ids"][0]
                    for item in target_profiles
                ]
                result = CompositionReduceResult(
                    genre_profiles=[
                        ReducedGenreCompositionProfile(
                            genre="commentary",
                            genre_label="评论 / 观点文章",
                            source_document_count=len(doc_ids),
                            typical_purposes=["解释现实问题并形成有边界的公共判断"],
                            audience_tendencies=["面向具有一般背景知识的公众"],
                            heading_strategy="以论证功能而不是主题名设置小标题",
                            paragraph_strategy="判断、依据、解释和限定形成连续推进",
                            patterns=[
                                ReducedCompositionPattern(
                                    pattern_id="composition_commentary_1",
                                    name="判断到限定的推进",
                                    scope="document",
                                    description="从明确判断出发，经依据和解释后落到适用边界",
                                    sequence=["提出判断", "给出依据", "解释意义", "限定边界"],
                                    relations=["递进", "因果", "限定"],
                                    applicability="面向公众评论现实问题时",
                                    variability="可根据篇幅合并依据与解释单元",
                                    evidence_chunk_ids=chunk_ids,
                                    supporting_doc_ids=doc_ids,
                                    recurrence_document_count=len(doc_ids),
                                    specificity="unverified",
                                    confidence="medium",
                                )
                            ],
                            declared_limits=["当前语料只覆盖评论文体，不能外推到其他文体"],
                        )
                    ]
                )
            return ChatResult(content=result.model_dump_json(), model="fixture")
        finally:
            with self.lock:
                self.active -= 1


def _source_payload(content: str) -> dict:
    raw = content.split("来源数据_JSON_开始\n", 1)[1].rsplit("\n来源数据_JSON_结束", 1)[0]
    return json.loads(raw)


def _corpus(count: int = 2) -> tuple[tuple[SourceUnit, ...], tuple[SourceInfo, ...]]:
    units = tuple(
        SourceUnit(
            unit_id=f"unit_{index}",
            segments=[
                SourceSegment(
                    chunk_id=f"chunk_{index}",
                    doc_id=f"doc_{index}",
                    document_title=f"评论{index}",
                    filename=f"commentary_{index}.txt",
                    text="先提出公共判断。随后提供依据并解释意义，最后说明判断的适用边界。",
                    section_heading="正文",
                )
            ],
        )
        for index in range(count)
    )
    sources = tuple(
        SourceInfo(
            doc_id=f"doc_{index}",
            title=f"评论{index}",
            filename=f"commentary_{index}.txt",
            chunk_count=1,
        )
        for index in range(count)
    )
    return units, sources


def test_composition_distillation_is_concurrent_cached_and_source_isolated(tmp_path) -> None:
    database = Database(tmp_path / "composition.db")
    database.initialize()
    KnowledgeBaseRepository(database).ensure_default()
    repository = PersonaRepository(database)
    client = CompositionFakeClient()
    distiller = CompositionDistiller(client, repository)
    units, sources = _corpus()
    first_run = repository.begin_or_resume(
        name="评论作者",
        mode="person",
        kb_id="kb_default",
        source_hash="source_hash",
        input_hash="input_one",
        source_doc_ids=[item.doc_id for item in sources],
        map_total=2,
    )

    composition = distiller.distill(
        run_id=first_run.run_id,
        name="评论作者",
        mode="person",
        target_units=units,
        target_source_info=sources,
        target_hash="source_hash",
        parallelism=2,
        progress=lambda _percent, _message: None,
        progress_start=0,
        progress_end=100,
        check_cancelled=lambda: None,
    )

    assert client.peak == 2
    assert client.calls.count("distill.structure_map") == 2
    assert client.calls.count("distill.structure_reduce") == 1
    profile = composition.genre_profiles[0]
    assert profile.genre == "commentary"
    assert profile.patterns[0].recurrence_document_count == 2
    assert profile.patterns[0].specificity == "unverified"
    first_persona = _persona(first_run.persona_id).model_copy(
        update={"composition_dna": composition}
    )
    repository.save_ready(
        run_id=first_run.run_id,
        persona=first_persona,
        markdown="first",
    )

    second_run = repository.begin_or_resume(
        name="评论作者",
        mode="person",
        kb_id="kb_default",
        source_hash="source_hash",
        input_hash="input_two",
        source_doc_ids=[item.doc_id for item in sources],
        map_total=2,
        strategy="upgrade",
        base_persona_id=first_run.persona_id,
    )
    reused = distiller.distill(
        run_id=second_run.run_id,
        name="评论作者",
        mode="person",
        target_units=units,
        target_source_info=sources,
        target_hash="source_hash",
        parallelism=2,
        reuse_persona_id=first_run.persona_id,
        progress=lambda _percent, _message: None,
        progress_start=0,
        progress_end=100,
        check_cancelled=lambda: None,
    )
    assert reused == composition
    assert client.calls.count("distill.structure_map") == 2
    assert client.calls.count("distill.structure_reduce") == 2
    second_persona = _persona(second_run.persona_id).model_copy(
        update={"composition_dna": reused}
    )
    repository.save_ready(
        run_id=second_run.run_id,
        persona=second_persona,
        markdown="second",
    )

    expanded_units, expanded_sources = _corpus(3)
    third_run = repository.begin_or_resume(
        name="评论作者",
        mode="person",
        kb_id="kb_default",
        source_hash="expanded_source_hash",
        input_hash="input_three",
        source_doc_ids=[item.doc_id for item in expanded_sources],
        map_total=3,
        strategy="upgrade",
        base_persona_id=second_run.persona_id,
    )
    expanded = distiller.distill(
        run_id=third_run.run_id,
        name="评论作者",
        mode="person",
        target_units=expanded_units,
        target_source_info=expanded_sources,
        target_hash="expanded_source_hash",
        parallelism=2,
        reuse_persona_id=second_run.persona_id,
        progress=lambda _percent, _message: None,
        progress_start=0,
        progress_end=100,
        check_cancelled=lambda: None,
    )
    assert expanded.genre_profiles[0].source_document_count == 3
    assert client.calls.count("distill.structure_map") == 3
    assert client.calls.count("distill.structure_reduce") == 3

    persona = _persona("persona_composition").model_copy(
        update={"composition_dna": composition}
    )
    runtime = build_runtime_persona(persona)
    runtime_text = runtime.model_dump_json()
    assert "chunk_0" not in runtime_text
    assert "doc_0" not in runtime_text
    assert "evidence_id" not in runtime_text

    commentary = persona_context_for_genre(runtime, "commentary")
    academic = persona_context_for_genre(runtime, "academic_paper")
    assert len(commentary["composition_dna"]["genre_profiles"]) == 1
    assert academic["composition_dna"]["genre_profiles"] == []


def test_composition_reduce_requires_exact_genre_partition() -> None:
    commentary_profile = DocumentCompositionProfile(
        doc_id="commentary_doc",
        genre="commentary",
        genre_label="评论 / 观点文章",
        purpose="形成公共判断",
        audience="普通读者",
        heading_strategy="功能标题",
        paragraph_strategy="判断后解释",
        patterns=[
            DocumentPatternCandidate(
                name="判断后解释",
                scope="document",
                description="先判断再解释",
                applicability="评论任务",
                variability="允许插入例证",
                evidence_chunk_ids=["commentary_chunk"],
            )
        ],
    )
    speech_profile = DocumentCompositionProfile(
        doc_id="speech_doc",
        genre="speech",
        genre_label="演讲 / 发言稿",
        purpose="促成听众理解",
        audience="现场听众",
        heading_strategy="不使用标题",
        paragraph_strategy="口头节奏推进",
        patterns=[
            DocumentPatternCandidate(
                name="口头递进",
                scope="document",
                description="由共识推进到行动",
                applicability="演讲任务",
                variability="允许听众互动",
                evidence_chunk_ids=["speech_chunk"],
            )
        ],
    )
    incomplete = CompositionReduceResult(
        genre_profiles=[
            ReducedGenreCompositionProfile(
                genre="commentary",
                genre_label="评论 / 观点文章",
                source_document_count=1,
                heading_strategy="功能标题",
                paragraph_strategy="判断后解释",
            )
        ]
    )
    with pytest.raises(StructuredDistillationError, match="完整覆盖"):
        validate_composition_reduce(incomplete, [commentary_profile, speech_profile], [])

    mixed = incomplete.model_copy(
        update={
            "genre_profiles": [
                incomplete.genre_profiles[0].model_copy(
                    update={
                        "patterns": [
                            ReducedCompositionPattern(
                                pattern_id="mixed_pattern",
                                name="错误混合",
                                scope="document",
                                description="错误地混合两种文体",
                                applicability="未知",
                                variability="未知",
                                evidence_chunk_ids=["speech_chunk"],
                                supporting_doc_ids=["speech_doc"],
                                recurrence_document_count=1,
                                specificity="provisional",
                                confidence="low",
                            )
                        ]
                    }
                ),
                ReducedGenreCompositionProfile(
                    genre="speech",
                    genre_label="演讲 / 发言稿",
                    source_document_count=1,
                    heading_strategy="不使用标题",
                    paragraph_strategy="口头节奏推进",
                ),
            ]
        }
    )
    with pytest.raises(StructuredDistillationError, match="混入了其他文体"):
        validate_composition_reduce(mixed, [commentary_profile, speech_profile], [])
