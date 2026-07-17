"""Recoverable whole-document Map/Reduce distillation of nonfiction composition DNA."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from contextvars import copy_context
from typing import Any

from pydantic import ValidationError

from writing_factory.distill.composition_models import (
    CompositionDNA,
    CompositionReduceResult,
    DocumentCompositionProfile,
)
from writing_factory.distill.composition_prompts import (
    composition_reduce_messages,
    document_composition_messages,
)
from writing_factory.distill.composition_validation import (
    assemble_composition_dna,
    ensure_composition_chinese,
    normalize_composition_evidence_ownership,
    validate_composition_reduce,
    validate_document_profile,
)
from writing_factory.distill.extraction import StructuredDistillationError
from writing_factory.distill.models import PersonaMode, SourceInfo, SourceSegment, SourceUnit
from writing_factory.llm import SiliconFlowClient
from writing_factory.llm.models import ChatResult
from writing_factory.store.persona_repository import PersonaRepository

ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]


class CompositionDistiller:
    """Extract ordered document structure and reduce it into safe reusable rules."""

    PROFILE_VERSION = "composition-document-v2-document-scoped-cache"
    REDUCE_VERSION = "composition-reduce-v2-evidence-ownership"
    MAX_DOCUMENT_CHARACTERS = 180_000

    def __init__(
        self,
        siliconflow: SiliconFlowClient,
        repository: PersonaRepository,
        *,
        max_attempts: int = 2,
    ) -> None:
        self.siliconflow = siliconflow
        self.repository = repository
        self.max_attempts = max_attempts

    def distill(
        self,
        *,
        run_id: str,
        name: str,
        mode: PersonaMode,
        target_units: tuple[SourceUnit, ...],
        target_source_info: tuple[SourceInfo, ...],
        target_hash: str,
        control_units: tuple[SourceUnit, ...] = (),
        control_source_info: tuple[SourceInfo, ...] = (),
        control_hash: str = "",
        parallelism: int = 3,
        reuse_persona_id: str | None = None,
        progress: ProgressCallback,
        progress_start: int,
        progress_end: int,
        check_cancelled: CancellationCheck,
    ) -> CompositionDNA:
        """Run or resume document profiles and one global composition Reduce."""

        span = max(1, progress_end - progress_start)

        def local_progress(percent: int, message: str) -> None:
            progress(progress_start + round(span * percent / 100), message)

        target_profiles = self._profile_corpus(
            run_id=run_id,
            stage="composition_target_document",
            mode=mode,
            units=target_units,
            source_info=target_source_info,
            corpus_role="target",
            reuse_persona_id=reuse_persona_id,
            parallelism=parallelism,
            progress=local_progress,
            progress_start=0,
            progress_end=65,
            check_cancelled=check_cancelled,
        )
        control_profiles: list[DocumentCompositionProfile] = []
        if control_source_info:
            control_profiles = self._profile_corpus(
                run_id=run_id,
                stage="composition_control_document",
                mode=mode,
                units=control_units,
                source_info=control_source_info,
                corpus_role="control",
                reuse_persona_id=reuse_persona_id,
                parallelism=parallelism,
                progress=local_progress,
                progress_start=65,
                progress_end=80,
                check_cancelled=check_cancelled,
            )
        check_cancelled()
        local_progress(82, "归并同文体谋篇模式")
        reduce_hash = self._hash(
            self.REDUCE_VERSION,
            name,
            mode,
            target_hash,
            control_hash,
            [item.model_dump(mode="json") for item in target_profiles],
            [item.model_dump(mode="json") for item in control_profiles],
        )
        reduced = self.repository.load_stage_result(
            run_id=run_id,
            stage="composition_reduce",
            item_id="global",
            model=CompositionReduceResult,
        )
        if reduced is None:
            reduced = self._reduce(name, mode, target_profiles, control_profiles)
        reduced = normalize_composition_evidence_ownership(
            reduced,
            target_profiles,
            control_available=bool(control_profiles),
        )
        validate_composition_reduce(reduced, target_profiles, control_profiles)
        self.repository.save_stage_result(
            run_id=run_id,
            stage="composition_reduce",
            item_id="global",
            input_hash=reduce_hash,
            result=reduced,
        )
        local_progress(96, "装配谋篇 DNA")
        segments = self._segments_by_chunk(target_units)
        result = assemble_composition_dna(reduced, segments, bool(control_profiles))
        local_progress(100, "谋篇 DNA 完成")
        return result

    def _profile_corpus(
        self,
        *,
        run_id: str,
        stage: str,
        mode: PersonaMode,
        units: tuple[SourceUnit, ...],
        source_info: tuple[SourceInfo, ...],
        corpus_role: str,
        reuse_persona_id: str | None,
        parallelism: int,
        progress: ProgressCallback,
        progress_start: int,
        progress_end: int,
        check_cancelled: CancellationCheck,
    ) -> list[DocumentCompositionProfile]:
        grouped = self._segments_by_document(units)
        profiles: dict[str, DocumentCompositionProfile] = {}
        missing: list[tuple[SourceInfo, list[SourceSegment], str]] = []
        for source in source_info:
            segments = grouped.get(source.doc_id, [])
            input_hash = self._hash(
                self.PROFILE_VERSION,
                mode,
                corpus_role,
                source.doc_id,
                source.title,
                source.filename,
                [
                    (item.chunk_id, hashlib.sha256(item.text.encode()).hexdigest())
                    for item in segments
                ],
            )
            cached = self.repository.load_stage_result(
                run_id=run_id,
                stage=stage,
                item_id=source.doc_id,
                model=DocumentCompositionProfile,
            )
            if cached is None:
                cached = self.repository.find_compatible_stage_result(
                    stage=stage,
                    item_id=source.doc_id,
                    input_hash=input_hash,
                    model=DocumentCompositionProfile,
                    persona_id=reuse_persona_id,
                )
            if cached is not None:
                validate_document_profile(cached, source.doc_id, segments)
                profiles[source.doc_id] = cached
                self.repository.save_stage_result(
                    run_id=run_id,
                    stage=stage,
                    item_id=source.doc_id,
                    input_hash=input_hash,
                    result=cached,
                )
            else:
                missing.append((source, segments, input_hash))
        completed = len(profiles)
        total = len(source_info)
        if missing:
            workers = min(max(1, parallelism), len(missing))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=stage) as executor:
                futures: dict[Future[DocumentCompositionProfile], tuple[SourceInfo, str]] = {
                    executor.submit(
                        copy_context().run,
                        self._profile_document,
                        source,
                        segments,
                        mode,
                        corpus_role,
                    ): (source, input_hash)
                    for source, segments, input_hash in missing
                }
                for future in as_completed(futures):
                    check_cancelled()
                    source, input_hash = futures[future]
                    profile = future.result()
                    profiles[source.doc_id] = profile
                    self.repository.save_stage_result(
                        run_id=run_id,
                        stage=stage,
                        item_id=source.doc_id,
                        input_hash=input_hash,
                        result=profile,
                    )
                    completed += 1
                    percent = progress_start + round(
                        (progress_end - progress_start) * completed / max(1, total)
                    )
                    progress(percent, f"并发分析完整文档结构（{completed}/{total}）")
        return [profiles[item.doc_id] for item in source_info]

    def _profile_document(
        self,
        source: SourceInfo,
        segments: list[SourceSegment],
        mode: PersonaMode,
        corpus_role: str,
    ) -> DocumentCompositionProfile:
        messages = document_composition_messages(
            source=source,
            segments=self._document_payload(segments),
            mode=mode,
            corpus_role=corpus_role,
        )
        last_error = "未知校验错误"
        role_label = "目标" if corpus_role == "target" else "对照"
        stage = getattr(self.siliconflow, "stream_stage", None)
        stream_context = (
            stage(f"{role_label}谋篇 Map · {source.filename}")
            if callable(stage)
            else nullcontext()
        )
        with stream_context:
            for attempt in range(self.max_attempts):
                active = messages if not attempt else [
                    *messages,
                    {"role": "user", "content": f"请修正并返回完整 JSON。校验错误：{last_error}"},
                ]

                def parse_result(value: ChatResult) -> DocumentCompositionProfile:
                    profile = DocumentCompositionProfile.model_validate_json(value.content)
                    validate_document_profile(profile, source.doc_id, segments)
                    ensure_composition_chinese(profile)
                    return profile

                try:
                    result = self.siliconflow.chat(
                        active,
                        thinking=True,
                        reasoning_effort="high",
                        temperature=0.0,
                        max_tokens=8192,
                        seed=23,
                        response_format="json_object",
                        use_cache=attempt == 0,
                        request_attempts=2,
                        stream=True,
                        result_validator=lambda value: parse_result(value),
                        step_id="distill.structure_map",
                    )
                    return parse_result(result)
                except (ValidationError, ValueError, StructuredDistillationError) as exc:
                    last_error = str(exc)[:2000]
            raise StructuredDistillationError(f"完整文档谋篇分析失败：{last_error}")

    def _reduce(
        self,
        name: str,
        mode: PersonaMode,
        target_profiles: list[DocumentCompositionProfile],
        control_profiles: list[DocumentCompositionProfile],
    ) -> CompositionReduceResult:
        messages = composition_reduce_messages(
            target_name=name,
            mode=mode,
            target_profiles=target_profiles,
            control_profiles=control_profiles,
        )
        last_error = "未知校验错误"
        stage = getattr(self.siliconflow, "stream_stage", None)
        with stage("谋篇 DNA 归并") if callable(stage) else nullcontext():
            for attempt in range(self.max_attempts):
                active = messages if not attempt else [
                    *messages,
                    {
                        "role": "user",
                        "content": f"请修正并返回完整 JSON。校验错误：{last_error}",
                    },
                ]

                def parse_result(value: ChatResult) -> CompositionReduceResult:
                    reduced = CompositionReduceResult.model_validate_json(value.content)
                    reduced = normalize_composition_evidence_ownership(
                        reduced,
                        target_profiles,
                        control_available=bool(control_profiles),
                    )
                    validate_composition_reduce(reduced, target_profiles, control_profiles)
                    ensure_composition_chinese(reduced)
                    return reduced

                try:
                    result = self.siliconflow.chat(
                        active,
                        thinking=True,
                        reasoning_effort="high",
                        temperature=0.0,
                        max_tokens=8192,
                        seed=29,
                        response_format="json_object",
                        use_cache=attempt == 0,
                        request_attempts=2,
                        stream=True,
                        result_validator=lambda value: parse_result(value),
                        step_id="distill.structure_reduce",
                    )
                    return parse_result(result)
                except (ValidationError, ValueError, StructuredDistillationError) as exc:
                    last_error = str(exc)[:2500]
            raise StructuredDistillationError(f"谋篇 DNA 归并失败：{last_error}")

    @classmethod
    def _document_payload(cls, segments: list[SourceSegment]) -> list[dict[str, Any]]:
        per_segment = max(
            800,
            min(8000, cls.MAX_DOCUMENT_CHARACTERS // max(1, len(segments))),
        )
        return [
            {
                "order": index,
                "chunk_id": item.chunk_id,
                "section_heading": item.section_heading,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "text": item.text[:per_segment],
                "truncated": len(item.text) > per_segment,
            }
            for index, item in enumerate(segments)
        ]

    @staticmethod
    def _segments_by_document(units: tuple[SourceUnit, ...]) -> dict[str, list[SourceSegment]]:
        grouped: dict[str, list[SourceSegment]] = {}
        for unit in units:
            for segment in unit.segments:
                grouped.setdefault(segment.doc_id, []).append(segment)
        return grouped

    @staticmethod
    def _segments_by_chunk(units: tuple[SourceUnit, ...]) -> dict[str, SourceSegment]:
        return {segment.chunk_id: segment for unit in units for segment in unit.segments}

    @staticmethod
    def _hash(*values: object) -> str:
        payload = json.dumps(values, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()
