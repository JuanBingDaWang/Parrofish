"""Recoverable map-reduce orchestration for person and topic PersonaSpecs."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from datetime import date

from writing_factory.distill.academic_pipeline import AcademicDistillationEngine
from writing_factory.distill.composition import CompositionDistiller
from writing_factory.distill.expression import ExpressionAnalyzer
from writing_factory.distill.extraction import (
    PersonaMapExtractor,
    StructuredDistillationError,
)
from writing_factory.distill.language import DEFAULT_OUTPUT_LANGUAGE, OutputLanguage
from writing_factory.distill.models import (
    DistillationOutcome,
    MapResult,
    PersonaMode,
    SourceUnit,
)
from writing_factory.distill.quality import run_static_quality_check
from writing_factory.distill.serialization import render_persona_markdown
from writing_factory.distill.sources import SourceCorpusBuilder
from writing_factory.distill.synthesis import CandidateBundleBuilder, PersonaSynthesizer
from writing_factory.store.persona_repository import PersonaRepository

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]


def _no_progress(_percent: int, _message: str) -> None:
    pass


def _no_cancellation() -> None:
    pass


class DistillationService:
    """Persist each map result and publish only a fully validated PersonaSpec."""

    MAP_PIPELINE_VERSION = "persona-map-v4-academic-zh"
    REDUCE_PIPELINE_VERSION = "persona-reduce-v7-composition-dna"

    def __init__(
        self,
        repository: PersonaRepository,
        sources: SourceCorpusBuilder,
        extractor: PersonaMapExtractor,
        synthesizer: PersonaSynthesizer,
        expression: ExpressionAnalyzer | None = None,
        *,
        map_concurrency: int = 3,
        output_language: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE,
        academic_engine: AcademicDistillationEngine | None = None,
        composition_distiller: CompositionDistiller | None = None,
    ) -> None:
        self.repository = repository
        self.sources = sources
        self.extractor = extractor
        self.synthesizer = synthesizer
        self.expression = expression or ExpressionAnalyzer()
        self.map_concurrency = max(1, min(8, map_concurrency))
        self.output_language = output_language
        self.academic_engine = academic_engine
        self.composition_distiller = composition_distiller

    def set_max_parallel_tasks(self, value: int) -> None:
        """调整各独立蒸馏阶段的任务提交数；全局请求仍由客户端闸门兜底。"""

        if not 1 <= value <= 8:
            raise ValueError("SiliconFlow 最大并发数必须在 1 至 8 之间")
        self.map_concurrency = value

    def distill(
        self,
        *,
        kb_id: str,
        name: str,
        mode: PersonaMode,
        doc_ids: set[str] | None = None,
        control_doc_ids: set[str] | None = None,
        domain: str = "",
        progress: ProgressCallback = _no_progress,
        check_cancelled: CancellationCheck = _no_cancellation,
    ) -> DistillationOutcome:
        """Distill or resume one profile from ready KB documents."""

        label = name.strip()
        if not label:
            raise ValueError("Persona name cannot be empty")
        progress(2, "读取蒸馏语料")
        corpus = self.sources.build(kb_id, doc_ids=doc_ids)
        target_source_info = tuple(
            item.model_copy(update={"corpus_role": "target", "domain": domain.strip()})
            for item in corpus.source_info
        )
        use_academic = self.academic_engine is not None and mode == "person"
        control_corpus = None
        if use_academic and control_doc_ids:
            if not domain.strip():
                raise ValueError("使用对照语料时必须填写研究领域")
            overlap = set(doc_ids or ()) & set(control_doc_ids)
            if overlap:
                raise ValueError("同一文档不能同时作为目标语料和对照语料")
            control_corpus = self.sources.build(kb_id, doc_ids=control_doc_ids)
        control_source_info = (
            tuple(
                item.model_copy(update={"corpus_role": "control", "domain": domain.strip()})
                for item in control_corpus.source_info
            )
            if control_corpus is not None
            else ()
        )
        map_input_hash = self._map_input_hash(
            label,
            mode,
            corpus.source_hash,
            self.output_language,
            "target",
            domain,
        )
        control_map_hash = (
            self._map_input_hash(
                "同领域对照语料",
                mode,
                control_corpus.source_hash,
                self.output_language,
                "control",
                domain,
            )
            if control_corpus is not None
            else None
        )
        input_hash = self._input_hash(
            label,
            mode,
            corpus.source_hash,
            control_source_hash=control_corpus.source_hash if control_corpus else "",
            domain=domain,
        )
        ready = self.repository.find_ready(
            name=label,
            mode=mode,
            kb_id=kb_id,
            source_hash=corpus.source_hash,
            input_hash=input_hash,
        )
        if ready is not None:
            progress(100, "复用作者档案")
            return DistillationOutcome(
                run_id=ready[0], persona=ready[1], markdown=ready[2], reused=True
            )
        run = self.repository.begin_or_resume(
            name=label,
            mode=mode,
            kb_id=kb_id,
            source_hash=corpus.source_hash,
            input_hash=input_hash,
            source_doc_ids=[item.doc_id for item in corpus.source_info],
            map_total=len(corpus.units) + (len(control_corpus.units) if control_corpus else 0),
            control_doc_ids=[item.doc_id for item in control_source_info],
            domain=domain.strip(),
        )
        try:
            map_results = self._run_maps(
                run_id=run.run_id,
                label=label,
                mode=mode,
                units=corpus.units,
                map_input_hash=map_input_hash,
                progress=progress,
                check_cancelled=check_cancelled,
                corpus_role="target",
                domain=domain,
                progress_start=5,
                progress_end=55 if control_corpus else 65,
            )
            control_map_results: list[MapResult] = []
            if control_corpus is not None and control_map_hash is not None:
                control_map_results = self._run_maps(
                    run_id=run.run_id,
                    label="同领域对照语料",
                    mode=mode,
                    units=control_corpus.units,
                    map_input_hash=control_map_hash,
                    progress=progress,
                    check_cancelled=check_cancelled,
                    corpus_role="control",
                    domain=domain,
                    progress_start=55,
                    progress_end=65,
                )
            check_cancelled()
            self.repository.update_stage(run.run_id, run.persona_id, "reducing")
            texts = [segment.text for unit in corpus.units for segment in unit.segments]
            expression = self.expression.analyze(texts)
            academic_registry = None
            if use_academic and self.academic_engine is not None:
                _registry, _gaps, target_bundle = CandidateBundleBuilder().build(
                    map_results, corpus.units
                )
                control_bundle = None
                if control_corpus is not None:
                    _control_registry, _control_gaps, control_bundle = (
                        CandidateBundleBuilder().build(
                            control_map_results,
                            control_corpus.units,
                        )
                    )
                academic_registry = self.academic_engine.build_registry(
                    run_id=run.run_id,
                    target_label=label,
                    domain=domain.strip(),
                    target_bundle=target_bundle,
                    target_source_info=target_source_info,
                    target_hash=corpus.source_hash,
                    control_bundle=control_bundle,
                    control_source_info=control_source_info,
                    control_hash=control_corpus.source_hash if control_corpus else None,
                    progress=progress,
                    check_cancelled=check_cancelled,
                )
            composition_dna = None
            if self.composition_distiller is not None:
                composition_dna = self.composition_distiller.distill(
                    run_id=run.run_id,
                    name=label,
                    mode=mode,
                    target_units=corpus.units,
                    target_source_info=target_source_info,
                    target_hash=corpus.source_hash,
                    control_units=control_corpus.units if control_corpus is not None else (),
                    control_source_info=control_source_info,
                    control_hash=control_corpus.source_hash if control_corpus is not None else "",
                    parallelism=self.map_concurrency,
                    progress=progress,
                    progress_start=91 if use_academic else 66,
                    progress_end=97 if use_academic else 90,
                    check_cancelled=check_cancelled,
                )
            progress(98 if use_academic else 92, "装配作者档案")
            synthesize_options = dict(
                persona_id=run.persona_id,
                name=label,
                mode=mode,
                map_results=map_results,
                units=corpus.units,
                source_info=target_source_info,
                expression=expression,
                research_date=date.today(),
                academic_registry=academic_registry,
                control_source_info=control_source_info,
            )
            if composition_dna is not None:
                synthesize_options["composition_dna"] = composition_dna
            persona = self.synthesizer.synthesize(**synthesize_options)
            check_cancelled()
            self.repository.update_stage(run.run_id, run.persona_id, "validating")
            progress(99, "生成确定性档案")
            markdown = render_persona_markdown(persona)
            quality = run_static_quality_check(persona)
            self.repository.save_evaluation(
                persona_id=persona.id,
                evaluation_type="nuwa_static",
                result_json=quality.model_dump_json(),
            )
            if not quality.passed:
                raise ValueError("PersonaSpec failed independent static quality checks")
            self.repository.save_ready(
                run_id=run.run_id,
                persona=persona,
                markdown=markdown,
            )
            progress(100, "蒸馏完成")
            return DistillationOutcome(
                run_id=run.run_id,
                persona=persona,
                markdown=markdown,
            )
        except Exception as exc:
            logger.exception("Distillation failed: %s", type(exc).__name__)
            self.repository.mark_failed(run.run_id, run.persona_id, type(exc).__name__)
            raise

    @classmethod
    def _map_input_hash(
        cls,
        name: str,
        mode: PersonaMode,
        source_hash: str,
        output_language: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE,
        corpus_role: str = "target",
        domain: str = "",
    ) -> str:
        value = (
            f"{cls.MAP_PIPELINE_VERSION}|{name}|{mode}|{source_hash}|"
            f"{output_language}|{corpus_role}|{domain.strip()}"
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _input_hash(
        self,
        name: str,
        mode: PersonaMode,
        source_hash: str,
        *,
        control_source_hash: str = "",
        domain: str = "",
    ) -> str:
        map_hash = self._map_input_hash(
            name,
            mode,
            source_hash,
            self.output_language,
        )
        value = f"{self.REDUCE_PIPELINE_VERSION}|{map_hash}|{control_source_hash}|{domain.strip()}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _run_maps(
        self,
        *,
        run_id: str,
        label: str,
        mode: PersonaMode,
        units: tuple[SourceUnit, ...],
        map_input_hash: str,
        progress: ProgressCallback,
        check_cancelled: CancellationCheck,
        corpus_role: str = "target",
        domain: str = "",
        progress_start: int = 5,
        progress_end: int = 65,
    ) -> list[MapResult]:
        results: dict[str, MapResult] = {}
        missing: list[SourceUnit] = []
        for unit in units:
            check_cancelled()
            mapped = self.repository.get_map_result(run_id, unit.unit_id)
            if mapped is None:
                mapped = self.repository.find_compatible_map_result(
                    input_hash=map_input_hash,
                    unit_id=unit.unit_id,
                )
                if mapped is not None:
                    self._save_map(run_id, unit, map_input_hash, mapped)
            if mapped is None:
                missing.append(unit)
            else:
                results[unit.unit_id] = mapped

        completed = len(results)
        if completed:
            progress(
                progress_start + round((progress_end - progress_start) * completed / len(units)),
                "复用思维候选",
            )
        if missing:
            completed = self._extract_maps_concurrently(
                run_id=run_id,
                label=label,
                mode=mode,
                missing=missing,
                total=len(units),
                completed=completed,
                map_input_hash=map_input_hash,
                results=results,
                progress=progress,
                check_cancelled=check_cancelled,
                corpus_role=corpus_role,
                domain=domain,
                progress_start=progress_start,
                progress_end=progress_end,
            )
        if completed != len(units):
            raise StructuredDistillationError("Map 结果数量与语料单元数量不一致")
        return [results[unit.unit_id] for unit in units]

    def _extract_maps_concurrently(
        self,
        *,
        run_id: str,
        label: str,
        mode: PersonaMode,
        missing: list[SourceUnit],
        total: int,
        completed: int,
        map_input_hash: str,
        results: dict[str, MapResult],
        progress: ProgressCallback,
        check_cancelled: CancellationCheck,
        corpus_role: str,
        domain: str,
        progress_start: int,
        progress_end: int,
    ) -> int:
        executor = ThreadPoolExecutor(
            max_workers=min(self.map_concurrency, len(missing)),
            thread_name_prefix="persona-map",
        )
        futures: dict[Future[MapResult], SourceUnit] = {}
        pending = iter(missing)

        def submit_next() -> bool:
            try:
                unit = next(pending)
            except StopIteration:
                return False
            if corpus_role == "target" and not domain.strip():
                context = copy_context()
                future = executor.submit(
                    context.run,
                    self.extractor.extract,
                    label,
                    mode,
                    unit,
                )
            else:
                context = copy_context()
                future = executor.submit(
                    context.run,
                    self.extractor.extract,
                    label,
                    mode,
                    unit,
                    corpus_role=corpus_role,
                    domain=domain.strip(),
                )
            futures[future] = unit
            return True

        for _ in range(min(self.map_concurrency, len(missing))):
            submit_next()
        first_error: Exception | None = None
        try:
            while futures:
                if first_error is None:
                    try:
                        check_cancelled()
                    except Exception as exc:
                        first_error = exc
                done, _not_done = wait(
                    futures,
                    timeout=0.25,
                    return_when=FIRST_COMPLETED,
                )
                successful: list[tuple[SourceUnit, MapResult]] = []
                for future in done:
                    unit = futures.pop(future)
                    try:
                        mapped = future.result()
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                        continue
                    successful.append((unit, mapped))
                for unit, mapped in successful:
                    try:
                        self._save_map(run_id, unit, map_input_hash, mapped)
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                        continue
                    results[unit.unit_id] = mapped
                    completed += 1
                    progress(
                        progress_start + round((progress_end - progress_start) * completed / total),
                        f"并发提取思维候选（{completed}/{total}）",
                    )
                if first_error is None:
                    for _unit, _mapped in successful:
                        try:
                            check_cancelled()
                        except Exception as exc:
                            first_error = exc
                            break
                        else:
                            submit_next()
            if first_error is not None:
                raise first_error
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        return completed

    def _save_map(
        self,
        run_id: str,
        unit: SourceUnit,
        map_input_hash: str,
        mapped: MapResult,
    ) -> None:
        self.repository.save_map_result(
            run_id=run_id,
            unit_id=unit.unit_id,
            input_hash=map_input_hash,
            chunk_ids=[segment.chunk_id for segment in unit.segments],
            result=mapped,
        )
