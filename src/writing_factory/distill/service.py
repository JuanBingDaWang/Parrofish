"""Recoverable map-reduce orchestration for person and topic PersonaSpecs."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import dataclass
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
    PersonaSpec,
    SourceInfo,
    SourceUnit,
)
from writing_factory.distill.options import LEGACY_DISTILLATION_OPTIONS, DistillationOptions
from writing_factory.distill.progress import WeightedProgress
from writing_factory.distill.quality import run_static_quality_check
from writing_factory.distill.serialization import render_persona_markdown
from writing_factory.distill.sources import SourceCorpus, SourceCorpusBuilder
from writing_factory.distill.synthesis import CandidateBundleBuilder, PersonaSynthesizer
from writing_factory.store.persona_repository import DistillationRunRecord, PersonaRepository

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]


def _no_progress(_percent: int, _message: str) -> None:
    pass


def _no_cancellation() -> None:
    pass


@dataclass(frozen=True, slots=True)
class _PreparedDistillation:
    corpus: SourceCorpus
    control_corpus: SourceCorpus | None
    target_source_info: tuple[SourceInfo, ...]
    control_source_info: tuple[SourceInfo, ...]
    options: DistillationOptions
    input_hash: str
    use_academic: bool
    control_maps_required: bool

    @property
    def map_total(self) -> int:
        control_count = (
            len(self.control_corpus.units)
            if self.control_corpus is not None and self.control_maps_required
            else 0
        )
        return len(self.corpus.units) + control_count


class DistillationService:
    """Persist each map result and publish only a fully validated PersonaSpec."""

    MAP_PIPELINE_VERSION = "persona-map-v5-unit-scoped"
    REDUCE_PIPELINE_VERSION = "persona-reduce-v9-topic-validation"
    LEGACY_MAP_PIPELINE_VERSION = "persona-map-v4-academic-zh"
    LEGACY_REDUCE_PIPELINE_VERSION = "persona-reduce-v7-composition-dna"

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
        options: DistillationOptions = LEGACY_DISTILLATION_OPTIONS,
        progress: ProgressCallback = _no_progress,
        check_cancelled: CancellationCheck = _no_cancellation,
    ) -> DistillationOutcome:
        """Create a genuinely new top-level profile without hidden name-based resume."""

        label = name.strip()
        if not label:
            raise ValueError("Persona name cannot be empty")
        prepared = self._prepare(
            kb_id=kb_id,
            name=label,
            mode=mode,
            doc_ids=doc_ids,
            control_doc_ids=control_doc_ids,
            domain=domain,
            options=options,
            progress=progress,
        )
        run = self.repository.begin_or_resume(
            name=label,
            mode=mode,
            kb_id=kb_id,
            source_hash=prepared.corpus.source_hash,
            input_hash=prepared.input_hash,
            source_doc_ids=[item.doc_id for item in prepared.corpus.source_info],
            map_total=prepared.map_total,
            control_doc_ids=[item.doc_id for item in prepared.control_source_info],
            domain=domain.strip(),
            quality_options=prepared.options,
            strategy="new",
        )
        return self._execute(
            run=run,
            label=label,
            mode=mode,
            domain=domain,
            prepared=prepared,
            reuse_persona_id=None,
            progress=progress,
            check_cancelled=check_cancelled,
        )

    def resume(
        self,
        *,
        kb_id: str,
        persona_id: str,
        progress: ProgressCallback = _no_progress,
        check_cancelled: CancellationCheck = _no_cancellation,
    ) -> DistillationOutcome:
        """Resume exactly the selected persisted run and no other same-name version."""

        context = self.repository.load_run_context(kb_id, persona_id)
        if context is None:
            raise ValueError("所选作者档案没有可读取的蒸馏记录")
        if context.status == "ready":
            raise ValueError("已经完成的档案不能继续；如需加入语料请使用“升级”")
        progress(1, f"读取精确断点 · 已完成 {context.map_completed}/{context.map_total} 个 Map")
        prepared = self._prepare(
            kb_id=kb_id,
            name=context.name,
            mode=context.mode,
            doc_ids=set(context.target_doc_ids),
            control_doc_ids=set(context.control_doc_ids),
            domain=context.domain,
            options=context.options,
            progress=progress,
        )
        if prepared.corpus.source_hash != context.source_hash:
            raise ValueError("原目标语料已经变化，不能混合续跑；请创建升级版本")
        if prepared.input_hash != context.input_hash:
            raise ValueError("原蒸馏输入已经变化，不能混合续跑；请创建升级版本")
        run = self.repository.prepare_exact_resume(context, map_total=prepared.map_total)
        return self._execute(
            run=run,
            label=context.name,
            mode=context.mode,
            domain=context.domain,
            prepared=prepared,
            reuse_persona_id=None,
            progress=progress,
            check_cancelled=check_cancelled,
        )

    def upgrade(
        self,
        *,
        kb_id: str,
        base_persona_id: str,
        doc_ids: set[str],
        control_doc_ids: set[str] | None = None,
        domain: str = "",
        options: DistillationOptions = LEGACY_DISTILLATION_OPTIONS,
        progress: ProgressCallback = _no_progress,
        check_cancelled: CancellationCheck = _no_cancellation,
    ) -> DistillationOutcome:
        """Create the next version and reuse only document work from its selected base."""

        base = self.repository.load_run_context(kb_id, base_persona_id)
        if base is None or base.status != "ready":
            raise ValueError("只能以已经完成的作者档案作为升级基础")
        prepared = self._prepare(
            kb_id=kb_id,
            name=base.name,
            mode=base.mode,
            doc_ids=doc_ids,
            control_doc_ids=control_doc_ids,
            domain=domain,
            options=options,
            progress=progress,
        )
        run = self.repository.begin_or_resume(
            name=base.name,
            mode=base.mode,
            kb_id=kb_id,
            source_hash=prepared.corpus.source_hash,
            input_hash=prepared.input_hash,
            source_doc_ids=[item.doc_id for item in prepared.corpus.source_info],
            map_total=prepared.map_total,
            control_doc_ids=[item.doc_id for item in prepared.control_source_info],
            domain=domain.strip(),
            quality_options=prepared.options,
            strategy="upgrade",
            base_persona_id=base_persona_id,
        )
        return self._execute(
            run=run,
            label=base.name,
            mode=base.mode,
            domain=domain,
            prepared=prepared,
            reuse_persona_id=base_persona_id,
            progress=progress,
            check_cancelled=check_cancelled,
        )

    def _prepare(
        self,
        *,
        kb_id: str,
        name: str,
        mode: PersonaMode,
        doc_ids: set[str] | None,
        control_doc_ids: set[str] | None,
        domain: str,
        options: DistillationOptions,
        progress: ProgressCallback,
    ) -> _PreparedDistillation:
        progress(2, "读取蒸馏语料")
        selected_targets = set(doc_ids or ())
        selected_controls = set(control_doc_ids or ())
        if selected_targets & selected_controls:
            raise ValueError("同一文档不能同时作为目标语料和对照语料")
        if selected_controls and not domain.strip():
            raise ValueError("使用对照语料时必须填写内容领域")
        normalized = options.normalized(
            mode=mode,
            has_control_corpus=bool(selected_controls),
        )
        corpus = self.sources.build(kb_id, doc_ids=selected_targets or doc_ids)
        control_corpus = (
            self.sources.build(kb_id, doc_ids=selected_controls) if selected_controls else None
        )
        target_source_info = tuple(
            item.model_copy(update={"corpus_role": "target", "domain": domain.strip()})
            for item in corpus.source_info
        )
        control_source_info = (
            tuple(
                item.model_copy(update={"corpus_role": "control", "domain": domain.strip()})
                for item in control_corpus.source_info
            )
            if control_corpus is not None
            else ()
        )
        use_academic = (
            self.academic_engine is not None
            and normalized.cross_document_validation
        )
        input_hash = self._input_hash(
            name,
            mode,
            corpus.source_hash,
            control_source_hash=control_corpus.source_hash if control_corpus else "",
            domain=domain,
            options=normalized,
        )
        return _PreparedDistillation(
            corpus=corpus,
            control_corpus=control_corpus,
            target_source_info=target_source_info,
            control_source_info=control_source_info,
            options=normalized,
            input_hash=input_hash,
            use_academic=use_academic,
            control_maps_required=use_academic and normalized.exclusivity_validation,
        )

    def _execute(
        self,
        *,
        run: DistillationRunRecord,
        label: str,
        mode: PersonaMode,
        domain: str,
        prepared: _PreparedDistillation,
        reuse_persona_id: str | None,
        progress: ProgressCallback,
        check_cancelled: CancellationCheck,
    ) -> DistillationOutcome:
        corpus = prepared.corpus
        control_corpus = prepared.control_corpus
        try:
            map_results = self._run_maps(
                run_id=run.run_id,
                label=label,
                mode=mode,
                units=corpus.units,
                progress=progress,
                check_cancelled=check_cancelled,
                corpus_role="target",
                domain=domain,
                progress_start=5,
                progress_end=55 if prepared.control_maps_required else 65,
                reuse_persona_id=reuse_persona_id,
            )
            control_map_results: list[MapResult] = []
            if control_corpus is not None and prepared.control_maps_required:
                control_map_results = self._run_maps(
                    run_id=run.run_id,
                    label="同领域对照语料",
                    mode=mode,
                    units=control_corpus.units,
                    progress=progress,
                    check_cancelled=check_cancelled,
                    corpus_role="control",
                    domain=domain,
                    progress_start=55,
                    progress_end=65,
                    reuse_persona_id=reuse_persona_id,
                )
            check_cancelled()
            self.repository.update_stage(run.run_id, run.persona_id, "reducing")
            texts = [segment.text for unit in corpus.units for segment in unit.segments]
            expression = self.expression.analyze(texts)
            branch_jobs: dict[str, Callable[[], object]] = {}
            stream_stage = getattr(
                getattr(self.extractor, "siliconflow", None),
                "stream_stage",
                None,
            )
            if prepared.use_academic and self.academic_engine is not None:
                _registry, _gaps, target_bundle = CandidateBundleBuilder().build(
                    map_results, corpus.units
                )
                control_bundle = None
                if control_corpus is not None and prepared.control_maps_required:
                    _control_registry, _control_gaps, control_bundle = (
                        CandidateBundleBuilder().build(
                            control_map_results,
                            control_corpus.units,
                        )
                    )
                def run_academic() -> object:
                    context = (
                        stream_stage("跨文档候选验证")
                        if callable(stream_stage)
                        else nullcontext()
                    )
                    with context:
                        return self.academic_engine.build_registry(
                            run_id=run.run_id,
                            mode=mode,
                            target_label=label,
                            domain=domain.strip(),
                            target_bundle=target_bundle,
                            target_source_info=prepared.target_source_info,
                            target_hash=corpus.source_hash,
                            control_bundle=control_bundle,
                            control_source_info=prepared.control_source_info,
                            control_hash=control_corpus.source_hash if control_bundle else None,
                            run_generative_validation=prepared.options.generative_validation,
                            run_exclusivity_validation=prepared.options.exclusivity_validation,
                            reuse_persona_id=reuse_persona_id,
                            progress=academic_progress,
                            check_cancelled=check_cancelled,
                        )

                branch_jobs["academic"] = run_academic
            if self.composition_distiller is not None and prepared.options.composition_dna:
                def run_composition() -> object:
                    context = (
                        stream_stage("谋篇 DNA")
                        if callable(stream_stage)
                        else nullcontext()
                    )
                    with context:
                        return self.composition_distiller.distill(
                            run_id=run.run_id,
                            name=label,
                            mode=mode,
                            target_units=corpus.units,
                            target_source_info=prepared.target_source_info,
                            target_hash=corpus.source_hash,
                            control_units=(
                                control_corpus.units if control_corpus is not None else ()
                            ),
                            control_source_info=prepared.control_source_info,
                            control_hash=(
                                control_corpus.source_hash if control_corpus is not None else ""
                            ),
                            parallelism=self.map_concurrency,
                            reuse_persona_id=reuse_persona_id,
                            progress=composition_progress,
                            progress_start=0,
                            progress_end=100,
                            check_cancelled=check_cancelled,
                        )

                branch_jobs["composition"] = run_composition

            branch_results: dict[str, object] = {}
            if branch_jobs:
                weights = {
                    "academic": max(1, len(prepared.target_source_info) + 3),
                    "composition": max(
                        1,
                        len(prepared.target_source_info) + len(prepared.control_source_info) + 1,
                    ),
                }
                selected_weights = {key: weights[key] for key in branch_jobs}
                tracker = WeightedProgress(progress, start=66, end=97, weights=selected_weights)

                def academic_progress(percent: int, message: str) -> None:
                    normalized = round((max(66, min(90, percent)) - 66) * 100 / 24)
                    tracker.update("academic", normalized, message)

                def composition_progress(percent: int, message: str) -> None:
                    tracker.update("composition", percent, message)

                if len(branch_jobs) == 1:
                    key, job = next(iter(branch_jobs.items()))
                    branch_results[key] = job()
                    tracker.complete(key)
                else:
                    with ThreadPoolExecutor(
                        max_workers=2,
                        thread_name_prefix="persona-branch",
                    ) as pool:
                        futures = {
                            pool.submit(copy_context().run, job): key
                            for key, job in branch_jobs.items()
                        }
                        for future in futures:
                            key = futures[future]
                            branch_results[key] = future.result()
                            tracker.complete(key)
            academic_registry = branch_results.get("academic")
            composition_dna = branch_results.get("composition")
            progress(98, "装配作者档案")
            synthesize_options = dict(
                persona_id=run.persona_id,
                name=label,
                mode=mode,
                map_results=map_results,
                units=corpus.units,
                source_info=prepared.target_source_info,
                expression=expression,
                research_date=date.today(),
                academic_registry=academic_registry,
                control_source_info=prepared.control_source_info,
            )
            if composition_dna is not None:
                synthesize_options["composition_dna"] = composition_dna
            context = (
                stream_stage("作者档案汇总与校验")
                if callable(stream_stage)
                else nullcontext()
            )
            with context:
                persona = self.synthesizer.synthesize(**synthesize_options)
                persona = self._apply_quality_scope(persona, prepared.options)
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
        unit: SourceUnit,
        output_language: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE,
        corpus_role: str = "target",
        domain: str = "",
    ) -> str:
        digest = hashlib.sha256()
        digest.update(
            (
                f"{cls.MAP_PIPELINE_VERSION}|{name}|{mode}|{output_language}|"
                f"{corpus_role}|{domain.strip()}|{unit.unit_id}"
            ).encode()
        )
        for segment in unit.segments:
            digest.update(segment.chunk_id.encode("utf-8"))
            digest.update(hashlib.sha256(segment.text.encode("utf-8")).digest())
        return digest.hexdigest()

    def _input_hash(
        self,
        name: str,
        mode: PersonaMode,
        source_hash: str,
        *,
        control_source_hash: str = "",
        domain: str = "",
        options: DistillationOptions = LEGACY_DISTILLATION_OPTIONS,
    ) -> str:
        if options.preset == "legacy":
            map_value = (
                f"{self.LEGACY_MAP_PIPELINE_VERSION}|{name}|{mode}|{source_hash}|"
                f"{self.output_language}|target|{domain.strip()}"
            )
            map_hash = hashlib.sha256(map_value.encode("utf-8")).hexdigest()
            value = (
                f"{self.LEGACY_REDUCE_PIPELINE_VERSION}|{map_hash}|"
                f"{control_source_hash}|{domain.strip()}"
            )
            return hashlib.sha256(value.encode("utf-8")).hexdigest()
        value = (
            f"{self.REDUCE_PIPELINE_VERSION}|{self.MAP_PIPELINE_VERSION}|{name}|{mode}|"
            f"{source_hash}|{control_source_hash}|{domain.strip()}|{options.cache_key}"
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _run_maps(
        self,
        *,
        run_id: str,
        label: str,
        mode: PersonaMode,
        units: tuple[SourceUnit, ...],
        progress: ProgressCallback,
        check_cancelled: CancellationCheck,
        corpus_role: str = "target",
        domain: str = "",
        progress_start: int = 5,
        progress_end: int = 65,
        reuse_persona_id: str | None = None,
    ) -> list[MapResult]:
        results: dict[str, MapResult] = {}
        missing: list[SourceUnit] = []
        input_hashes = {
            unit.unit_id: self._map_input_hash(
                label,
                mode,
                unit,
                self.output_language,
                corpus_role,
                domain,
            )
            for unit in units
        }
        for unit in units:
            check_cancelled()
            mapped = self.repository.get_map_result(run_id, unit.unit_id)
            if mapped is None and reuse_persona_id:
                mapped = self.repository.find_compatible_map_result(
                    input_hash=input_hashes[unit.unit_id],
                    unit_id=unit.unit_id,
                    persona_id=reuse_persona_id,
                )
                if mapped is not None and not self._map_matches_unit(mapped, unit):
                    mapped = None
                if mapped is not None:
                    self._save_map(run_id, unit, input_hashes[unit.unit_id], mapped)
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
                map_input_hashes=input_hashes,
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
        map_input_hashes: dict[str, str],
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
                        self._save_map(
                            run_id,
                            unit,
                            map_input_hashes[unit.unit_id],
                            mapped,
                        )
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

    @staticmethod
    def _map_matches_unit(mapped: MapResult, unit: SourceUnit) -> bool:
        if mapped.unit_id != unit.unit_id:
            return False
        allowed = {segment.chunk_id for segment in unit.segments}
        referenced = {
            evidence.chunk_id
            for candidate in mapped.mental_candidates
            for evidence in candidate.evidence
        }
        referenced.update(
            evidence.chunk_id
            for candidate in mapped.heuristic_candidates
            for evidence in candidate.evidence
        )
        referenced.update(
            evidence.chunk_id
            for tension in mapped.tensions
            for evidence in tension.evidence
        )
        return referenced.issubset(allowed)

    @staticmethod
    def _apply_quality_scope(
        persona: PersonaSpec,
        options: DistillationOptions,
    ) -> PersonaSpec:
        """Prevent disabled validation stages from being presented as completed."""

        def scope(model):
            validation = model.validation.model_copy(
                update={
                    "generative": (
                        model.validation.generative if options.generative_validation else False
                    ),
                    "exclusive": (
                        model.validation.exclusive if options.exclusivity_validation else False
                    ),
                }
            )
            updates = {"validation": validation}
            if not options.exclusivity_validation:
                updates["specificity"] = "unverified"
            if not options.cross_document_validation:
                updates["attribution_scope"] = "uncertain"
            return model.model_copy(update=updates)

        return persona.model_copy(
            update={
                "schema_version": max(4, persona.schema_version),
                "distillation_options": options,
                "mental_models": [scope(model) for model in persona.mental_models],
                "academic_conventions": [
                    scope(model) for model in persona.academic_conventions
                ],
            }
        )

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
