"""Recoverable map-reduce orchestration for person and topic PersonaSpecs."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import date

from writing_factory.distill.expression import ExpressionAnalyzer
from writing_factory.distill.extraction import PersonaMapExtractor
from writing_factory.distill.models import DistillationOutcome, MapResult, PersonaMode
from writing_factory.distill.quality import run_static_quality_check
from writing_factory.distill.serialization import render_persona_markdown
from writing_factory.distill.sources import SourceCorpusBuilder
from writing_factory.distill.synthesis import PersonaSynthesizer
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

    PIPELINE_VERSION = "persona-v2"

    def __init__(
        self,
        repository: PersonaRepository,
        sources: SourceCorpusBuilder,
        extractor: PersonaMapExtractor,
        synthesizer: PersonaSynthesizer,
        expression: ExpressionAnalyzer | None = None,
    ) -> None:
        self.repository = repository
        self.sources = sources
        self.extractor = extractor
        self.synthesizer = synthesizer
        self.expression = expression or ExpressionAnalyzer()

    def distill(
        self,
        *,
        kb_id: str,
        name: str,
        mode: PersonaMode,
        doc_ids: set[str] | None = None,
        progress: ProgressCallback = _no_progress,
        check_cancelled: CancellationCheck = _no_cancellation,
    ) -> DistillationOutcome:
        """Distill or resume one profile from ready KB documents."""

        label = name.strip()
        if not label:
            raise ValueError("Persona name cannot be empty")
        progress(2, "读取蒸馏语料")
        corpus = self.sources.build(kb_id, doc_ids=doc_ids)
        input_hash = self._input_hash(label, mode, corpus.source_hash)
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
            map_total=len(corpus.units),
        )
        try:
            map_results: list[MapResult] = []
            for index, unit in enumerate(corpus.units, start=1):
                check_cancelled()
                mapped = self.repository.get_map_result(run.run_id, unit.unit_id)
                if mapped is None:
                    mapped = self.extractor.extract(label, mode, unit)
                    self.repository.save_map_result(
                        run_id=run.run_id,
                        unit_id=unit.unit_id,
                        input_hash=input_hash,
                        chunk_ids=[segment.chunk_id for segment in unit.segments],
                        result=mapped,
                    )
                map_results.append(mapped)
                progress(5 + round(60 * index / len(corpus.units)), "提取思维候选")
            check_cancelled()
            self.repository.update_stage(run.run_id, run.persona_id, "reducing")
            texts = [segment.text for unit in corpus.units for segment in unit.segments]
            expression = self.expression.analyze(texts)
            progress(70, "三重验证与归并")
            persona = self.synthesizer.synthesize(
                persona_id=run.persona_id,
                name=label,
                mode=mode,
                map_results=map_results,
                units=corpus.units,
                source_info=corpus.source_info,
                expression=expression,
                research_date=date.today(),
            )
            check_cancelled()
            self.repository.update_stage(run.run_id, run.persona_id, "validating")
            progress(92, "生成确定性档案")
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
    def _input_hash(cls, name: str, mode: PersonaMode, source_hash: str) -> str:
        value = f"{cls.PIPELINE_VERSION}|{name}|{mode}|{source_hash}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
