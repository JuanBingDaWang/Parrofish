"""Evaluation runner — entry point for running all or selected evaluations.

Provides a CLI-style interface to:
- Run faithfulness evaluation on generated drafts
- Run LLM-as-judge evaluation on generated drafts
- Run golden regression suites
- Run injection detection on untrusted inputs
- Persist evaluation results to the database

Usage:
    from writing_factory.eval.run_eval import EvaluationRunner

    runner = EvaluationRunner(siliconflow, database)
    result = runner.evaluate_faithfulness(
        question="...", answer="...", context=[...]
    )
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from writing_factory.eval.faithfulness import FaithfulnessEvaluator
from writing_factory.eval.golden import GoldenRunner
from writing_factory.eval.injection import InjectionDetector
from writing_factory.eval.llm_judge import LLMJudge
from writing_factory.eval.models import (
    CitationTraceabilityResult,
    FaithfulnessResult,
    GoldenSuite,
    GoldenSuiteResult,
    InjectionVerdict,
    JudgeResult,
)
from writing_factory.eval.traceability import evaluate_citation_traceability

if TYPE_CHECKING:
    from writing_factory.llm.siliconflow import SiliconFlowClient
    from writing_factory.store.database import Database


class EvaluationRunner:
    """Main entry point for running evaluations and persisting results.

    Encapsulates all evaluators and database persistence.
    """

    def __init__(
        self,
        siliconflow: SiliconFlowClient,
        database: Database,
    ) -> None:
        self._faithfulness = FaithfulnessEvaluator(siliconflow)
        self._judge = LLMJudge(siliconflow)
        self._golden = GoldenRunner(self._faithfulness, self._judge)
        self._injection = InjectionDetector()
        self._database = database

    # ── Faithfulness ──

    def evaluate_traceability(
        self,
        state: dict,
        *,
        persist: bool = False,
        kb_id: str | None = None,
        pipeline_run_id: str | None = None,
    ) -> CitationTraceabilityResult:
        """Run the mandatory code-only citation safety metric."""

        result = evaluate_citation_traceability(state)
        if persist:
            self._save_evaluation(
                evaluation_type="citation_traceability",
                score=result.verified_support_ratio,
                pass_threshold=1.0,
                result_json=result.model_dump_json(),
                kb_id=kb_id,
                pipeline_run_id=pipeline_run_id,
            )
        return result

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: list[str],
        *,
        persist: bool = False,
        kb_id: str | None = None,
        pipeline_run_id: str | None = None,
    ) -> FaithfulnessResult:
        """Run faithfulness evaluation and optionally persist."""

        result = self._faithfulness.evaluate(question, answer, context)

        if persist:
            self._save_evaluation(
                evaluation_type="faithfulness",
                score=result.score,
                pass_threshold=0.8,
                result_json=result.model_dump_json(),
                kb_id=kb_id,
                task_description=question[:500],
                pipeline_run_id=pipeline_run_id,
            )

        return result

    # ── LLM-as-Judge ──

    def evaluate_judge(
        self,
        thesis: str,
        draft: str,
        persona_spec: str | None = None,
        *,
        persist: bool = False,
        kb_id: str | None = None,
        pipeline_run_id: str | None = None,
    ) -> JudgeResult:
        """Run LLM-as-judge evaluation and optionally persist."""

        result = self._judge.evaluate(thesis, draft, persona_spec)

        if persist:
            self._save_evaluation(
                evaluation_type="judge",
                score=result.overall_score,
                pass_threshold=3.0,
                result_json=result.model_dump_json(),
                kb_id=kb_id,
                task_description=thesis[:500],
                pipeline_run_id=pipeline_run_id,
            )

        return result

    # ── Golden Regression ──

    def run_golden_suite(
        self,
        suite: GoldenSuite,
        output_provider,
        *,
        persist: bool = False,
        pipeline_run_id: str | None = None,
    ) -> GoldenSuiteResult:
        """Run a golden regression suite and optionally persist."""

        result = self._golden.run_suite(suite, output_provider)

        if persist:
            self._save_evaluation(
                evaluation_type="golden",
                score=result.pass_rate,
                pass_threshold=0.8,
                result_json=result.model_dump_json(),
                task_description=f"golden_suite:{suite.suite_name}",
                pipeline_run_id=pipeline_run_id,
            )

        return result

    # ── Injection Detection ──

    def check_injection(
        self,
        text: str,
        *,
        use_llm: bool = False,
    ) -> InjectionVerdict:
        """Run injection detection on untrusted text."""

        verdict = self._injection.check(text)
        if use_llm and verdict.risk_level == "medium":
            verdict = self._injection.check_with_llm(
                self._faithfulness._client,
                text,  # type: ignore[arg-type]
            )
        return verdict

    # ── Persistence ──

    def _save_evaluation(
        self,
        *,
        evaluation_type: str,
        score: float,
        pass_threshold: float | None,
        result_json: str,
        kb_id: str | None = None,
        task_description: str | None = None,
        pipeline_run_id: str | None = None,
    ) -> None:
        """Persist an evaluation result to the database."""

        try:
            from writing_factory.store.database import utc_now

            now = utc_now()
            evaluation_id = str(uuid.uuid4())

            with self._database.connection() as connection:
                connection.execute(
                    """
                    INSERT INTO generation_evaluations (
                        evaluation_id, kb_id, task_description, pipeline_run_id,
                        evaluation_type, score, pass_threshold, passed,
                        result_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evaluation_id,
                        kb_id,
                        task_description,
                        pipeline_run_id,
                        evaluation_type,
                        score,
                        pass_threshold,
                        1 if pass_threshold is None or score >= pass_threshold else 0,
                        result_json,
                        now,
                    ),
                )
        except Exception:
            logging.getLogger(__name__).exception("保存生成评估结果失败")
