"""Golden regression set framework for generation quality regression testing.

Maintains a curated set of "task → expected quality" test cases (10-20 cases)
that are run after every pipeline change to guard against regressions.

Usage:
    runner = GoldenRunner(faithfulness_evaluator, llm_judge)
    suite = GoldenSuite.model_validate(json.loads(file.read_text()))
    result = runner.run_suite(suite, pipeline_outputs)
    assert result.overall_pass, "Golden regression suite failed!"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from writing_factory.eval.faithfulness import FaithfulnessEvaluator
from writing_factory.eval.llm_judge import LLMJudge
from writing_factory.eval.models import (
    GoldenCase,
    GoldenRunResult,
    GoldenSuite,
    GoldenSuiteResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class GoldenRunner:
    """Runs golden regression suites against pipeline outputs.

    The runner requires two evaluators (faithfulness + judge) and a factory
    function that produces pipeline outputs given a task description.
    """

    def __init__(
        self,
        faithfulness_evaluator: FaithfulnessEvaluator,
        llm_judge: LLMJudge,
    ) -> None:
        self._faithfulness = faithfulness_evaluator
        self._judge = llm_judge

    def run_suite(
        self,
        suite: GoldenSuite,
        output_provider: Callable[[str], dict[str, str]],
    ) -> GoldenSuiteResult:
        """Run a golden suite.

        Args:
            suite: The GoldenSuite to evaluate.
            output_provider: Callable that takes a task_description and returns
                a dict with keys: 'thesis', 'draft', 'context' (list of str),
                optionally 'persona_spec'.

        Returns:
            GoldenSuiteResult with per-case and aggregated results.
        """

        case_results: list[GoldenRunResult] = []

        for case in suite.cases:
            result = self._run_case(case, output_provider)
            case_results.append(result)

        passed = sum(1 for r in case_results if r.all_passed)
        total = len(case_results)
        pass_rate = passed / total if total > 0 else 0.0

        return GoldenSuiteResult(
            suite_name=suite.suite_name,
            case_results=case_results,
            pass_rate=pass_rate,
            overall_pass=pass_rate >= 0.8,  # 80% pass threshold for suite
        )

    def _run_case(
        self,
        case: GoldenCase,
        output_provider: Callable[[str], dict[str, str]],
    ) -> GoldenRunResult:
        """Evaluate a single golden case."""

        outputs = output_provider(case.task_description)
        thesis = outputs.get("thesis", "")
        draft = outputs.get("draft", "")
        context_raw = outputs.get("context", "[]")
        persona_spec = outputs.get("persona_spec")

        # Parse context
        if isinstance(context_raw, str):
            try:
                context = json.loads(context_raw)
            except (json.JSONDecodeError, TypeError):
                context = [context_raw]
        else:
            context = list(context_raw)

        actual_scores: dict[str, float] = {}
        failures: list[str] = []

        # Evaluate faithfulness if context is available
        if context and draft:
            try:
                faith_result = self._faithfulness.evaluate(
                    question=case.task_description,
                    answer=draft,
                    context=context,
                )
                actual_scores["faithfulness"] = faith_result.score
                if (
                    "faithfulness" in case.expected_min_scores
                    and faith_result.score < case.expected_min_scores["faithfulness"]
                ):
                    failures.append(
                        f"faithfulness: got {faith_result.score:.2f}, "
                        f"expected >= {case.expected_min_scores['faithfulness']}"
                    )
            except Exception as exc:
                failures.append(f"faithfulness evaluation error: {exc}")

        # Evaluate judge dimensions
        if draft and thesis:
            try:
                judge_result = self._judge.evaluate(
                    thesis=thesis,
                    draft=draft,
                    persona_spec=persona_spec,
                )
                for dim in judge_result.dimensions:
                    dim_key = dim.dimension.replace(" ", "_").lower()
                    actual_scores[dim_key] = dim.score
                    if (
                        dim_key in case.expected_min_scores
                        and dim.score < case.expected_min_scores[dim_key]
                    ):
                        failures.append(
                            f"{dim_key}: got {dim.score}, "
                            f"expected >= {case.expected_min_scores[dim_key]}"
                        )
                actual_scores["overall"] = judge_result.overall_score
            except Exception as exc:
                failures.append(f"judge evaluation error: {exc}")

        # Also check if expected keys are missing
        for key in case.expected_min_scores:
            if key not in actual_scores:
                failures.append(f"{key}: missing from evaluation results")

        return GoldenRunResult(
            case_id=case.case_id,
            actual_scores=actual_scores,
            all_passed=len(failures) == 0,
            failures=failures,
        )


# ── File helpers ──


def load_golden_suite(path: Path) -> GoldenSuite:
    """Load a GoldenSuite from a JSON file."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return GoldenSuite.model_validate(data)


def save_golden_suite(suite: GoldenSuite, path: Path) -> None:
    """Save a GoldenSuite to a JSON file."""

    path.write_text(
        json.dumps(suite.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
