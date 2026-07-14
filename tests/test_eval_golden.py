"""Tests for golden regression set framework — no live API calls."""

from __future__ import annotations

import json
from pathlib import Path

from writing_factory.eval.golden import (
    GoldenRunner,
    load_golden_suite,
    save_golden_suite,
)
from writing_factory.eval.models import GoldenCase, GoldenSuite


class _MockFaithfulness:
    """Mock faithfulness evaluator that returns predictable scores."""

    def evaluate(self, question, answer, context):
        from writing_factory.eval.models import FaithfulnessResult

        return FaithfulnessResult(
            score=0.85,
            atomic_claims=[],
            supported_count=8,
            unsupported_count=2,
        )


class _MockJudge:
    """Mock LLM judge that returns predictable scores."""

    def evaluate(self, thesis, draft, persona_spec=None):
        from writing_factory.eval.models import JudgeDimension, JudgeResult

        return JudgeResult(
            dimensions=[
                JudgeDimension(dimension="论点清晰度", score=4, rationale="清晰"),
                JudgeDimension(dimension="论证质量", score=4, rationale="扎实"),
                JudgeDimension(dimension="结构与组织", score=3, rationale="可接受"),
                JudgeDimension(dimension="事实整合", score=4, rationale="良好"),
                JudgeDimension(dimension="文风与表达", score=4, rationale="流畅"),
            ],
            overall_score=3.8,
            judge_rationale="总体良好。",
        )


def _dummy_output_provider(task: str) -> dict[str, str]:
    """Return canned outputs for any task."""

    return {
        "thesis": f"关于{task}的论点。",
        "draft": f"本文探讨{task}的相关问题。研究认为，这是重要议题。",
        "context": json.dumps([f"{task}的相关背景资料。", "更多上下文信息。"]),
    }


def test_golden_runner_returns_suite_result() -> None:
    """GoldenRunner.run_suite should return a properly structured result."""

    runner = GoldenRunner(_MockFaithfulness(), _MockJudge())
    suite = GoldenSuite(
        suite_name="test_suite",
        cases=[
            GoldenCase(
                case_id="case_1",
                task_description="测试任务",
                expected_min_scores={"faithfulness": 0.7, "论点清晰度": 3},
            ),
        ],
    )

    result = runner.run_suite(suite, _dummy_output_provider)

    assert result.suite_name == "test_suite"
    assert len(result.case_results) == 1
    assert result.pass_rate >= 0.0


def test_golden_case_passes_when_scores_exceed_thresholds() -> None:
    """Case should pass when all scores meet or exceed minimums."""

    runner = GoldenRunner(_MockFaithfulness(), _MockJudge())
    suite = GoldenSuite(
        suite_name="pass_test",
        cases=[
            GoldenCase(
                case_id="pass_case",
                task_description="通过测试",
                expected_min_scores={"faithfulness": 0.7, "论点清晰度": 3},
            ),
        ],
    )

    result = runner.run_suite(suite, _dummy_output_provider)

    assert result.case_results[0].all_passed
    assert len(result.case_results[0].failures) == 0


def test_golden_case_fails_when_below_threshold() -> None:
    """Case should fail when a score is below the minimum."""

    runner = GoldenRunner(_MockFaithfulness(), _MockJudge())
    suite = GoldenSuite(
        suite_name="fail_test",
        cases=[
            GoldenCase(
                case_id="fail_case",
                task_description="失败测试",
                expected_min_scores={
                    "faithfulness": 0.95,  # Mock returns 0.85 → below threshold
                },
            ),
        ],
    )

    result = runner.run_suite(suite, _dummy_output_provider)

    assert not result.case_results[0].all_passed
    assert len(result.case_results[0].failures) > 0


def test_load_golden_suite_from_file(tmp_path: Path) -> None:
    """Golden suite should load correctly from a JSON file."""

    fixture = {
        "suite_name": "loaded_suite",
        "cases": [
            {
                "case_id": "loaded_case",
                "task_description": "从文件加载",
                "expected_min_scores": {"faithfulness": 0.8},
            },
        ],
    }
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")

    suite = load_golden_suite(path)

    assert suite.suite_name == "loaded_suite"
    assert len(suite.cases) == 1
    assert suite.cases[0].case_id == "loaded_case"


def test_save_and_load_golden_suite_roundtrip(tmp_path: Path) -> None:
    """Save then load should preserve all data."""

    original = GoldenSuite(
        suite_name="roundtrip",
        cases=[
            GoldenCase(
                case_id="rt1",
                task_description="往返测试",
                expected_min_scores={"faithfulness": 0.8, "论证质量": 3},
            ),
        ],
    )
    path = tmp_path / "roundtrip.json"
    save_golden_suite(original, path)

    loaded = load_golden_suite(path)

    assert loaded.suite_name == original.suite_name
    assert loaded.cases[0].case_id == original.cases[0].case_id


def test_empty_suite_returns_zero_pass_rate() -> None:
    """An empty suite should not crash and return zero pass rate."""

    runner = GoldenRunner(_MockFaithfulness(), _MockJudge())
    suite = GoldenSuite(suite_name="empty", cases=[])

    result = runner.run_suite(suite, _dummy_output_provider)

    assert result.pass_rate == 0.0
    assert len(result.case_results) == 0
