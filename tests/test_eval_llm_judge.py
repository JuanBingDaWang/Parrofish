"""Tests for LLM-as-judge evaluator — no live API calls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from writing_factory.eval.llm_judge import LLMJudge, judge_draft
from writing_factory.eval.models import JudgeResult


class _MockResult:
    def __init__(self, content: str) -> None:
        self.content = content


def _judge_json_response(
    scores: list[int] | None = None,
) -> str:
    """Generate a realistic judge JSON response."""

    if scores is None:
        scores = [4, 4, 3, 4, 4]
    dims = ["论点清晰度", "论证质量", "结构与组织", "事实整合", "文风与表达"]
    dimensions = [
        {"dimension": d, "score": s, "rationale": f"{d}的评分理由"}
        for d, s in zip(dims, scores, strict=True)
    ]
    return json.dumps(
        {
            "dimensions": dimensions,
            "judge_rationale": "总体评价：论文质量良好，论证清晰。",
        },
        ensure_ascii=False,
    )


def _mock_client(response_json: str) -> MagicMock:
    client = MagicMock()
    client.chat.return_value = _MockResult(response_json)
    return client


def test_judge_returns_expected_structure() -> None:
    """Standard judge evaluation should return a properly structured JudgeResult."""

    client = _mock_client(_judge_json_response())
    judge = LLMJudge(client, shuffle_dimensions=False)

    result = judge.evaluate(
        thesis="数字人文对传统人文学科的方法论重构。",
        draft="本文探讨数字人文方法论的影响...（正文略）",
    )

    assert isinstance(result, JudgeResult)
    assert len(result.dimensions) == 5
    assert 1 <= result.overall_score <= 5
    assert result.judge_rationale


def test_judge_all_perfect_scores() -> None:
    """All-5 scores should yield overall 5.0."""

    client = _mock_client(_judge_json_response([5, 5, 5, 5, 5]))
    judge = LLMJudge(client, shuffle_dimensions=False)

    result = judge.evaluate(thesis="好论点。", draft="好文章。")

    assert result.overall_score == 5.0
    assert all(d.score == 5 for d in result.dimensions)


def test_judge_with_persona_spec() -> None:
    """Judge should handle persona_spec gracefully."""

    client = _mock_client(_judge_json_response())
    judge = LLMJudge(client, shuffle_dimensions=False)

    result = judge.evaluate(
        thesis="测试论点。",
        draft="测试文章。",
        persona_spec=(
            "这是一份 PersonaSpec 的内容，描述了一位福柯式话语分析者的思维方式和表达习惯。"
        ),
    )

    assert isinstance(result, JudgeResult)
    assert len(result.dimensions) == 5


def test_judge_handles_json_in_markdown_fence() -> None:
    """Judge should strip Markdown code fences from LLM response."""

    raw = f"```json\n{_judge_json_response()}\n```"
    client = _mock_client(raw)
    judge = LLMJudge(client, shuffle_dimensions=False)

    result = judge.evaluate(thesis="论点", draft="文章")

    assert isinstance(result, JudgeResult)
    assert result.overall_score > 0


def test_judge_fallback_on_parse_error() -> None:
    """When LLM returns invalid JSON, the result must fail closed."""

    client = _mock_client("这不是有效的JSON，完全无法解析{{{")
    judge = LLMJudge(client, shuffle_dimensions=False)

    result = judge.evaluate(thesis="论点", draft="文章")

    assert isinstance(result, JudgeResult)
    assert result.overall_score == 1.0
    assert result.evaluation_error is not None
    assert len(result.dimensions) == 5


def test_judge_with_empty_draft() -> None:
    """Empty draft should still produce a result without crashing."""

    client = _mock_client(_judge_json_response([3, 3, 3, 3, 3]))
    judge = LLMJudge(client, shuffle_dimensions=False)

    result = judge.evaluate(thesis="论点", draft="")

    assert isinstance(result, JudgeResult)
    assert result.overall_score == 3.0


def test_convenience_function() -> None:
    """judge_draft should work as expected."""

    client = _mock_client(_judge_json_response())
    result = judge_draft(client, thesis="论点", draft="文章")

    assert isinstance(result, JudgeResult)
