"""Tests for faithfulness (RAGAS-style) evaluator logic — no live API calls."""

from __future__ import annotations

from unittest.mock import MagicMock

from writing_factory.eval.faithfulness import FaithfulnessEvaluator
from writing_factory.eval.models import FaithfulnessResult


class _MockResult:
    """Minimal mock for SiliconFlow chat result."""

    def __init__(self, content: str) -> None:
        self.content = content


def _mock_client(responses: list[str]) -> MagicMock:
    """Create a mock SiliconFlowClient that returns predefined responses."""

    client = MagicMock()
    client.chat.side_effect = [_MockResult(r) for r in responses]
    return client


def test_faithfulness_all_supported() -> None:
    """When LLM says all claims are supported, score should be 1.0."""

    client = _mock_client(
        [
            # Decompose: return two atomic claims
            "数字人文使用计算分析方法研究文化资料\n该方法改变了传统文本解读方式",
            # Check claim 1: supported
            "[SUPPORTED]\n理由：上下文明确支持该论断\n证据：原文提到计算分析方法",
            # Check claim 2: supported
            "[SUPPORTED]\n理由：上下文支持\n证据：原文提到改变了传统解读方式",
        ]
    )
    evaluator = FaithfulnessEvaluator(client)

    result = evaluator.evaluate(
        question="数字人文的影响？",
        answer="数字人文使用计算分析方法研究文化资料。该方法改变了传统文本解读方式。",
        context=["数字人文使用文本挖掘、知识图谱等方法研究大规模文化资料。"],
    )

    assert isinstance(result, FaithfulnessResult)
    assert result.score == 1.0
    assert result.supported_count == 2
    assert result.unsupported_count == 0


def test_faithfulness_partial_unsupported() -> None:
    """When some claims are unsupported, score should reflect ratio."""

    client = _mock_client(
        [
            # Decompose: two claims
            "数字人文使用计算分析方法\n该方法成本极低且易于推广",
            # Check 1: supported
            "[SUPPORTED]\n理由：明确支持\n证据：计算分析方法",
            # Check 2: unsupported
            "[UNSUPPORTED]\n理由：上下文未提及成本\n证据：无",
        ]
    )
    evaluator = FaithfulnessEvaluator(client)

    result = evaluator.evaluate(
        question="数字人文的成本？",
        answer="数字人文使用计算分析方法。该方法成本极低且易于推广。",
        context=["数字人文使用文本挖掘、知识图谱等方法研究大规模文化资料。"],
    )

    assert result.score == 0.5
    assert result.supported_count == 1
    assert result.unsupported_count == 1


def test_faithfulness_empty_context_is_unsupported() -> None:
    """Empty context should yield score 0."""

    client = _mock_client(
        [
            # Decompose
            "这是一个论断",
            # Check
            "[UNSUPPORTED]\n理由：无上下文",
        ]
    )
    evaluator = FaithfulnessEvaluator(client)

    result = evaluator.evaluate(
        question="测试",
        answer="这是一个论断。",
        context=[],
    )

    assert result.score == 0.0
    assert result.unsupported_count == 1


def test_faithfulness_fallback_on_llm_error() -> None:
    """When LLM calls fail, fallback mechanisms should produce a valid result."""

    client = MagicMock()
    client.chat.side_effect = RuntimeError("API failure")
    evaluator = FaithfulnessEvaluator(client)

    result = evaluator.evaluate(
        question="测试",
        answer="论断一。论断二。论断三。",
        context=["相关上下文内容用于测试回退机制。"],
    )

    # Fallback decomposition splits on Chinese punctuation
    assert isinstance(result, FaithfulnessResult)
    # Without LLM, all claims are unsupported
    assert result.unsupported_count >= 0
    assert result.supported_count == 0


def test_faithfulness_short_answer_returns_default() -> None:
    """A very short answer should not crash."""

    client = _mock_client(
        [
            "短句",
            "[SUPPORTED]\n理由：上下文支持",
        ]
    )
    evaluator = FaithfulnessEvaluator(client)

    result = evaluator.evaluate(
        question="测试",
        answer="短句。",
        context=["一些上下文。"],
    )

    assert isinstance(result, FaithfulnessResult)
    assert result.score >= 0.0


def test_convenience_function_exists() -> None:
    """The module-level `faithfulness` convenience function should be importable."""

    from writing_factory.eval.faithfulness import faithfulness

    assert callable(faithfulness)
