"""LLM-as-judge evaluator for soft quality dimensions.

Supports the five dimensions defined in the project brief:
- 论点清晰度
- 论证质量
- 结构与组织
- 事实整合
- 文风与表达

Includes position bias mitigation via randomized judgment.
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING

from writing_factory.eval.models import JudgeDimension, JudgeResult
from writing_factory.eval.prompts import (
    JUDGE_SYSTEM_TEMPLATE,
    JUDGE_USER_TEMPLATE,
)

if TYPE_CHECKING:
    from writing_factory.llm.siliconflow import SiliconFlowClient

# Default rubric — used when no persona is provided
_DEFAULT_DIMENSIONS_ORDER = [
    "论点清晰度",
    "论证质量",
    "结构与组织",
    "事实整合",
    "文风与表达",
]


class LLMJudge:
    """LLM-as-judge for evaluating nonfiction writing quality.

    Usage:
        judge = LLMJudge(siliconflow)
        result = judge.evaluate(
            thesis="核心论点...",
            draft="论文全文...",
            persona_spec="可选：persona spec 用于风格评估",
        )
        print(result.overall_score)
    """

    def __init__(
        self,
        client: SiliconFlowClient,
        *,
        shuffle_dimensions: bool = True,
    ) -> None:
        self._client = client
        self._shuffle = shuffle_dimensions

    def evaluate(
        self,
        thesis: str,
        draft: str,
        persona_spec: str | None = None,
    ) -> JudgeResult:
        """Run LLM-as-judge evaluation.

        Args:
            thesis: The core thesis statement.
            draft: The full polished draft.
            persona_spec: Optional PersonaSpec for style fidelity assessment.

        Returns:
            JudgeResult with per-dimension scores and overall score.
        """

        # Prepare persona context if available
        persona_context = ""
        if persona_spec:
            # Truncate long persona specs to avoid token waste
            truncated = persona_spec[:1500]
            persona_context = f"\n## 作者风格参考 (PersonaSpec)\n\n{truncated}"

        # Build randomization notice
        if self._shuffle:
            randomized_order = list(_DEFAULT_DIMENSIONS_ORDER)
            random.shuffle(randomized_order)
            order_notice = (
                "注意：以下维度顺序已随机化以抵消位置偏差。"
                f"维度将按此顺序评估：{' → '.join(randomized_order)}"
            )
        else:
            order_notice = ""

        system_prompt = JUDGE_SYSTEM_TEMPLATE.format(
            randomization_notice=order_notice,
        )

        user_prompt = JUDGE_USER_TEMPLATE.format(
            thesis=thesis,
            draft=draft,
            persona_context=persona_context,
        )

        try:
            result = self._client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                thinking=False,
                temperature=0.0,
                max_tokens=8192,
                seed=42,
                response_format="json_object",
                stream=True,
                step_id="evaluation.llm_judge",
            )

            return self._parse_result(result.content or "")
        except Exception as exc:
            # Fail closed: an evaluator outage must never look like an average pass.
            return JudgeResult(
                dimensions=[
                    JudgeDimension(
                        dimension=d,
                        score=1,
                        rationale=f"评估调用失败，按失败闭合原则记为未通过：{exc}",
                    )
                    for d in _DEFAULT_DIMENSIONS_ORDER
                ],
                overall_score=1.0,
                judge_rationale=f"评估过程遇到异常，本结果不可作为质量评分：{exc}",
                evaluation_error=str(exc),
            )

    def _parse_result(self, raw: str) -> JudgeResult:
        """Parse LLM JSON response into JudgeResult."""

        # Strip Markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()

        data = json.loads(cleaned)
        dimensions = [
            JudgeDimension(
                dimension=d["dimension"],
                score=d["score"],
                rationale=d.get("rationale", ""),
            )
            for d in data.get("dimensions", [])
        ]
        overall = sum(d.score for d in dimensions) / len(dimensions) if dimensions else 3.0
        return JudgeResult(
            dimensions=dimensions,
            overall_score=round(overall, 2),
            judge_rationale=data.get("judge_rationale", ""),
        )


def judge_draft(
    client: SiliconFlowClient,
    thesis: str,
    draft: str,
    persona_spec: str | None = None,
    *,
    shuffle_dimensions: bool = True,
) -> JudgeResult:
    """Convenience function for one-shot LLM-as-judge evaluation."""

    return LLMJudge(client, shuffle_dimensions=shuffle_dimensions).evaluate(
        thesis=thesis,
        draft=draft,
        persona_spec=persona_spec,
    )
