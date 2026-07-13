"""Query rewriting and HyDE for academic retrieval quality.

Academic queries are often abstract ("政策驱动的问题化机制"). Rewriting expands a
single query into concrete sub-questions, and HyDE first asks the model to write a
hypothetical answer whose embedding is then used for retrieval. Both call the
unified SiliconFlow client (non-thinking, low temperature) and share the global
concurrency gate, so they never bypass cost control or the UI thread boundary.
"""

from __future__ import annotations

import json
import re

from writing_factory.llm import SiliconFlowClient

REWRITE_SYSTEM = (
    "你是人文社科文献检索的查询扩展助手。"
    "给定用户的研究问题，产出 3 到 5 个更具体、可检索的子查询。"
    "每个子查询应聚焦一个不同角度（概念界定、因果机制、经验证据、理论对话、批评与边界）。"
    '只输出 JSON：{"queries": ["...", "..."]}，不要任何解释。'
)

HYDE_SYSTEM = (
    "你是一位严谨的人文社科学者。针对下面的研究问题，写一段简短的、"
    "假设性的回答要点（2-4 句），仿佛你正在论文中论述它。只输出论述文本本身，不要前缀或解释。"
)


class QueryExpander:
    """Expand one query into sub-queries and an optional HyDE hypothesis."""

    def __init__(self, siliconflow: SiliconFlowClient) -> None:
        self.siliconflow = siliconflow

    def rewrite(self, query: str, *, use_cache: bool = True) -> list[str]:
        """Return 3-5 concrete sub-queries derived from the original question."""

        if not query.strip():
            return []
        result = self.siliconflow.chat(
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": query},
            ],
            thinking=False,
            temperature=0.2,
            max_tokens=512,
            seed=42,
            response_format="json_object",
            use_cache=use_cache,
            priority=20,
        )
        parsed = self._parse_queries(result.content)
        return parsed or [query]

    def hyde_passage(self, query: str, *, use_cache: bool = True) -> str | None:
        """Generate a hypothetical answer passage used only for its embedding."""

        if not query.strip():
            return None
        result = self.siliconflow.chat(
            [
                {"role": "system", "content": HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
            thinking=False,
            temperature=0.3,
            max_tokens=256,
            seed=42,
            use_cache=use_cache,
            priority=20,
        )
        passage = result.content.strip()
        return passage or None

    @staticmethod
    def _parse_queries(content: str) -> list[str]:
        """Extract a clean list of query strings from a JSON or loose response."""

        content = content.strip()
        if not content:
            return []
        try:
            data = json.loads(content)
            queries = data.get("queries") if isinstance(data, dict) else []
            if not isinstance(queries, list):
                return []
        except (json.JSONDecodeError, ValueError):
            found = re.findall(r"[\"“]([^\"”]+)[\"”]", content)
            queries = found or ([content] if content else [])
        cleaned = [q.strip().strip("。") for q in queries if isinstance(q, str) and q.strip()]
        return cleaned[:5]
