"""生成阶段隔离作者蒸馏语料与当前任务事实证据。"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from writing_factory.distill.models import PersonaSpec


class GenerationSourcePolicy(BaseModel):
    """明确记录本次任务可用于事实检索的文档白名单。"""

    model_config = ConfigDict(frozen=True)

    allowed_task_doc_ids: set[str] = Field(default_factory=set)
    excluded_persona_doc_ids: set[str] = Field(default_factory=set)

    def permits(self, doc_id: str) -> bool:
        """只有当前任务白名单中的文档才能支持事实和引用。"""

        return doc_id in self.allowed_task_doc_ids


def build_generation_source_policy(
    *,
    persona: PersonaSpec,
    selected_task_doc_ids: Iterable[str],
    explicitly_allowed_persona_doc_ids: Iterable[str] = (),
) -> GenerationSourcePolicy:
    """默认排除全部蒸馏来源，用户明确复用时才重新放入白名单。"""

    persona_ids = {item.doc_id for item in persona.source_info}
    explicit = set(explicitly_allowed_persona_doc_ids) & persona_ids
    selected = set(selected_task_doc_ids)
    excluded = persona_ids - explicit
    return GenerationSourcePolicy(
        allowed_task_doc_ids=(selected - excluded) | (selected & explicit),
        excluded_persona_doc_ids=excluded,
    )


def find_suspicious_source_overlap(
    draft: str,
    persona_source_texts: Iterable[str],
    *,
    minimum_characters: int = 24,
) -> list[str]:
    """查找与蒸馏语料连续重合的中文片段，供生成后核对器复查。"""

    if minimum_characters < 8:
        raise ValueError("相似片段门槛不能低于 8 个字符")
    normalized_draft = _normalize(draft)
    if len(normalized_draft) < minimum_characters:
        return []
    draft_windows = {
        normalized_draft[index : index + minimum_characters]
        for index in range(len(normalized_draft) - minimum_characters + 1)
    }
    matches: list[str] = []
    for source in persona_source_texts:
        normalized_source = _normalize(source)
        for index in range(len(normalized_source) - minimum_characters + 1):
            window = normalized_source[index : index + minimum_characters]
            if window in draft_windows:
                matches.append(window)
                break
    return list(dict.fromkeys(matches))


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()
