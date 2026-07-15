"""生成阶段隔离作者蒸馏语料与当前任务事实证据。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from hashlib import sha256
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from writing_factory.distill.models import PersonaSpec
from writing_factory.eval.injection import InjectionDetector
from writing_factory.generate.models import GenerationContext
from writing_factory.kb.models import MetadataFilter

if TYPE_CHECKING:
    from writing_factory.store.persona_repository import PersonaRepository


class GenerationSourcePolicy(BaseModel):
    """明确记录本次任务可用于事实检索的文档白名单。"""

    model_config = ConfigDict(frozen=True)

    policy_id: str
    allowed_task_doc_ids: set[str] = Field(default_factory=set)
    excluded_persona_doc_ids: set[str] = Field(default_factory=set)

    def permits(self, doc_id: str) -> bool:
        """只有当前任务白名单中的文档才能支持事实和引用。"""

        return doc_id in self.allowed_task_doc_ids

    def require_nonempty(self) -> None:
        """Reject a task that has no explicitly selected factual corpus."""

        if not self.allowed_task_doc_ids:
            raise ValueError("写作任务至少需要选择一篇可作为事实来源的知识库文档")


def build_generation_source_policy(
    *,
    persona: PersonaSpec,
    selected_task_doc_ids: Iterable[str],
    explicitly_allowed_persona_doc_ids: Iterable[str] = (),
    target_persona_doc_ids: Iterable[str] | None = None,
) -> GenerationSourcePolicy:
    """默认只排除作者目标语料；对照语料仍可作为任务事实来源。

    ``target_persona_doc_ids`` 为 ``None`` 表示旧档案缺少来源角色元数据，
    此时保守回退为排除 PersonaSpec 中的全部蒸馏来源。
    """

    persona_ids = (
        {item.doc_id for item in persona.source_info}
        if target_persona_doc_ids is None
        else set(target_persona_doc_ids)
    )
    explicit = set(explicitly_allowed_persona_doc_ids) & persona_ids
    selected = set(selected_task_doc_ids)
    excluded = persona_ids - explicit
    allowed = (selected - excluded) | (selected & explicit)
    digest_input = "\n".join([*sorted(allowed), "--", *sorted(excluded)])
    return GenerationSourcePolicy(
        policy_id=f"source_policy_{sha256(digest_input.encode('utf-8')).hexdigest()[:16]}",
        allowed_task_doc_ids=allowed,
        excluded_persona_doc_ids=excluded,
    )


def build_persona_generation_source_policy(
    *,
    persona_repository: PersonaRepository,
    persona_id: str,
    selected_task_doc_ids: Iterable[str],
    explicitly_allowed_persona_doc_ids: Iterable[str] = (),
) -> GenerationSourcePolicy:
    """Build a policy from the profile plus its persisted target/control roles."""

    loaded = persona_repository.load_ready(persona_id)
    if loaded is None:
        raise ValueError(f"persona '{persona_id}' 未就绪")
    persona, _markdown = loaded
    role_loader = getattr(persona_repository, "load_source_roles", None)
    roles = role_loader(persona_id) if role_loader is not None else None
    return build_generation_source_policy(
        persona=persona,
        selected_task_doc_ids=selected_task_doc_ids,
        explicitly_allowed_persona_doc_ids=explicitly_allowed_persona_doc_ids,
        target_persona_doc_ids=roles.target_doc_ids if roles is not None else None,
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


def task_document_filter(context: GenerationContext) -> MetadataFilter:
    """Build the mandatory retrieval filter from the persisted task policy."""

    allowed = set(context.allowed_doc_ids)
    if not allowed:
        raise ValueError("写作任务没有可用的事实来源白名单")
    excluded = set(context.excluded_persona_doc_ids)
    overlap = allowed & excluded
    if overlap:
        raise ValueError(f"事实来源白名单包含被隔离的作者语料: {sorted(overlap)}")
    return MetadataFilter(doc_ids=allowed)


def enforce_retrieval_safety(retrieval_result, siliconflow) -> None:
    """Scan untrusted retrieved text before placing it in a prompt data zone."""

    content = "\n\n".join(hit.text for hit in retrieval_result.hits)
    InjectionDetector().enforce(siliconflow, content)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()
