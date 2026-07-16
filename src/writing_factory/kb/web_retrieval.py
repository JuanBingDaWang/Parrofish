"""Optional Bocha web results merged with the existing local hybrid retriever."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256

from writing_factory.kb.models import FusedHit, RetrievalRequest, RetrievalResult
from writing_factory.llm.bocha import BochaClient, BochaWebPage


class WebAugmentedRetriever:
    """Interleave local and web evidence while preserving the local retriever contract."""

    def __init__(
        self,
        local_retriever,
        bocha: BochaClient,
        *,
        result_count: int,
    ) -> None:
        self.local_retriever = local_retriever
        self.bocha = bocha
        self.result_count = result_count
        self.repository = local_retriever.repository

    def search(
        self,
        request: RetrievalRequest,
        *,
        progress: Callable[[int, str], None] = lambda _percent, _message: None,
        check_cancelled: Callable[[], None] = lambda: None,
        **kwargs,
    ) -> RetrievalResult:
        """Run local retrieval when possible, then add cached Bocha web results."""

        filters = request.filters
        has_local_scope = filters is None or filters.doc_ids is None or bool(filters.doc_ids)
        if has_local_scope:
            local = self.local_retriever.search(
                request,
                progress=lambda percent, message: progress(
                    min(65, round(percent * 0.65)), message
                ),
                check_cancelled=check_cancelled,
                **kwargs,
            )
        else:
            local = RetrievalResult(query=request.query)
            progress(65, "本次没有本地事实语料")

        check_cancelled()
        progress(70, "博查联网检索")
        web = self.bocha.search(
            request.query,
            count=self.result_count,
            check_cancelled=check_cancelled,
        )
        check_cancelled()
        web_hits = [_page_to_hit(page, index) for index, page in enumerate(web.pages, 1)]
        merged = _interleave(list(local.hits), web_hits)
        ranked = tuple(
            hit.model_copy(update={"final_rank": index})
            for index, hit in enumerate(merged, 1)
        )
        progress(100, f"联网检索完成 · {len(web_hits)} 条")
        return RetrievalResult(
            query=request.query,
            expanded_queries=local.expanded_queries,
            hits=ranked,
        )


def _page_to_hit(page: BochaWebPage, rank: int) -> FusedHit:
    identity = sha256(page.url.encode("utf-8")).hexdigest()[:20]
    body = page.summary or page.snippet
    if page.summary and page.snippet and page.snippet not in page.summary:
        body = f"{page.summary}\n{page.snippet}"
    text = f"网页标题：{page.title}\n网页摘要：{body}".strip()[:6000]
    return FusedHit(
        chunk_id=f"web_chunk_{identity}",
        doc_id=f"web_doc_{identity}",
        text=text,
        source="web",
        rrf_score=1 / (60 + rank),
        final_rank=rank,
        section_heading="联网搜索摘要",
        title=page.title,
        url=page.url,
        site_name=page.site_name,
        date_published=page.date_published,
    )


def _interleave(local: list[FusedHit], web: list[FusedHit]) -> list[FusedHit]:
    merged: list[FusedHit] = []
    for index in range(max(len(local), len(web))):
        if index < len(local):
            merged.append(local[index])
        if index < len(web):
            merged.append(web[index])
    return merged
