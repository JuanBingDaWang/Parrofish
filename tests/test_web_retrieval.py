"""Offline contracts for optional Bocha web evidence and code-owned citations."""

from __future__ import annotations

from writing_factory.generate.drafting import _build_evidence_pack
from writing_factory.generate.models import OutlineNode
from writing_factory.kb.models import MetadataFilter, RetrievalRequest
from writing_factory.kb.web_retrieval import WebAugmentedRetriever
from writing_factory.llm.bocha import BochaSearchResult, BochaWebPage
from writing_factory.orchestration.reference_assembler import assemble_reference_list


class EmptyRepository:
    def ready_child_chunks_by_ids(self, _kb_id: str, _chunk_ids: set[str]):
        return []

    def get_bibliographies(self, _kb_id: str, _doc_ids: set[str]):
        return {}


class RejectingLocalRetriever:
    def __init__(self) -> None:
        self.repository = EmptyRepository()
        self.calls = 0

    def search(self, _request, **_kwargs):
        self.calls += 1
        raise AssertionError("空本地白名单不应调用本地检索器")


class FakeBocha:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, count: int, check_cancelled=None):
        if check_cancelled is not None:
            check_cancelled()
        self.calls.append((query, count))
        return BochaSearchResult(
            query=query,
            pages=(
                BochaWebPage(
                    title="数字出版公共服务的新动向",
                    url="https://example.org/publishing",
                    summary="公开网页介绍了数字出版公共服务的近期实践。",
                    site_name="示例研究网",
                    date_published="2026-06-18",
                ),
            ),
        )


def test_web_only_retrieval_freezes_traceable_evidence_and_gbt_reference() -> None:
    local = RejectingLocalRetriever()
    bocha = FakeBocha()
    retriever = WebAugmentedRetriever(local, bocha, result_count=6)
    request = RetrievalRequest(
        kb_id="kb",
        query="数字出版公共服务",
        filters=MetadataFilter(doc_ids=set()),
        use_rewrite=False,
        use_hyde=False,
    )

    result = retriever.search(request)
    pack = _build_evidence_pack(
        outline_node=OutlineNode(
            node_id="1",
            heading="公共服务",
            rhetorical_purpose="分析可能路径",
        ),
        retrieval_result=result,
        repository=local.repository,
        kb_id="kb",
    )
    references = assemble_reference_list(
        list(pack.items),
        citation_style="gb-t-7714",
        kb_repository=local.repository,
        kb_id="kb",
    )

    assert local.calls == 0
    assert bocha.calls == [("数字出版公共服务", 6)]
    assert result.hits[0].source == "web"
    assert pack.items[0].source_type == "web"
    assert pack.items[0].url == "https://example.org/publishing"
    assert references.items[0].url == "https://example.org/publishing"
    assert "数字出版公共服务的新动向" in references.items[0].citation_text
    assert "https://example.org/publishing" in references.items[0].citation_text
