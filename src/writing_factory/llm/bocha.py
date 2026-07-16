"""Typed, cached Bocha Web Search API client."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, SecretStr

from writing_factory.config import Settings
from writing_factory.llm.base import ServiceTransport
from writing_factory.store import Database


class BochaWebPage(BaseModel):
    """One normalized result from ``data.webPages.value``."""

    model_config = ConfigDict(frozen=True)

    result_id: str = ""
    title: str
    url: str
    snippet: str = ""
    summary: str = ""
    site_name: str = ""
    date_published: str = ""
    language: str = ""


class BochaSearchResult(BaseModel):
    """Normalized web-search response used by retrieval adapters."""

    model_config = ConfigDict(frozen=True)

    query: str
    pages: tuple[BochaWebPage, ...] = ()
    total_estimated_matches: int | None = None


class BochaClient:
    """Expose Bocha search without leaking provider HTTP into business modules."""

    def __init__(self, settings: Settings, database: Database) -> None:
        self.transport = ServiceTransport(
            provider="bocha",
            base_url=settings.bocha_base_url,
            credential=settings.bocha_api_key,
            database=database,
            connect_timeout_seconds=settings.connect_timeout_seconds,
            read_timeout_seconds=settings.read_timeout_seconds,
            max_retries=settings.max_retries,
            minimum_interval_seconds=settings.min_request_interval_seconds,
        )

    def close(self) -> None:
        self.transport.close()

    def configure(
        self,
        *,
        credential: SecretStr | None = None,
        base_url: str | None = None,
    ) -> None:
        if credential is not None:
            self.transport.set_credential(credential)
        if base_url is not None:
            self.transport.set_base_url(base_url)

    def search(
        self,
        query: str,
        *,
        count: int = 5,
        check_cancelled: Callable[[], None] | None = None,
    ) -> BochaSearchResult:
        """Search the public web and retain only traceable webpage metadata."""

        normalized = query.strip()
        if not normalized:
            raise ValueError("联网检索词不能为空")
        if not 1 <= count <= 20:
            raise ValueError("博查每次联网检索条目数必须在 1 至 20 之间")

        def validate(response: dict) -> None:
            if response.get("code") != 200:
                raise ValueError(f"博查检索失败: {response.get('msg') or '未知错误'}")
            values = response.get("data", {}).get("webPages", {}).get("value", [])
            if not isinstance(values, list):
                raise ValueError("博查响应缺少 data.webPages.value")

        response = self.transport.request_json(
            "POST",
            "/web-search",
            operation="web_search",
            payload={
                "query": normalized,
                "freshness": "noLimit",
                "summary": True,
                "count": count,
            },
            prompt_summary={"query_chars": len(normalized), "count": count},
            use_cache=True,
            response_validator=validate,
            check_cancelled=check_cancelled,
        )
        validate(response)
        web_pages = response.get("data", {}).get("webPages", {})
        pages = tuple(
            BochaWebPage(
                result_id=str(item.get("id", "")),
                title=str(item.get("name", "")).strip() or "未命名网页",
                url=str(item.get("url", "")).strip(),
                snippet=str(item.get("snippet", "")).strip(),
                summary=str(item.get("summary", "")).strip(),
                site_name=str(item.get("siteName", "")).strip(),
                date_published=str(item.get("datePublished", "")).strip(),
                language=str(item.get("language", "")).strip(),
            )
            for item in web_pages.get("value", [])
            if isinstance(item, dict) and str(item.get("url", "")).strip()
        )
        total = web_pages.get("totalEstimatedMatches")
        return BochaSearchResult(
            query=normalized,
            pages=pages,
            total_estimated_matches=total if isinstance(total, int) else None,
        )
