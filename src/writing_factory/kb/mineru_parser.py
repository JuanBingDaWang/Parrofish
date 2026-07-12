"""Recoverable MinerU upload, polling, artifact download, and adaptation."""

from __future__ import annotations

import time
from collections.abc import Callable

from writing_factory.config import Settings
from writing_factory.kb.models import ManagedDocument, ParsedDocument
from writing_factory.kb.parsing import MinerUResultAdapter, TextParser, UnsupportedDocumentError
from writing_factory.llm import MinerUClient
from writing_factory.llm.base import ExternalServiceError

ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]


def _no_progress(_percent: int, _message: str) -> None:
    pass


def _no_cancellation() -> None:
    pass


class MinerUDocumentParser:
    """Parse one managed file and retain the provider artifact for reproducibility."""

    def __init__(
        self,
        settings: Settings,
        client: MinerUClient,
        adapter: MinerUResultAdapter | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.client = client
        self.adapter = adapter or MinerUResultAdapter()
        self._sleep = sleep
        self._monotonic = monotonic

    def parse(
        self,
        document: ManagedDocument,
        *,
        progress: ProgressCallback = _no_progress,
        check_cancelled: CancellationCheck = _no_cancellation,
    ) -> ParsedDocument:
        """Run or reuse a completed MinerU parse for a content hash."""

        artifact_dir = self.settings.mineru_artifacts_dir / document.sha256
        archive = artifact_dir / "result.zip"
        if archive.is_file():
            progress(45, "复用 MinerU 解析结果")
            return self.adapter.from_archive(archive, document.filename)

        check_cancelled()
        progress(8, "申请 MinerU 上传地址")
        batch = self.client.create_batch_upload([document.filename], model_version="vlm")
        check_cancelled()
        self.client.upload_file(batch.file_urls[0], document.managed_path)
        progress(18, "文档已上传")

        deadline = self._monotonic() + self.settings.mineru_timeout_seconds
        result_item: dict[str, object] | None = None
        while self._monotonic() < deadline:
            check_cancelled()
            payload = self.client.get_batch_result(batch.batch_id)
            items = payload.get("extract_result") or payload.get("extract_results") or []
            if isinstance(items, list) and items and isinstance(items[0], dict):
                result_item = items[0]
                state = result_item.get("state")
                if state == "done":
                    break
                if state == "failed":
                    raise ExternalServiceError("MinerU document parsing failed")
                extracted = result_item.get("extract_progress")
                progress(self._progress_percent(extracted), f"MinerU: {state or 'pending'}")
            self._sleep(self.settings.mineru_poll_interval_seconds)
        else:
            raise ExternalServiceError("MinerU document parsing timed out")

        download_url = result_item.get("full_zip_url") if result_item else None
        if not isinstance(download_url, str) or not download_url:
            raise ExternalServiceError("MinerU result did not contain a download URL")
        check_cancelled()
        progress(40, "下载 MinerU 解析结果")
        self.client.download_result(download_url, archive)
        parsed = self.adapter.from_archive(archive, document.filename)
        progress(48, "结构化解析完成")
        return parsed

    @staticmethod
    def _progress_percent(value: object) -> int:
        if not isinstance(value, dict):
            return 28
        extracted = value.get("extracted_pages")
        total = value.get("total_pages")
        if not isinstance(extracted, int) or not isinstance(total, int) or total <= 0:
            return 28
        return min(38, 20 + round(18 * extracted / total))


class DocumentParserRouter:
    """Route approved formats to MinerU or the explicit UTF-8 fallback loader."""

    MINERU_FORMATS = frozenset({"pdf", "doc", "docx", "ppt", "pptx"})

    def __init__(self, mineru_parser: MinerUDocumentParser) -> None:
        self.mineru_parser = mineru_parser
        self.text_parser = TextParser()

    def parse(
        self,
        document: ManagedDocument,
        *,
        progress: ProgressCallback = _no_progress,
        check_cancelled: CancellationCheck = _no_cancellation,
    ) -> ParsedDocument:
        """Parse one supported managed document without consulting its original path."""

        if document.format in self.MINERU_FORMATS:
            return self.mineru_parser.parse(
                document,
                progress=progress,
                check_cancelled=check_cancelled,
            )
        if document.format == "txt":
            check_cancelled()
            progress(20, "读取 UTF-8 文本")
            result = self.text_parser.parse(document.managed_path)
            progress(48, "文本解析完成")
            return result.model_copy(update={"filename": document.filename})
        raise UnsupportedDocumentError(f"Unsupported document format: {document.format}")
