"""Typed MinerU task and batch-upload API client."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from writing_factory.config import Settings
from writing_factory.llm.base import ExternalServiceError, ServiceTransport
from writing_factory.llm.models import MinerUBatchUpload, MinerUTask
from writing_factory.llm.transfers import FileTransferTransport
from writing_factory.store import Database


class MinerUClient:
    """The only MinerU API entry point exposed to ingestion code."""

    def __init__(self, settings: Settings, database: Database) -> None:
        self.transport = ServiceTransport(
            provider="mineru",
            base_url=settings.mineru_base_url,
            credential=settings.mineru_api_token,
            database=database,
            connect_timeout_seconds=settings.connect_timeout_seconds,
            read_timeout_seconds=settings.read_timeout_seconds,
            max_retries=settings.max_retries,
            minimum_interval_seconds=settings.min_request_interval_seconds,
        )
        self.transfers = FileTransferTransport(
            provider="mineru",
            database=database,
            connect_timeout_seconds=settings.connect_timeout_seconds,
            read_timeout_seconds=settings.read_timeout_seconds,
            max_retries=settings.max_retries,
            minimum_interval_seconds=settings.min_request_interval_seconds,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self.transport.close()
        self.transfers.close()

    def configure(
        self,
        *,
        credential: SecretStr | None = None,
        base_url: str | None = None,
    ) -> None:
        """Apply saved provider settings to subsequent MinerU API calls."""

        if credential is not None:
            self.transport.set_credential(credential)
        if base_url is not None:
            self.transport.set_base_url(base_url)

    def submit_url(
        self,
        url: str,
        *,
        model_version: str = "vlm",
        enable_formula: bool = True,
        enable_table: bool = True,
    ) -> MinerUTask:
        """Submit a remotely accessible document for asynchronous parsing."""

        response = self.transport.request_json(
            "POST",
            "/extract/task",
            operation="submit_url_parse",
            payload={
                "url": url,
                "model_version": model_version,
                "enable_formula": enable_formula,
                "enable_table": enable_table,
            },
            prompt_summary={"source": "remote_url", "model_version": model_version},
            use_cache=False,
        )
        data = self._data(response)
        task_id = data.get("task_id")
        if not isinstance(task_id, str):
            raise ExternalServiceError("MinerU response did not contain a task id")
        return MinerUTask(task_id=task_id, raw_data=data)

    def get_task(self, task_id: str) -> MinerUTask:
        """Read current state and download URL for a parsing task."""

        response = self.transport.request_json(
            "GET",
            f"/extract/task/{task_id}",
            operation="get_parse_task",
            prompt_summary={"task_id_hash": task_id[:8]},
            use_cache=False,
        )
        data = self._data(response)
        return MinerUTask(
            task_id=task_id,
            state=data.get("state"),
            full_zip_url=data.get("full_zip_url"),
            err_msg=data.get("err_msg"),
            raw_data=data,
        )

    def create_batch_upload(
        self,
        filenames: Sequence[str],
        *,
        model_version: str = "vlm",
        enable_formula: bool = True,
        enable_table: bool = True,
    ) -> MinerUBatchUpload:
        """Request presigned URLs for local document uploads."""

        response = self.transport.request_json(
            "POST",
            "/file-urls/batch",
            operation="create_batch_upload",
            payload={
                "files": [{"name": name} for name in filenames],
                "model_version": model_version,
                "enable_formula": enable_formula,
                "enable_table": enable_table,
            },
            prompt_summary={"file_count": len(filenames), "model_version": model_version},
            use_cache=False,
        )
        data = self._data(response)
        batch_id = data.get("batch_id")
        file_urls = data.get("file_urls")
        if not isinstance(batch_id, str) or not isinstance(file_urls, list):
            raise ExternalServiceError("MinerU returned invalid batch upload details")
        return MinerUBatchUpload(batch_id=batch_id, file_urls=file_urls)

    def upload_file(self, upload_url: str, file_path: Path) -> None:
        """Upload a local document to a MinerU presigned URL without auth leakage."""

        self.transfers.upload_file(
            upload_url,
            file_path,
            operation="upload_document",
        )

    def download_result(self, download_url: str, destination: Path) -> Path:
        """Download a completed result archive without forwarding MinerU auth."""

        return self.transfers.download_file(
            download_url,
            destination,
            operation="download_parse_result",
        )

    def get_batch_result(self, batch_id: str) -> dict[str, Any]:
        """Read parsing states for all files in a batch."""

        response = self.transport.request_json(
            "GET",
            f"/extract-results/batch/{batch_id}",
            operation="get_batch_result",
            prompt_summary={"batch_id_hash": batch_id[:8]},
            use_cache=False,
        )
        return self._data(response)

    @staticmethod
    def _data(response: dict[str, Any]) -> dict[str, Any]:
        code = response.get("code", 0)
        if code not in (0, 200):
            raise ExternalServiceError(f"MinerU rejected the operation (code {code})")
        data = response.get("data")
        if not isinstance(data, dict):
            raise ExternalServiceError("MinerU returned an invalid response shape")
        return data
