"""Typed SiliconFlow and MinerU adapter contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from writing_factory.llm import MinerUClient, SiliconFlowClient
from writing_factory.store import Database


class FakeTransport:
    """Return provider-shaped fixtures while retaining request arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.uploads: list[tuple[str, Path, str]] = []

    def request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, path, kwargs))
        if path == "/chat/completions":
            return {
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "choices": [
                    {
                        "message": {"content": "OK", "reasoning_content": None},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        if path == "/embeddings":
            return {
                "model": "BAAI/bge-m3",
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            }
        if path == "/rerank":
            return {
                "model": "BAAI/bge-reranker-v2-m3",
                "results": [{"index": 1, "relevance_score": 0.9}],
            }
        if path == "/extract/task" and method == "POST":
            return {"code": 0, "data": {"task_id": "task-1"}}
        if path == "/extract/task/task-1":
            return {
                "code": 0,
                "data": {"state": "done", "full_zip_url": "https://download.invalid/a.zip"},
            }
        if path == "/file-urls/batch":
            return {
                "code": 0,
                "data": {
                    "batch_id": "batch-1",
                    "file_urls": ["https://upload.invalid/a"],
                },
            }
        if path == "/extract-results/batch/batch-1":
            return {"code": 0, "data": {"extract_result": []}}
        raise AssertionError(f"Unexpected path: {path}")

    def upload_file(self, upload_url: str, file_path: Path, *, operation: str) -> None:
        self.uploads.append((upload_url, file_path, operation))

    def close(self) -> None:
        pass


def test_siliconflow_typed_operations(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    client = SiliconFlowClient(settings, database)
    client.transport.close()
    transport = FakeTransport()
    client.transport = transport

    chat = client.chat(
        [{"role": "user", "content": "test"}],
        thinking=False,
    )
    embeddings = client.embeddings(["甲", "乙"])
    rerank = client.rerank("查询", ["无关", "相关"])

    assert chat.content == "OK"
    assert chat.usage.total_tokens == 3
    assert "reasoning_effort" not in transport.calls[0][2]["payload"]
    assert embeddings.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert rerank.results[0].index == 1
    client.close()


def test_mineru_typed_operations(settings, tmp_path: Path) -> None:
    database = Database(settings.database_path)
    database.initialize()
    client = MinerUClient(settings, database)
    client.transport.close()
    client.transfers.close()
    transport = FakeTransport()
    client.transport = transport
    client.transfers = transport
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"pdf")

    submitted = client.submit_url("https://source.invalid/paper.pdf")
    task = client.get_task(submitted.task_id)
    batch = client.create_batch_upload([source.name])
    batch_result = client.get_batch_result(batch.batch_id)
    client.upload_file(batch.file_urls[0], source)

    assert task.state == "done"
    assert task.full_zip_url == "https://download.invalid/a.zip"
    assert batch_result == {"extract_result": []}
    assert transport.uploads == [("https://upload.invalid/a", source, "upload_document")]
    client.close()
