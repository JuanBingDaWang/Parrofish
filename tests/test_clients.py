"""Typed SiliconFlow and MinerU adapter contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from writing_factory.llm import MinerUClient, SiliconFlowClient
from writing_factory.llm.common import RetryableServiceError
from writing_factory.llm.configuration import ChatStepConfig
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
        if path.startswith("/models?"):
            return {
                "object": "list",
                "data": [
                    {
                        "id": "provider/test-model",
                        "object": "model",
                        "created": 1,
                        "owned_by": "provider",
                    }
                ],
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
        request_timeout_seconds=600,
        request_total_timeout_seconds=900,
        request_attempts=1,
    )
    embeddings = client.embeddings(["甲", "乙"])
    rerank = client.rerank("查询", ["无关", "相关"])

    assert chat.content == "OK"
    assert chat.usage.total_tokens == 3
    assert "reasoning_effort" not in transport.calls[0][2]["payload"]
    assert transport.calls[0][2]["request_timeout_seconds"] == 600
    assert transport.calls[0][2]["request_total_timeout_seconds"] == 900
    assert transport.calls[0][2]["request_attempts"] == 1
    assert embeddings.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert rerank.results[0].index == 1
    client.close()


def test_siliconflow_applies_runtime_model_and_step_profile(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    client = SiliconFlowClient(settings, database)
    client.transport.close()
    transport = FakeTransport()
    client.transport = transport
    profile = ChatStepConfig(
        temperature=1.1,
        thinking=True,
        reasoning_effort="max",
        max_tokens=4096,
        stream=False,
        retry_count=0,
        timeout_seconds=300,
    )
    client.configure_models(chat_model="provider/test-model")
    client.set_step_config_provider(lambda _step_id: profile)

    client.chat(
        [{"role": "user", "content": "test"}],
        thinking=False,
        step_id="writing.draft",
    )
    models = client.list_models("chat")

    call = transport.calls[0][2]
    assert call["payload"]["model"] == "provider/test-model"
    assert call["payload"]["enable_thinking"] is True
    assert call["payload"]["reasoning_effort"] == "max"
    assert call["payload"]["temperature"] == 1.1
    assert call["payload"]["max_tokens"] == 4096
    assert call["payload"]["stream"] is False
    assert call["request_attempts"] == 1
    assert call["request_total_timeout_seconds"] == 300
    assert models[0].id == "provider/test-model"
    client.close()


def test_siliconflow_freezes_settings_for_one_background_run(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    client = SiliconFlowClient(settings, database)
    client.transport.close()
    transport = FakeTransport()
    client.transport = transport
    initial = ChatStepConfig(
        temperature=0.2,
        thinking=False,
        reasoning_effort="auto",
        max_tokens=2048,
        stream=False,
        retry_count=0,
        timeout_seconds=300,
    )
    changed = initial.model_copy(update={"temperature": 1.2, "max_tokens": 4096})
    active = {"profile": initial}
    client.set_step_config_provider(lambda _step_id: active["profile"])

    with client.freeze_runtime_settings():
        active["profile"] = changed
        client.configure_models(chat_model="provider/changed-model")
        client.chat(
            [{"role": "user", "content": "frozen"}],
            thinking=False,
            step_id="writing.draft",
        )
    client.chat(
        [{"role": "user", "content": "new run"}],
        thinking=False,
        step_id="writing.draft",
    )

    first_payload = transport.calls[0][2]["payload"]
    second_payload = transport.calls[1][2]["payload"]
    assert first_payload["model"] == settings.chat_model
    assert first_payload["temperature"] == 0.2
    assert first_payload["max_tokens"] == 2048
    assert second_payload["model"] == "provider/changed-model"
    assert second_payload["temperature"] == 1.2
    assert second_payload["max_tokens"] == 4096
    client.close()


def test_siliconflow_assembles_streamed_chat(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    client = SiliconFlowClient(settings, database)
    client.transport.close()
    transport = FakeTransport()
    client.transport = transport
    chunks = [
            {
                "model": "stream-model",
                "choices": [{"delta": {"reasoning_content": "思考"}}],
            },
            {
                "choices": [{"delta": {"content": "完成"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
        ]
    observed: list[tuple[str, str]] = []

    def streamed_response(*_args, **kwargs):
        callback = kwargs["stream_event_callback"]
        for chunk in chunks:
            callback(chunk)
        return {"chunks": chunks}

    transport.request_json = streamed_response

    with client.observe_stream(lambda kind, text: observed.append((kind, text))):
        with client.stream_stage("并发 Map 测试"):
            result = client.chat(
                [{"role": "user", "content": "test"}], thinking=True, stream=True
            )

    assert result.content == "完成"
    assert result.reasoning_content == "思考"
    assert result.finish_reason == "stop"
    assert result.usage.total_tokens == 5
    assert observed == [
        ("reasoning::并发 Map 测试", "activity"),
        ("content::并发 Map 测试", "完成"),
    ]
    client.close()


def test_siliconflow_falls_back_to_non_streaming_inside_one_timeout(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    client = SiliconFlowClient(settings, database)
    client.transport.close()

    class FallbackTransport(FakeTransport):
        max_retries = 3
        default_request_timeout_seconds = 900.0

        def request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append((method, path, kwargs))
            if kwargs["stream_response"]:
                raise RetryableServiceError("planned stream network failure")
            response = {
                "model": "fallback-model",
                "choices": [
                    {
                        "message": {"content": '{"ok": true}', "reasoning_content": None},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 3},
            }
            validator = kwargs.get("response_validator")
            if validator is not None:
                validator(response)
            return response

    transport = FallbackTransport()
    client.transport = transport
    observed: list[tuple[str, str]] = []

    with client.observe_stream(lambda kind, text: observed.append((kind, text))):
        result = client.chat(
            [{"role": "user", "content": "test"}],
            thinking=False,
            response_format="json_object",
            stream=True,
        )

    assert result.content == '{"ok": true}'
    assert len(transport.calls) == 2
    assert transport.calls[0][2]["request_attempts"] == 2
    assert transport.calls[0][2]["stream_response"] is True
    assert transport.calls[1][2]["request_attempts"] == 1
    assert transport.calls[1][2]["stream_response"] is False
    assert transport.calls[1][2]["payload"]["stream"] is False
    assert transport.calls[1][2]["request_total_timeout_seconds"] <= 900.0
    assert ("status", "流式重试仍未完成，最后一次改用非流式请求") in observed
    assert observed[-1] == ("content", '{"ok": true}')
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
