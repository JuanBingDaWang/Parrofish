"""Typed SiliconFlow and MinerU adapter contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from writing_factory.llm import BochaClient, MinerUClient, SiliconFlowClient
from writing_factory.llm.common import IncompleteStreamError, RetryableServiceError
from writing_factory.llm.configuration import STEP_DEFINITIONS, ChatStepConfig
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
        if path == "/web-search":
            return {
                "code": 200,
                "data": {
                    "webPages": {
                        "totalEstimatedMatches": 12,
                        "value": [
                            {
                                "id": "web-1",
                                "name": "测试网页",
                                "url": "https://example.invalid/article",
                                "snippet": "网页摘要",
                                "summary": "更完整的网页摘要",
                                "siteName": "示例站点",
                                "datePublished": "2026-07-16",
                                "language": "zh",
                            }
                        ],
                    }
                },
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


def test_heavy_distillation_steps_have_longer_independent_time_budgets() -> None:
    definitions = {item.step_id: item for item in STEP_DEFINITIONS}
    heavy_steps = {
        "distill.structure_map",
        "distill.structure_reduce",
        "distill.paper_profile",
        "distill.reduce",
    }

    for step_id in heavy_steps:
        profile = definitions[step_id].default
        assert profile.timeout_seconds == 1800
        assert profile.total_timeout_seconds == 3600

    assert definitions["writing.draft"].default.timeout_seconds is None
    assert definitions["writing.draft"].default.total_timeout_seconds is None


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
    assert transport.calls[0][2]["request_total_timeout_seconds"] == 600
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
        total_timeout_seconds=1200,
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
    assert call["request_timeout_seconds"] == 300
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
        total_timeout_seconds=1200,
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
    forwarded_checks = []

    def cancellation_check() -> None:
        pass

    def streamed_response(*_args, **kwargs):
        forwarded_checks.append(kwargs["check_cancelled"])
        callback = kwargs["stream_event_callback"]
        for chunk in chunks:
            callback(chunk)
        return {"chunks": chunks}

    transport.request_json = streamed_response

    with client.observe_stream(
        lambda kind, text: observed.append((kind, text)),
        check_cancelled=cancellation_check,
    ):
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
        ("complete::并发 Map 测试", "done"),
    ]
    assert forwarded_checks == [cancellation_check]
    client.close()


def test_siliconflow_reports_one_named_business_failure_after_complete_response(
    settings,
) -> None:
    database = Database(settings.database_path)
    database.initialize()
    client = SiliconFlowClient(settings, database)
    client.transport.close()
    transport = FakeTransport()
    client.transport = transport
    observed: list[tuple[str, str]] = []

    with pytest.raises(ValueError, match="文档级 Schema 校验失败"):
        with client.observe_stream(lambda kind, text: observed.append((kind, text))):
            with client.stream_stage("认知 Map · 失败论文.pdf · abc123"):
                client.chat(
                    [{"role": "user", "content": "test"}],
                    thinking=False,
                    stream=False,
                )
                raise ValueError("文档级 Schema 校验失败")

    failures = [item for item in observed if item[0].startswith("error::")]
    assert failures == [
        ("error::认知 Map · 失败论文.pdf · abc123", "文档级 Schema 校验失败")
    ]
    client.close()


def test_siliconflow_fallback_gets_an_independent_attempt_timeout(settings) -> None:
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
            request_timeout_seconds=600,
            request_total_timeout_seconds=900,
            request_attempts=2,
        )

    assert result.content == '{"ok": true}'
    assert len(transport.calls) == 2
    assert transport.calls[0][2]["request_attempts"] == 1
    assert transport.calls[0][2]["stream_response"] is True
    assert transport.calls[1][2]["request_attempts"] == 1
    assert transport.calls[1][2]["stream_response"] is False
    assert transport.calls[1][2]["payload"]["stream"] is False
    assert transport.calls[0][2]["request_timeout_seconds"] == 600
    assert transport.calls[0][2]["request_total_timeout_seconds"] == 600
    assert transport.calls[1][2]["request_timeout_seconds"] == 600
    assert transport.calls[1][2]["request_total_timeout_seconds"] == 600
    assert (
        "attempt_reset",
        "第 1/2 次流式尝试未完整结束：planned stream network failure",
    ) in observed
    assert (
        "status",
        "正在启动第 2/2 次非流式兜底，单次上限 600 秒",
    ) in observed
    assert ("content", '{"ok": true}') in observed
    assert observed[-1] == ("complete", "done")
    client.close()


def test_siliconflow_inflight_fallback_can_finish_after_soft_budget(
    settings,
    monkeypatch,
) -> None:
    database = Database(settings.database_path)
    database.initialize()
    client = SiliconFlowClient(settings, database)
    client.transport.close()
    clock = [0.0]

    class SlowFallbackTransport(FakeTransport):
        default_request_timeout_seconds = 600.0
        max_retries = 2

        def request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append((method, path, kwargs))
            if kwargs["stream_response"]:
                clock[0] = 850.0
                raise RetryableServiceError("planned late stream failure")
            clock[0] = 1000.0
            return {
                "model": "fallback-model",
                "choices": [
                    {
                        "message": {"content": "完整兜底结果", "reasoning_content": None},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 3},
            }

    transport = SlowFallbackTransport()
    client.transport = transport
    monkeypatch.setattr(client, "_monotonic", lambda: clock[0])

    result = client.chat(
        [{"role": "user", "content": "test"}],
        thinking=False,
        stream=True,
        request_timeout_seconds=600,
        request_total_timeout_seconds=900,
        request_attempts=2,
    )

    assert result.content == "完整兜底结果"
    assert clock[0] > 900
    assert len(transport.calls) == 2
    assert transport.calls[1][2]["request_total_timeout_seconds"] == 600
    client.close()


def test_siliconflow_inflight_stream_retry_can_finish_after_soft_budget(
    settings,
    monkeypatch,
) -> None:
    database = Database(settings.database_path)
    database.initialize()
    client = SiliconFlowClient(settings, database)
    client.transport.close()
    clock = [0.0]

    class SlowStreamRetryTransport(FakeTransport):
        default_request_timeout_seconds = 600.0
        max_retries = 3

        def request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append((method, path, kwargs))
            if len(self.calls) == 1:
                clock[0] = 850.0
                raise RetryableServiceError("planned late stream failure")
            clock[0] = 1000.0
            chunks = [
                {
                    "model": "stream-retry-model",
                    "choices": [
                        {
                            "delta": {"content": "第二次流式完整结果"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 3},
                }
            ]
            return {"chunks": chunks}

    transport = SlowStreamRetryTransport()
    client.transport = transport
    monkeypatch.setattr(client, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(client, "_wait_before_retry", lambda *_args: None)

    result = client.chat(
        [{"role": "user", "content": "test"}],
        thinking=False,
        stream=True,
        request_timeout_seconds=600,
        request_total_timeout_seconds=900,
        request_attempts=3,
    )

    assert result.content == "第二次流式完整结果"
    assert clock[0] > 900
    assert len(transport.calls) == 2
    assert all(call[2]["stream_response"] for call in transport.calls)
    assert transport.calls[1][2]["request_total_timeout_seconds"] == 600
    client.close()


def test_siliconflow_soft_budget_prevents_starting_another_attempt(
    settings,
    monkeypatch,
) -> None:
    database = Database(settings.database_path)
    database.initialize()
    client = SiliconFlowClient(settings, database)
    client.transport.close()
    clock = [0.0]

    class ExhaustedBudgetTransport(FakeTransport):
        default_request_timeout_seconds = 600.0
        max_retries = 2

        def request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append((method, path, kwargs))
            clock[0] = 901.0
            raise RetryableServiceError("planned late stream failure")

    transport = ExhaustedBudgetTransport()
    client.transport = transport
    monkeypatch.setattr(client, "_monotonic", lambda: clock[0])
    observed: list[tuple[str, str]] = []

    with pytest.raises(IncompleteStreamError, match="未启动第 2/2 次非流式兜底"):
        with client.observe_stream(lambda kind, text: observed.append((kind, text))):
            client.chat(
                [{"role": "user", "content": "test"}],
                thinking=False,
                stream=True,
                request_timeout_seconds=600,
                request_total_timeout_seconds=900,
                request_attempts=2,
            )

    assert len(transport.calls) == 1
    assert (
        "status",
        "整项调用总预算 900 秒已耗尽，未启动第 2/2 次非流式兜底",
    ) in observed
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


def test_bocha_typed_search_uses_traceable_web_fields(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    client = BochaClient(settings, database)
    client.transport.close()
    transport = FakeTransport()
    client.transport = transport

    result = client.search("测试查询", count=3)

    method, path, kwargs = transport.calls[0]
    assert (method, path) == ("POST", "/web-search")
    assert kwargs["payload"]["count"] == 3
    assert kwargs["payload"]["summary"] is True
    assert kwargs["use_cache"] is True
    assert result.total_estimated_matches == 12
    assert result.pages[0].title == "测试网页"
    assert result.pages[0].url == "https://example.invalid/article"
    client.close()
