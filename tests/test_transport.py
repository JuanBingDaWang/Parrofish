"""Unified transport retry, cache, and failure-boundary tests."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from writing_factory.llm.base import ExternalServiceError, ServiceTransport
from writing_factory.llm.transfers import FileTransferTransport
from writing_factory.store import Database


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "transport.db")
    database.initialize()
    return database


def test_retries_transient_failure_then_caches(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(500, json={"message": "temporary"})
        return httpx.Response(
            200,
            json={"data": [{"index": 0}], "usage": {"total_tokens": 3}},
        )

    client = httpx.Client(
        base_url="https://example.invalid/v1",
        transport=httpx.MockTransport(handler),
    )
    database = _database(tmp_path)
    transport = ServiceTransport(
        provider="test",
        base_url="https://example.invalid/v1",
        credential=SecretStr("private-token"),
        database=database,
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=2,
        http_client=client,
    )

    first = transport.request_json(
        "POST",
        "/operation",
        operation="operation",
        payload={"text": "sensitive-input"},
        prompt_summary={"character_count": 15},
        use_cache=True,
    )
    second = transport.request_json(
        "POST",
        "/operation",
        operation="operation",
        payload={"text": "sensitive-input"},
        prompt_summary={"character_count": 15},
        use_cache=True,
    )

    assert first == second
    assert requests == 2
    with database.connection() as connection:
        calls = connection.execute(
            "SELECT cache_hit, prompt_summary, result_summary FROM api_calls ORDER BY created_at"
        ).fetchall()
    assert [row["cache_hit"] for row in calls] == [0, 1]
    assert all("sensitive-input" not in row["prompt_summary"] for row in calls)
    assert all("sensitive-input" not in row["result_summary"] for row in calls)


def test_provider_error_does_not_echo_response_or_secret(tmp_path: Path) -> None:
    secret = "private-token"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": secret, "input": "private text"})

    transport = ServiceTransport(
        provider="test",
        base_url="https://example.invalid/v1",
        credential=SecretStr(secret),
        database=_database(tmp_path),
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=1,
        http_client=httpx.Client(
            base_url="https://example.invalid/v1",
            transport=httpx.MockTransport(handler),
        ),
    )

    with pytest.raises(ExternalServiceError) as error:
        transport.request_json(
            "POST",
            "/operation",
            operation="operation",
            payload={"text": "private text"},
        )

    assert secret not in str(error.value)
    assert "private text" not in str(error.value)


def test_request_can_limit_attempts_and_override_timeout(tmp_path: Path) -> None:
    requests = 0
    timeout_extensions: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        timeout_extensions.append(request.extensions["timeout"])
        return httpx.Response(500, json={"message": "temporary"})

    transport = ServiceTransport(
        provider="test",
        base_url="https://example.invalid/v1",
        credential=SecretStr("private-token"),
        database=_database(tmp_path),
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=3,
        http_client=httpx.Client(
            base_url="https://example.invalid/v1",
            transport=httpx.MockTransport(handler),
        ),
    )

    with pytest.raises(ExternalServiceError):
        transport.request_json(
            "POST",
            "/operation",
            operation="operation",
            request_timeout_seconds=600,
            request_attempts=1,
        )

    assert requests == 1
    assert timeout_extensions[0]["read"] == 600


def test_total_timeout_window_prevents_another_retry(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(500, json={"message": "temporary"})

    transport = ServiceTransport(
        provider="test",
        base_url="https://example.invalid/v1",
        credential=SecretStr("private-token"),
        database=_database(tmp_path),
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=3,
        http_client=httpx.Client(
            base_url="https://example.invalid/v1",
            transport=httpx.MockTransport(handler),
        ),
    )

    with pytest.raises(ExternalServiceError):
        transport.request_json(
            "POST",
            "/operation",
            operation="operation",
            request_total_timeout_seconds=0.1,
        )

    assert requests == 1


def test_default_total_timeout_applies_without_per_call_override(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(500, json={"message": "temporary"})

    transport = ServiceTransport(
        provider="test",
        base_url="https://example.invalid/v1",
        credential=SecretStr("private-token"),
        database=_database(tmp_path),
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=3,
        default_request_timeout_seconds=0.1,
        http_client=httpx.Client(
            base_url="https://example.invalid/v1",
            transport=httpx.MockTransport(handler),
        ),
    )

    with pytest.raises(ExternalServiceError):
        transport.request_json("POST", "/operation", operation="operation")

    assert requests == 1


def test_protocol_disconnect_is_a_retryable_transport_failure(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            raise httpx.RemoteProtocolError("disconnected", request=request)
        return httpx.Response(200, json={"result": "ok"})

    transport = ServiceTransport(
        provider="test",
        base_url="https://example.invalid/v1",
        credential=SecretStr("private-token"),
        database=_database(tmp_path),
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=2,
        http_client=httpx.Client(
            base_url="https://example.invalid/v1",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = transport.request_json("POST", "/operation", operation="operation")

    assert result == {"result": "ok"}
    assert requests == 2


def test_sse_response_is_collected_inside_unified_transport(tmp_path: Path) -> None:
    body = (
        'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"好"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}\n\n'
        "data: [DONE]\n\n"
    )
    database = _database(tmp_path)
    events: list[dict] = []

    transport = ServiceTransport(
        provider="test",
        base_url="https://example.invalid/v1",
        credential=SecretStr("private-token"),
        database=database,
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=1,
        http_client=httpx.Client(
            base_url="https://example.invalid/v1",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)),
        ),
    )

    result = transport.request_json(
        "POST",
        "/stream",
        operation="stream",
        payload={"stream": True},
        stream_response=True,
        stream_event_callback=events.append,
    )

    assert [chunk["choices"][0]["delta"]["content"] for chunk in result["chunks"]] == [
        "你",
        "好",
    ]
    assert events[-1] == {"_stream_event": "done"}
    with database.connection() as connection:
        call = connection.execute(
            "SELECT input_tokens, output_tokens, total_tokens FROM api_calls"
        ).fetchone()
    assert dict(call) == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}


def test_sse_response_without_done_marker_is_rejected(tmp_path: Path) -> None:
    body = 'data: {"choices":[{"delta":{"content":"未完成"}}]}\n\n'
    transport = ServiceTransport(
        provider="test",
        base_url="https://example.invalid/v1",
        credential=SecretStr("private-token"),
        database=_database(tmp_path),
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=1,
        http_client=httpx.Client(
            base_url="https://example.invalid/v1",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)),
        ),
    )

    with pytest.raises(ExternalServiceError, match=r"before \[DONE\]"):
        transport.request_json(
            "POST",
            "/stream",
            operation="stream",
            payload={"stream": True},
            stream_response=True,
        )


def test_invalid_cached_response_is_quarantined_and_refetched(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"valid": requests > 1, "attempt": requests})

    database = _database(tmp_path)
    transport = ServiceTransport(
        provider="test",
        base_url="https://example.invalid/v1",
        credential=SecretStr("private-token"),
        database=database,
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=1,
        http_client=httpx.Client(
            base_url="https://example.invalid/v1",
            transport=httpx.MockTransport(handler),
        ),
    )
    arguments = {
        "method": "POST",
        "path": "/validated",
        "operation": "validated",
        "payload": {"input": "same"},
        "use_cache": True,
    }
    first = transport.request_json(**arguments)

    def require_valid(response: dict) -> None:
        if not response.get("valid"):
            raise ValueError("invalid fixture")

    second = transport.request_json(**arguments, response_validator=require_valid)
    third = transport.request_json(**arguments, response_validator=require_valid)

    assert first["valid"] is False
    assert second == third == {"valid": True, "attempt": 2}
    assert requests == 2
    with database.connection() as connection:
        cached = connection.execute(
            "SELECT cache_key, operation FROM api_cache ORDER BY operation"
        ).fetchall()
    assert len(cached) == 2
    assert any(row["cache_key"].startswith("invalid:") for row in cached)
    assert {row["operation"] for row in cached} == {"validated", "validated:invalid"}


def test_fresh_invalid_response_is_only_written_to_quarantine(tmp_path: Path) -> None:
    database = _database(tmp_path)
    transport = ServiceTransport(
        provider="test",
        base_url="https://example.invalid/v1",
        credential=SecretStr("private-token"),
        database=database,
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=1,
        http_client=httpx.Client(
            base_url="https://example.invalid/v1",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"valid": False})
            ),
        ),
    )

    def reject(_response: dict) -> None:
        raise ValueError("invalid fixture")

    with pytest.raises(ValueError, match="invalid fixture"):
        transport.request_json(
            "POST",
            "/validated",
            operation="validated",
            payload={"input": "same"},
            use_cache=True,
            response_validator=reject,
        )

    with database.connection() as connection:
        cached = connection.execute("SELECT cache_key, operation FROM api_cache").fetchall()
        call = connection.execute("SELECT status, error_type FROM api_calls").fetchone()
    assert len(cached) == 1
    assert cached[0]["cache_key"].startswith("invalid:")
    assert cached[0]["operation"] == "validated:invalid"
    assert dict(call) == {"status": "error", "error_type": "ValueError"}


def test_presigned_upload_never_sends_provider_authorization(tmp_path: Path) -> None:
    uploaded_headers: list[httpx.Headers] = []

    def upload_handler(request: httpx.Request) -> httpx.Response:
        uploaded_headers.append(request.headers)
        return httpx.Response(200)

    source = tmp_path / "source.pdf"
    source.write_bytes(b"document")
    unauthenticated_client = httpx.Client(transport=httpx.MockTransport(upload_handler))
    transport = FileTransferTransport(
        provider="mineru",
        database=_database(tmp_path),
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=1,
        http_client=unauthenticated_client,
    )

    transport.upload_file(
        "https://object-storage.example/presigned?signature=private",
        source,
        operation="upload_document",
    )

    assert len(uploaded_headers) == 1
    assert "authorization" not in uploaded_headers[0]
    assert "content-type" not in uploaded_headers[0]


def test_presigned_download_is_atomic_and_credential_free(tmp_path: Path) -> None:
    headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers)
        return httpx.Response(200, content=b"archive")

    transport = FileTransferTransport(
        provider="mineru",
        database=_database(tmp_path),
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=1,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    destination = tmp_path / "result.zip"

    returned = transport.download_file(
        "https://object-storage.example/result?signature=private",
        destination,
        operation="download_parse_result",
    )

    assert returned == destination.resolve()
    assert destination.read_bytes() == b"archive"
    assert "authorization" not in headers[0]
    assert not list(tmp_path.glob("*.part"))
