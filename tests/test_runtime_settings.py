"""Persistent runtime provider, model, and step configuration tests."""

from __future__ import annotations

from writing_factory.app import build_application
from writing_factory.kb.chunking import StructureChunker
from writing_factory.kb.files import ManagedFileStore
from writing_factory.kb.models import Bibliography, ParsedBlock, ParsedDocument, RetrievalResult
from writing_factory.llm.configuration import ChatStepConfig
from writing_factory.llm.models import EmbeddingResult


class MemoryCredentialStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        self.values[name] = value

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


def test_runtime_settings_persist_and_apply_to_client(settings) -> None:
    secrets = MemoryCredentialStore()
    context = build_application(settings, credential_store=secrets)
    custom = ChatStepConfig(
        temperature=0.8,
        thinking=True,
        reasoning_effort="max",
        max_tokens=12288,
        stream=False,
        retry_count=4,
        timeout_seconds=1500,
    )
    try:
        service = context.settings_service
        service.set_step_config("writing.draft", custom)
        context.hybrid_retriever._cache["stale"] = RetrievalResult(query="旧查询")
        service.set_model("chat", "provider/chat-model")
        service.set_model("reranker", "provider/reranker-model")
        service.save_provider(
            "siliconflow",
            secret="new-secret",
            base_url="https://example.invalid/v1",
        )

        assert service.get_step_config("writing.draft") == custom
        assert context.siliconflow.step_config("writing.draft") == custom
        assert context.siliconflow.chat_model == "provider/chat-model"
        assert context.siliconflow.rerank_model == "provider/reranker-model"
        assert context.siliconflow.transport.base_url == "https://example.invalid/v1"
        assert context.siliconflow.transport.credential_configured
        assert secrets.values["siliconflow-api-key"] == "new-secret"
        assert not context.hybrid_retriever._cache
    finally:
        context.close()

    reopened = build_application(settings, credential_store=secrets)
    try:
        assert reopened.settings_service.get_step_config("writing.draft") == custom
        assert reopened.siliconflow.chat_model == "provider/chat-model"
        assert reopened.siliconflow.rerank_model == "provider/reranker-model"
        assert reopened.siliconflow.transport.base_url == "https://example.invalid/v1"
    finally:
        reopened.close()


def test_embedding_model_switch_waits_for_successful_vector_rebuild(
    settings,
    tmp_path,
    monkeypatch,
) -> None:
    context = build_application(settings, credential_store=MemoryCredentialStore())
    source = tmp_path / "资料.txt"
    source.write_text("数字人文方法能够扩展证据处理能力。", encoding="utf-8")
    managed = ManagedFileStore(settings.managed_documents_dir).import_file(source)
    parsed = ParsedDocument(
        filename=source.name,
        format="txt",
        parser_name="fixture",
        parser_version="1",
        blocks=[ParsedBlock(order=0, block_type="text", text=source.read_text(encoding="utf-8"))],
    )
    chunked = StructureChunker(child_target_chars=20).chunk(managed.doc_id, parsed)
    job_id = context.repository.create_job(context.default_kb_id, source)
    context.repository.save_document_and_chunks(
        kb_id=context.default_kb_id,
        job_id=job_id,
        managed=managed,
        bibliography=Bibliography(title="资料"),
        parsed=parsed,
        chunked=chunked,
    )
    context.repository.mark_ready(context.default_kb_id, managed.doc_id, job_id)

    try:
        pending = context.settings_service.set_model("embedding", "provider/new-embedding")
        assert pending
        assert context.siliconflow.embedding_model == settings.embedding_model

        def embed(texts, *, model=None, **_kwargs):
            return EmbeddingResult(
                vectors=[[float(index), 1.0] for index, _text in enumerate(texts)],
                model=model or "",
            )

        monkeypatch.setattr(context.siliconflow, "embeddings", embed)
        selected = context.settings_service.rebuild_embedding_index(
            progress=lambda _percent, _message: None,
            check_cancelled=lambda: None,
        )

        assert selected == "provider/new-embedding"
        assert context.siliconflow.embedding_model == "provider/new-embedding"
        assert context.settings_service.vectors.embedding_model == "provider/new-embedding"
        assert context.settings_service.get_model_selections().pending_embedding_model is None
    finally:
        context.close()
