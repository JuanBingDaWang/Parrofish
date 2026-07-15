"""Runtime persistence and application of provider, model, and step settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import SecretStr

from writing_factory.config import CredentialStore, Settings, load_settings
from writing_factory.config.logging import register_secret
from writing_factory.config.secrets import MINERU_CREDENTIAL, SILICONFLOW_CREDENTIAL
from writing_factory.llm.configuration import (
    STEP_DEFINITIONS,
    ChatStepConfig,
    ChatStepDefinition,
    ModelCatalogEntry,
    ModelSelections,
    get_step_definition,
)
from writing_factory.llm.mineru import MinerUClient
from writing_factory.llm.siliconflow import SiliconFlowClient
from writing_factory.store.database import utc_now
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.settings_repository import RuntimeSettingsRepository
from writing_factory.store.vector_index import LanceVectorIndex

ProviderName = Literal["siliconflow", "mineru"]
ModelKind = Literal["chat", "embedding", "reranker"]
ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]

MODEL_SELECTIONS_KEY = "siliconflow_model_selections_v1"
STEP_CONFIGS_KEY = "siliconflow_step_configs_v1"


class ApplicationSettingsService:
    """Back the settings UI while keeping all provider mutation centralized."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository: RuntimeSettingsRepository,
        credential_store: CredentialStore,
        siliconflow: SiliconFlowClient,
        mineru: MinerUClient,
        kb_repository: KnowledgeBaseRepository,
        vectors: LanceVectorIndex,
        kb_id: str,
        on_retrieval_configuration_changed: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.credential_store = credential_store
        self.siliconflow = siliconflow
        self.mineru = mineru
        self.kb_repository = kb_repository
        self.vectors = vectors
        self.kb_id = kb_id
        self.on_retrieval_configuration_changed = (
            on_retrieval_configuration_changed or (lambda: None)
        )
        self._credential_sources = {
            "siliconflow": settings.siliconflow_credential_source,
            "mineru": settings.mineru_credential_source,
        }
        self._apply_saved_runtime_settings()

    def _apply_saved_runtime_settings(self) -> None:
        models = self.get_model_selections()
        if self.vectors.embedding_model and self.vectors.embedding_model != models.embedding_model:
            models = models.model_copy(
                update={
                    "embedding_model": self.vectors.embedding_model,
                    "pending_embedding_model": None,
                }
            )
            self.repository.set(MODEL_SELECTIONS_KEY, models.model_dump(mode="json"))
        self.siliconflow.configure_models(
            chat_model=models.chat_model,
            embedding_model=models.embedding_model,
            rerank_model=models.rerank_model,
        )
        self.siliconflow.set_step_config_provider(self.get_step_config)
        siliconflow_url = self.repository.get(
            "siliconflow_base_url", self.settings.siliconflow_base_url
        )
        mineru_url = self.repository.get("mineru_base_url", self.settings.mineru_base_url)
        self.siliconflow.configure_provider(base_url=str(siliconflow_url))
        self.mineru.configure(base_url=str(mineru_url))

    def provider_snapshot(self, provider: ProviderName) -> dict[str, object]:
        if provider == "siliconflow":
            configured = self.siliconflow.transport.credential_configured
            base_url = self.siliconflow.transport.base_url
        else:
            configured = self.mineru.transport.credential_configured
            base_url = self.mineru.transport.base_url
        return {
            "provider": provider,
            "configured": configured,
            "source": self._credential_sources[provider],
            "base_url": base_url,
        }

    def save_provider(
        self,
        provider: ProviderName,
        *,
        secret: str | None,
        base_url: str,
    ) -> None:
        credential_name = (
            SILICONFLOW_CREDENTIAL if provider == "siliconflow" else MINERU_CREDENTIAL
        )
        credential = None
        if secret and secret.strip():
            self.credential_store.set(credential_name, secret)
            credential = SecretStr(secret.strip())
            register_secret(secret.strip())
            self._credential_sources[provider] = "credential_store"
        self.repository.set(f"{provider}_base_url", base_url.rstrip("/"))
        if provider == "siliconflow":
            self.siliconflow.configure_provider(credential=credential, base_url=base_url)
        else:
            self.mineru.configure(credential=credential, base_url=base_url)

    def delete_provider_credential(self, provider: ProviderName) -> None:
        credential_name = (
            SILICONFLOW_CREDENTIAL if provider == "siliconflow" else MINERU_CREDENTIAL
        )
        self.credential_store.delete(credential_name)
        fallback = load_settings(
            project_root=self.settings.project_root,
            credential_store=self.credential_store,
        )
        if provider == "siliconflow":
            credential = fallback.siliconflow_api_key
            source = fallback.siliconflow_credential_source
            self.siliconflow.configure_provider(credential=credential)
        else:
            credential = fallback.mineru_api_token
            source = fallback.mineru_credential_source
            self.mineru.configure(credential=credential)
        self._credential_sources[provider] = source

    def get_model_selections(self) -> ModelSelections:
        defaults = ModelSelections(
            chat_model=self.settings.chat_model,
            embedding_model=self.settings.embedding_model,
            rerank_model=self.settings.rerank_model,
        )
        raw = self.repository.get(MODEL_SELECTIONS_KEY, defaults.model_dump(mode="json"))
        try:
            return ModelSelections.model_validate(raw)
        except Exception:
            return defaults

    def set_model(self, kind: ModelKind, model_id: str) -> bool:
        selected = model_id.strip()
        if not selected:
            raise ValueError("模型 ID 不能为空")
        current = self.get_model_selections()
        if kind == "chat":
            updated = current.model_copy(update={"chat_model": selected})
            self.siliconflow.configure_models(chat_model=selected)
        elif kind == "reranker":
            updated = current.model_copy(update={"rerank_model": selected})
            self.siliconflow.configure_models(rerank_model=selected)
        else:
            has_chunks = bool(self.kb_repository.ready_child_chunks(self.kb_id))
            if has_chunks and selected != current.embedding_model:
                updated = current.model_copy(update={"pending_embedding_model": selected})
                self.repository.set(MODEL_SELECTIONS_KEY, updated.model_dump(mode="json"))
                return True
            updated = current.model_copy(
                update={"embedding_model": selected, "pending_embedding_model": None}
            )
            self.siliconflow.configure_models(embedding_model=selected)
        self.repository.set(MODEL_SELECTIONS_KEY, updated.model_dump(mode="json"))
        self.on_retrieval_configuration_changed()
        return False

    def refresh_models(self, kind: ModelKind) -> list[ModelCatalogEntry]:
        entries = self.siliconflow.list_models(kind)
        self.repository.set(
            f"siliconflow_model_catalog_{kind}_v1",
            {
                "updated_at": utc_now(),
                "items": [entry.model_dump(mode="json") for entry in entries],
            },
        )
        return entries

    def cached_models(self, kind: ModelKind) -> tuple[list[ModelCatalogEntry], str | None]:
        raw = self.repository.get(f"siliconflow_model_catalog_{kind}_v1", {})
        if not isinstance(raw, dict):
            return [], None
        items = raw.get("items", [])
        try:
            entries = [ModelCatalogEntry.model_validate(item) for item in items]
        except Exception:
            return [], None
        updated_at = raw.get("updated_at")
        return entries, str(updated_at) if updated_at else None

    def step_definitions(self) -> tuple[ChatStepDefinition, ...]:
        return STEP_DEFINITIONS

    def get_step_config(self, step_id: str) -> ChatStepConfig:
        definition = get_step_definition(step_id)
        raw_configs = self.repository.get(STEP_CONFIGS_KEY, {})
        if not isinstance(raw_configs, dict) or step_id not in raw_configs:
            return definition.default
        try:
            return ChatStepConfig.model_validate(raw_configs[step_id])
        except Exception:
            return definition.default

    def set_step_config(self, step_id: str, config: ChatStepConfig) -> None:
        definition = get_step_definition(step_id)
        raw = self.repository.get(STEP_CONFIGS_KEY, {})
        configs = dict(raw) if isinstance(raw, dict) else {}
        configs[step_id] = config.model_dump(mode="json")
        self.repository.set(STEP_CONFIGS_KEY, configs)
        if definition.group == "retrieval":
            self.on_retrieval_configuration_changed()

    def reset_step_config(self, step_id: str) -> ChatStepConfig:
        definition = get_step_definition(step_id)
        raw = self.repository.get(STEP_CONFIGS_KEY, {})
        configs = dict(raw) if isinstance(raw, dict) else {}
        configs.pop(step_id, None)
        self.repository.set(STEP_CONFIGS_KEY, configs)
        if definition.group == "retrieval":
            self.on_retrieval_configuration_changed()
        return definition.default

    def rebuild_embedding_index(
        self,
        *,
        progress: ProgressCallback,
        check_cancelled: CancellationCheck,
    ) -> str:
        models = self.get_model_selections()
        pending = models.pending_embedding_model
        if not pending:
            raise ValueError("没有等待重建的 Embedding 模型")
        chunks = self.kb_repository.ready_child_chunks(self.kb_id)
        vectors: list[list[float]] = []
        total = max(1, len(chunks))
        for start in range(0, len(chunks), self.settings.embedding_batch_size):
            check_cancelled()
            batch = chunks[start : start + self.settings.embedding_batch_size]
            result = self.siliconflow.embeddings(
                [chunk.text for chunk in batch],
                model=pending,
                use_cache=True,
            )
            vectors.extend(result.vectors)
            progress(round(90 * (start + len(batch)) / total), "使用新模型重建向量")
        check_cancelled()
        self.vectors.rebuild(chunks, vectors, model_id=pending)
        updated = models.model_copy(
            update={"embedding_model": pending, "pending_embedding_model": None}
        )
        self.repository.set(MODEL_SELECTIONS_KEY, updated.model_dump(mode="json"))
        self.siliconflow.configure_models(embedding_model=pending)
        self.on_retrieval_configuration_changed()
        progress(100, "向量索引已切换")
        return pending
