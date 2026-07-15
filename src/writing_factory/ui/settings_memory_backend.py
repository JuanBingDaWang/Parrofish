"""Non-network settings backend for isolated UI construction and tests."""

from __future__ import annotations

from writing_factory.config.settings import (
    DEFAULT_MINERU_BASE_URL,
    DEFAULT_SILICONFLOW_BASE_URL,
)
from writing_factory.llm.configuration import (
    STEP_DEFINITIONS,
    ChatStepConfig,
    ChatStepDefinition,
    ModelCatalogEntry,
    ModelSelections,
)
from writing_factory.llm.settings_service import ModelKind, ProviderName


class InMemorySettingsBackend:
    """Retain edits locally when the real application backend is unavailable."""

    def __init__(self) -> None:
        self.models = ModelSelections(
            chat_model="deepseek-ai/DeepSeek-V4-Flash",
            embedding_model="BAAI/bge-m3",
            rerank_model="BAAI/bge-reranker-v2-m3",
        )
        self.steps: dict[str, ChatStepConfig] = {}
        self.providers = {
            "siliconflow": {
                "configured": False,
                "source": "missing",
                "base_url": DEFAULT_SILICONFLOW_BASE_URL,
            },
            "mineru": {
                "configured": False,
                "source": "missing",
                "base_url": DEFAULT_MINERU_BASE_URL,
            },
        }

    def provider_snapshot(self, provider: ProviderName) -> dict[str, object]:
        return dict(self.providers[provider])

    def save_provider(
        self,
        provider: ProviderName,
        *,
        secret: str | None,
        base_url: str,
    ) -> None:
        self.providers[provider] = {
            "configured": bool(secret) or self.providers[provider]["configured"],
            "source": "credential_store" if secret else self.providers[provider]["source"],
            "base_url": base_url,
        }

    def delete_provider_credential(self, provider: ProviderName) -> None:
        self.providers[provider]["configured"] = False
        self.providers[provider]["source"] = "missing"

    def get_model_selections(self) -> ModelSelections:
        return self.models

    def set_model(self, kind: ModelKind, model_id: str) -> bool:
        field = {
            "chat": "chat_model",
            "embedding": "embedding_model",
            "reranker": "rerank_model",
        }[kind]
        self.models = self.models.model_copy(update={field: model_id})
        return False

    def refresh_models(self, kind: ModelKind) -> list[ModelCatalogEntry]:
        raise RuntimeError(f"{kind} 模型列表服务不可用")

    def cached_models(self, _kind: ModelKind) -> tuple[list[ModelCatalogEntry], str | None]:
        return [], None

    def rebuild_embedding_index(self, **_kwargs) -> str:
        raise RuntimeError("向量重建服务不可用")

    def step_definitions(self) -> tuple[ChatStepDefinition, ...]:
        return STEP_DEFINITIONS

    def get_step_config(self, step_id: str) -> ChatStepConfig:
        definition = next(item for item in STEP_DEFINITIONS if item.step_id == step_id)
        return self.steps.get(step_id, definition.default)

    def set_step_config(self, step_id: str, config: ChatStepConfig) -> None:
        self.steps[step_id] = config

    def reset_step_config(self, step_id: str) -> ChatStepConfig:
        self.steps.pop(step_id, None)
        return next(item.default for item in STEP_DEFINITIONS if item.step_id == step_id)
