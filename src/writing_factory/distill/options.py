"""Persisted quality presets and optional stages for persona distillation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

DistillationPreset = Literal["fast", "balanced", "deep", "custom", "legacy"]
PersonaMode = Literal["person", "topic"]


class DistillationOptions(BaseModel):
    """One run's immutable selection of expensive distillation stages."""

    model_config = ConfigDict(frozen=True)

    preset: DistillationPreset = "legacy"
    cross_document_validation: bool = True
    generative_validation: bool = True
    exclusivity_validation: bool = True
    composition_dna: bool = True

    @model_validator(mode="after")
    def enforce_dependencies(self) -> DistillationOptions:
        if (self.generative_validation or self.exclusivity_validation) and not (
            self.cross_document_validation
        ):
            raise ValueError("生成力和排他性验证依赖跨文档复现与聚类")
        return self

    @classmethod
    def from_preset(
        cls,
        preset: Literal["fast", "balanced", "deep", "custom"],
        *,
        has_control_corpus: bool = False,
    ) -> DistillationOptions:
        """Build the documented defaults without inventing unavailable validation."""

        if preset == "fast":
            return cls(
                preset=preset,
                cross_document_validation=False,
                generative_validation=False,
                exclusivity_validation=False,
                composition_dna=False,
            )
        if preset == "balanced":
            return cls(
                preset=preset,
                cross_document_validation=True,
                generative_validation=False,
                exclusivity_validation=False,
                composition_dna=True,
            )
        if preset == "deep":
            return cls(
                preset=preset,
                cross_document_validation=True,
                generative_validation=True,
                exclusivity_validation=has_control_corpus,
                composition_dna=True,
            )
        return cls(preset="custom")

    def normalized(
        self,
        *,
        mode: PersonaMode,
        has_control_corpus: bool,
    ) -> DistillationOptions:
        """Remove stages that cannot honestly run for the selected inputs."""

        if self.preset == "legacy":
            return self
        if mode == "topic":
            return self.model_copy(
                update={
                    "cross_document_validation": False,
                    "generative_validation": False,
                    "exclusivity_validation": False,
                }
            )
        if not has_control_corpus and self.exclusivity_validation:
            return self.model_copy(update={"exclusivity_validation": False})
        return self

    @property
    def label(self) -> str:
        return {
            "fast": "快速",
            "balanced": "均衡",
            "deep": "深度",
            "custom": "自定义",
            "legacy": "历史完整模式",
        }[self.preset]

    @property
    def enabled_step_labels(self) -> tuple[str, ...]:
        labels = ["基础 Map 与 Reduce", "表达 DNA", "本地质量门"]
        if self.cross_document_validation:
            labels.append("跨文档复现与聚类")
        if self.generative_validation:
            labels.append("留出语料生成力验证")
        if self.exclusivity_validation:
            labels.append("对照语料排他性验证")
        if self.composition_dna:
            labels.append("完整谋篇 DNA")
        return tuple(labels)

    @property
    def cache_key(self) -> str:
        return self.model_dump_json(exclude={"preset"})


LEGACY_DISTILLATION_OPTIONS = DistillationOptions()
