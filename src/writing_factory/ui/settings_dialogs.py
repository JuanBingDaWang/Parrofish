"""Independent dialogs for provider, model, and per-step settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from writing_factory.llm.configuration import (
    ChatStepConfig,
    ChatStepDefinition,
    ModelCatalogEntry,
    ModelSelections,
)
from writing_factory.llm.settings_service import ModelKind, ProviderName
from writing_factory.ui.widgets import NoWheelComboBox


class SettingsDialogBackend(Protocol):
    def provider_snapshot(self, provider: ProviderName) -> dict[str, object]: ...

    def save_provider(
        self,
        provider: ProviderName,
        *,
        secret: str | None,
        base_url: str,
    ) -> None: ...

    def delete_provider_credential(self, provider: ProviderName) -> None: ...

    def get_model_selections(self) -> ModelSelections: ...

    def set_model(self, kind: ModelKind, model_id: str) -> bool: ...

    def refresh_models(self, kind: ModelKind) -> list[ModelCatalogEntry]: ...

    def cached_models(self, kind: ModelKind) -> tuple[list[ModelCatalogEntry], str | None]: ...

    def rebuild_embedding_index(
        self,
        *,
        progress: Callable[[int, str], None],
        check_cancelled: Callable[[], None],
    ) -> str: ...

    def get_step_config(self, step_id: str) -> ChatStepConfig: ...

    def set_step_config(self, step_id: str, config: ChatStepConfig) -> None: ...

    def reset_step_config(self, step_id: str) -> ChatStepConfig: ...


class ProviderSettingsDialog(QDialog):
    """Edit one provider credential and endpoint without exposing saved secrets."""

    def __init__(
        self,
        backend: SettingsDialogBackend,
        provider: ProviderName,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.backend = backend
        self.provider = provider
        snapshot = backend.provider_snapshot(provider)
        display_name = "SiliconFlow" if provider == "siliconflow" else "MinerU"
        self.setWindowTitle(f"{display_name} API 设置")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        status = "已保存凭据" if snapshot.get("configured") else "尚未配置凭据"
        source = _credential_source_label(str(snapshot.get("source", "missing")))
        self.status_label = QLabel(f"{status} · 来源：{source}")
        layout.addWidget(self.status_label)

        form = QFormLayout()
        self.secret_input = QLineEdit()
        self.secret_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.secret_input.setPlaceholderText("留空表示保留当前凭据")
        form.addRow("API Key / Token:", self.secret_input)
        self.show_secret = QCheckBox("显示本次输入")
        self.show_secret.toggled.connect(self._toggle_secret)
        form.addRow("", self.show_secret)
        self.base_url_input = QLineEdit(str(snapshot.get("base_url", "")))
        form.addRow("Base URL:", self.base_url_input)
        layout.addLayout(form)

        note = QLabel("凭据保存在 Windows Credential Manager，不写入项目或 SQLite。")
        note.setObjectName("mutedText")
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.delete_button = buttons.addButton(
            "删除系统凭据", QDialogButtonBox.ButtonRole.DestructiveRole
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        self.delete_button.clicked.connect(self._delete)
        layout.addWidget(buttons)

    def _toggle_secret(self, visible: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        self.secret_input.setEchoMode(mode)

    def _save(self) -> None:
        try:
            self.backend.save_provider(
                self.provider,
                secret=self.secret_input.text().strip() or None,
                base_url=self.base_url_input.text().strip(),
            )
        except Exception as exc:
            self.status_label.setText(str(exc))
            self.status_label.setObjectName("errorText")
            return
        self.accept()

    def _delete(self) -> None:
        try:
            self.backend.delete_provider_credential(self.provider)
        except Exception as exc:
            self.status_label.setText(str(exc))
            self.status_label.setObjectName("errorText")
            return
        self.accept()


class StepSettingsDialog(QDialog):
    """Edit one logical chat stage and preserve its recommended reset profile."""

    def __init__(
        self,
        backend: SettingsDialogBackend,
        definition: ChatStepDefinition,
        *,
        on_changed: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.backend = backend
        self.definition = definition
        self.on_changed = on_changed
        self.setWindowTitle(f"{definition.name} · 模型调用设置")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        description = QLabel(definition.description)
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(0.0, 2.0)
        self.temperature_input.setSingleStep(0.1)
        self.temperature_input.setDecimals(1)
        form.addRow("Temperature:", self.temperature_input)
        self.thinking_combo = NoWheelComboBox()
        self.thinking_combo.addItem("按流程推荐", None)
        self.thinking_combo.addItem("开启", True)
        self.thinking_combo.addItem("关闭", False)
        self.thinking_combo.currentIndexChanged.connect(self._sync_effort_state)
        form.addRow("Thinking:", self.thinking_combo)
        self.effort_combo = NoWheelComboBox()
        self.effort_combo.addItem("自动 / 沿用推荐", "auto")
        self.effort_combo.addItem("high", "high")
        self.effort_combo.addItem("max", "max")
        form.addRow("Reasoning effort:", self.effort_combo)
        self.max_tokens_input = QSpinBox()
        self.max_tokens_input.setRange(256, 65536)
        self.max_tokens_input.setSingleStep(512)
        form.addRow("Max tokens:", self.max_tokens_input)
        self.stream_checkbox = QCheckBox("使用流式请求并显示实时输出")
        form.addRow("传输:", self.stream_checkbox)
        self.retry_input = QSpinBox()
        self.retry_input.setRange(0, 5)
        self.retry_input.setSuffix(" 次")
        self.retry_input.setToolTip("0 表示只请求一次；这里填写的是失败后的重试次数")
        form.addRow("网络重试:", self.retry_input)
        self.timeout_input = QSpinBox()
        self.timeout_input.setRange(0, 3600)
        self.timeout_input.setSingleStep(60)
        self.timeout_input.setSpecialValueText("继承全局")
        self.timeout_input.setSuffix(" 秒")
        self.timeout_input.setToolTip("一次逻辑调用的总时限，包含流式重试和非流式兜底")
        form.addRow("单次请求超时:", self.timeout_input)
        layout.addLayout(form)
        if definition.framework_token_escalation:
            framework_note = QLabel("内容规划截断后按基础 Max tokens 的 1×、2×、4× 重新生成。")
            framework_note.setObjectName("mutedText")
            layout.addWidget(framework_note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        reset = buttons.addButton("恢复推荐值", QDialogButtonBox.ButtonRole.ResetRole)
        reset.clicked.connect(self._reset)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._set_values(backend.get_step_config(definition.step_id))

    def _set_values(self, config: ChatStepConfig) -> None:
        self.temperature_input.setValue(config.temperature)
        thinking_index = self.thinking_combo.findData(config.thinking)
        self.thinking_combo.setCurrentIndex(max(0, thinking_index))
        effort_index = self.effort_combo.findData(config.reasoning_effort)
        self.effort_combo.setCurrentIndex(max(0, effort_index))
        self.max_tokens_input.setValue(config.max_tokens)
        self.stream_checkbox.setChecked(config.stream)
        self.retry_input.setValue(config.retry_count)
        self.timeout_input.setValue(config.timeout_seconds or 0)
        self._sync_effort_state()

    def _config(self) -> ChatStepConfig:
        return ChatStepConfig(
            temperature=self.temperature_input.value(),
            thinking=cast(bool | None, self.thinking_combo.currentData()),
            reasoning_effort=cast(str, self.effort_combo.currentData()),
            max_tokens=self.max_tokens_input.value(),
            stream=self.stream_checkbox.isChecked(),
            retry_count=self.retry_input.value(),
            timeout_seconds=self.timeout_input.value() or None,
        )

    def _sync_effort_state(self) -> None:
        self.effort_combo.setEnabled(self.thinking_combo.currentData() is not False)

    def _reset(self) -> None:
        config = self.backend.reset_step_config(self.definition.step_id)
        self._set_values(config)
        self.on_changed()

    def _save(self) -> None:
        self.backend.set_step_config(self.definition.step_id, self._config())
        self.on_changed()
        self.accept()


def _credential_source_label(source: str) -> str:
    return {
        "credential_store": "Windows 凭据库",
        "environment": "环境变量",
        "key_test": "key_test.txt",
        "missing": "未配置",
    }.get(source, source)
