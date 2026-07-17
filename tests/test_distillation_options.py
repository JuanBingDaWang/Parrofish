"""Quality presets and author-profile upgrade dialog tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialogButtonBox, QLabel

from writing_factory.distill.options import DistillationOptions
from writing_factory.ui.distillation_dialogs import (
    DISTILLATION_STEP_HELP,
    DistillationQualityHelpDialog,
    DistillationQualityPanel,
    PersonaUpgradeDialog,
)


def test_distillation_presets_have_documented_optional_steps() -> None:
    fast = DistillationOptions.from_preset("fast")
    balanced = DistillationOptions.from_preset("balanced")
    deep_without_control = DistillationOptions.from_preset("deep")
    deep_with_control = DistillationOptions.from_preset(
        "deep",
        has_control_corpus=True,
    )

    assert not any(
        (
            fast.cross_document_validation,
            fast.generative_validation,
            fast.exclusivity_validation,
            fast.composition_dna,
        )
    )
    assert balanced.cross_document_validation
    assert balanced.composition_dna
    assert not balanced.generative_validation
    assert not balanced.exclusivity_validation
    assert deep_without_control.generative_validation
    assert not deep_without_control.exclusivity_validation
    assert deep_with_control.exclusivity_validation


def test_distillation_options_enforce_dependencies_and_source_capabilities() -> None:
    with pytest.raises(ValidationError, match="依赖跨文档"):
        DistillationOptions(
            preset="custom",
            cross_document_validation=False,
            generative_validation=True,
        )

    topic = DistillationOptions.from_preset(
        "deep",
        has_control_corpus=True,
    ).normalized(mode="topic", has_control_corpus=True)
    assert topic.cross_document_validation
    assert topic.generative_validation
    assert not topic.exclusivity_validation
    assert topic.composition_dna


def test_quality_help_describes_every_optional_step(qtbot) -> None:
    dialog = DistillationQualityHelpDialog(
        target_count=10,
        control_count=5,
        concurrency=3,
    )
    qtbot.addWidget(dialog)

    assert dialog.table.rowCount() == len(DISTILLATION_STEP_HELP)
    assert dialog.table.item(0, 0).text() == "跨文档复现与聚类"
    assert any("基础 Map" in label.text() for label in dialog.findChildren(QLabel))
    assert any("6–12 篇" in label.text() for label in dialog.findChildren(QLabel))
    assert any("4–8 篇" in label.text() for label in dialog.findChildren(QLabel))


def test_quality_panel_supports_topic_cross_document_and_generative_steps(qtbot) -> None:
    panel = DistillationQualityPanel(
        counts_provider=lambda: (10, 3),
        concurrency_provider=lambda: 4,
    )
    qtbot.addWidget(panel)
    panel.preset_combo.setCurrentIndex(panel.preset_combo.findData("custom"))
    panel.cross_document_checkbox.setChecked(True)
    panel.generative_checkbox.setChecked(True)

    panel.set_context(mode="topic", has_control=True)

    assert panel.help_button.accessibleName() == "蒸馏质量步骤说明"
    assert panel.cross_document_checkbox.isChecked()
    assert panel.generative_checkbox.isChecked()
    assert panel.cross_document_checkbox.isEnabled()
    assert panel.generative_checkbox.isEnabled()
    assert panel.options().cross_document_validation
    assert panel.options().generative_validation
    assert not panel.options().exclusivity_validation

    panel.preset_combo.setCurrentIndex(panel.preset_combo.findData("deep"))
    assert panel.cross_document_checkbox.isChecked()
    assert panel.generative_checkbox.isChecked()
    assert panel.composition_checkbox.isChecked()
    assert not panel.exclusivity_checkbox.isChecked()


def test_upgrade_dialog_restores_original_roles_when_returning_to_incremental(qtbot) -> None:
    dialog = PersonaUpgradeDialog(
        sources=[
            {"doc_id": "old", "filename": "旧稿.pdf", "status": "ready"},
            {"doc_id": "new", "filename": "新稿.pdf", "status": "ready"},
        ],
        target_doc_ids=frozenset({"old"}),
        control_doc_ids=frozenset(),
        domain="",
    )
    qtbot.addWidget(dialog)
    ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok is not None and not ok.isEnabled()

    dialog.strategy_combo.setCurrentIndex(dialog.strategy_combo.findData("rebuild"))
    dialog.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    dialog.table.item(0, 1).setCheckState(Qt.CheckState.Checked)
    dialog.strategy_combo.setCurrentIndex(dialog.strategy_combo.findData("incremental"))

    assert dialog.table.item(0, 0).checkState() == Qt.CheckState.Checked
    assert dialog.table.item(0, 1).checkState() == Qt.CheckState.Unchecked
    assert not bool(dialog.table.item(0, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable)
    dialog.table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    assert ok.isEnabled()
    assert dialog.selection().target_doc_ids == frozenset({"old", "new"})
