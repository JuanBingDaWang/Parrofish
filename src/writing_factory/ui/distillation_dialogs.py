"""Focused dialogs and controls for persona quality and version upgrades."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from writing_factory.distill.models import PersonaMode
from writing_factory.distill.options import DistillationOptions
from writing_factory.ui.widgets import NoWheelComboBox

DISTILLATION_STEP_HELP = (
    (
        "跨文档复现与聚类",
        "按完整文档归并候选，再判断同一机制是否跨文档复现。",
        "约每篇 1 次画像 + 1 次聚类",
        "关闭后只形成基础候选，不标为跨文档稳定。",
    ),
    (
        "留出语料生成力验证",
        "用其余语料提炼出的候选预测留出文档的选择和组织方式。",
        "约 1 次，通常 3–12 分钟",
        "关闭后不声称模型能预测作者对新问题的处理路径。",
    ),
    (
        "对照语料排他性验证",
        "与同领域对照文本比较，区分作者特征和通用非虚构惯例。",
        "约每篇对照画像 + 1 次验证",
        "关闭后区分度标为未验证；没有对照语料时不可用。",
    ),
    (
        "完整谋篇 DNA",
        "提炼全文、章节、段落、句群和过渡层级的结构规则。",
        "约每篇 1 次分析 + 1 次归并",
        "关闭后生成阶段只使用心智模型和表达 DNA 规划结构。",
    ),
)


class DistillationQualityHelpDialog(QDialog):
    """Explain optional distillation work, cost, and confidence consequences."""

    def __init__(
        self,
        *,
        target_count: int,
        control_count: int,
        concurrency: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("蒸馏质量步骤说明")
        self.setMinimumSize(760, 440)
        self.resize(880, 540)
        layout = QVBoxLayout(self)
        intro = QLabel(
            f"当前选择：目标 {target_count} 篇、对照 {control_count} 篇、并发 {concurrency}。"
            "耗时会随篇幅、模型负载和重试变化；基础 Map、最终 Reduce、表达 DNA "
            "和本地质量门始终执行。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        recommendation = QLabel(
            "语料数量建议：1–2 篇通常只能形成暂定观察，3–5 篇可建立初步档案，"
            "6–12 篇代表性目标语料通常最能兼顾质量与效率。需要提炼某种文体的谋篇 DNA 时，"
            "该文体最好至少有 3 篇。对照语料建议选择 4–8 篇同领域、同文体、非目标作者文本。\n"
            "选择质量比单纯增加数量更重要：优先覆盖目标作者的代表主题、时期和实际需要生成的文体，"
            "不要用大量高度重复的文本凑数。超过 12 篇后通常进入收益递减区，可先完成一版，"
            "再每次加入 2–5 篇新语料进行升级；若核心模型和谋篇规律不再明显变化，就已接近稳定。"
        )
        recommendation.setObjectName("mutedText")
        recommendation.setWordWrap(True)
        layout.addWidget(recommendation)
        self.table = QTableWidget(len(DISTILLATION_STEP_HELP), 4)
        self.table.setHorizontalHeaderLabels(("步骤", "负责内容", "预计成本", "关闭影响"))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        for row, values in enumerate(DISTILLATION_STEP_HELP):
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.resizeRowsToContents()
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class DistillationQualityPanel(QWidget):
    """Preset and custom switches shared by new and upgraded versions."""

    options_changed = pyqtSignal()

    def __init__(
        self,
        *,
        counts_provider: Callable[[], tuple[int, int]],
        concurrency_provider: Callable[[], int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._counts_provider = counts_provider
        self._concurrency_provider = concurrency_provider
        self._updating = False
        self._mode: PersonaMode = "person"
        self._has_control = False
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        root.addWidget(QLabel("质量模式"))
        self.preset_combo = NoWheelComboBox()
        self.preset_combo.addItem("快速", "fast")
        self.preset_combo.addItem("均衡", "balanced")
        self.preset_combo.addItem("深度", "deep")
        self.preset_combo.addItem("自定义", "custom")
        self.preset_combo.setCurrentIndex(self.preset_combo.findData("balanced"))
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        root.addWidget(self.preset_combo)
        label_row = QHBoxLayout()
        label_row.setSpacing(4)
        label_row.addWidget(QLabel("质量步骤"))
        self.help_button = QToolButton()
        self.help_button.setText("?")
        self.help_button.setFixedSize(24, 24)
        self.help_button.setToolTip("查看蒸馏质量步骤、成本和关闭影响")
        self.help_button.setAccessibleName("蒸馏质量步骤说明")
        self.help_button.clicked.connect(self._show_help)
        label_row.addWidget(self.help_button)
        root.addLayout(label_row)
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        self.cross_document_checkbox = QCheckBox("跨文档复现")
        self.generative_checkbox = QCheckBox("生成力验证")
        self.exclusivity_checkbox = QCheckBox("排他性验证")
        self.composition_checkbox = QCheckBox("谋篇 DNA")
        self.checkboxes = (
            self.cross_document_checkbox,
            self.generative_checkbox,
            self.exclusivity_checkbox,
            self.composition_checkbox,
        )
        for index, checkbox in enumerate(self.checkboxes):
            grid.addWidget(checkbox, index // 3, index % 3)
            checkbox.toggled.connect(self._customized)
        root.addWidget(grid_widget, 1)
        self._apply_preset()

    def set_context(self, *, mode: PersonaMode, has_control: bool) -> None:
        self._mode = mode
        self._has_control = has_control
        self._apply_preset()

    def options(self) -> DistillationOptions:
        return self.options_for_context(mode=self._mode, has_control=self._has_control)

    def options_for_context(
        self,
        *,
        mode: PersonaMode,
        has_control: bool,
    ) -> DistillationOptions:
        """Read widgets on the GUI thread and normalize for the actual run inputs."""

        preset = str(self.preset_combo.currentData() or "balanced")
        if preset != "custom":
            return DistillationOptions.from_preset(
                preset, has_control_corpus=has_control
            ).normalized(mode=mode, has_control_corpus=has_control)
        return DistillationOptions(
            preset="custom",
            cross_document_validation=(
                mode == "person" and self.cross_document_checkbox.isChecked()
            ),
            generative_validation=(
                mode == "person" and self.generative_checkbox.isChecked()
            ),
            exclusivity_validation=(
                mode == "person"
                and has_control
                and self.exclusivity_checkbox.isChecked()
            ),
            composition_dna=self.composition_checkbox.isChecked(),
        )

    def set_options(self, options: DistillationOptions) -> None:
        preset = options.preset if options.preset != "legacy" else "deep"
        index = self.preset_combo.findData(preset)
        self.preset_combo.setCurrentIndex(max(0, index))
        if preset == "custom":
            self._updating = True
            try:
                values = (
                    options.cross_document_validation,
                    options.generative_validation,
                    options.exclusivity_validation,
                    options.composition_dna,
                )
                for checkbox, checked in zip(self.checkboxes, values, strict=True):
                    checkbox.setChecked(checked)
            finally:
                self._updating = False
            self._update_enabled()

    def _apply_preset(self) -> None:
        preset = str(self.preset_combo.currentData() or "balanced")
        options = DistillationOptions.from_preset(
            preset if preset in {"fast", "balanced", "deep", "custom"} else "balanced",
            has_control_corpus=self._has_control,
        ).normalized(mode=self._mode, has_control_corpus=self._has_control)
        if preset != "custom":
            self._updating = True
            try:
                values = (
                    options.cross_document_validation,
                    options.generative_validation,
                    options.exclusivity_validation,
                    options.composition_dna,
                )
                for checkbox, checked in zip(self.checkboxes, values, strict=True):
                    checkbox.setChecked(checked)
            finally:
                self._updating = False
        self._update_enabled()
        self.options_changed.emit()

    def _update_enabled(self) -> None:
        custom = self.preset_combo.currentData() == "custom"
        person = self._mode == "person"
        self._updating = True
        try:
            if custom and not person:
                self.cross_document_checkbox.setChecked(False)
            cross = self.cross_document_checkbox.isChecked()
            if custom and (not person or not cross):
                self.generative_checkbox.setChecked(False)
                self.exclusivity_checkbox.setChecked(False)
            elif custom and not self._has_control:
                self.exclusivity_checkbox.setChecked(False)
        finally:
            self._updating = False
        self.cross_document_checkbox.setEnabled(custom and person)
        cross = self.cross_document_checkbox.isChecked()
        self.generative_checkbox.setEnabled(custom and person and cross)
        self.exclusivity_checkbox.setEnabled(custom and person and cross and self._has_control)
        self.composition_checkbox.setEnabled(custom)

    def _customized(self) -> None:
        if self._updating:
            return
        if self.preset_combo.currentData() != "custom":
            index = self.preset_combo.findData("custom")
            self.preset_combo.setCurrentIndex(index)
        self._update_enabled()
        self.options_changed.emit()

    def _show_help(self) -> None:
        target_count, control_count = self._counts_provider()
        DistillationQualityHelpDialog(
            target_count=target_count,
            control_count=control_count,
            concurrency=self._concurrency_provider(),
            parent=self,
        ).exec()


@dataclass(frozen=True, slots=True)
class UpgradeSelection:
    target_doc_ids: frozenset[str]
    control_doc_ids: frozenset[str]
    domain: str
    strategy: str


class PersonaUpgradeDialog(QDialog):
    """Select incremental or replacement source roles for a new persona version."""

    def __init__(
        self,
        *,
        sources: list[dict[str, object]],
        target_doc_ids: frozenset[str],
        control_doc_ids: frozenset[str],
        domain: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("升级作者档案")
        self.setMinimumSize(720, 480)
        self.resize(820, 600)
        self._original_targets = target_doc_ids
        self._original_controls = control_doc_ids
        self._missing_originals = (target_doc_ids | control_doc_ids) - {
            str(item.get("doc_id")) for item in sources
        }
        self._updating = False
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("升级方式"))
        self.strategy_combo = NoWheelComboBox()
        self.strategy_combo.addItem("增量升级", "incremental")
        self.strategy_combo.addItem("重新构建版本", "rebuild")
        self.strategy_combo.currentIndexChanged.connect(self._strategy_changed)
        top.addWidget(self.strategy_combo)
        top.addWidget(QLabel("内容领域"))
        self.domain_input = QLineEdit(domain)
        self.domain_input.setMaxLength(80)
        self.domain_input.textChanged.connect(self._validate)
        top.addWidget(self.domain_input, 1)
        layout.addLayout(top)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(("目标", "对照", "文件", "原角色"))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        ready_sources = [item for item in sources if item.get("status") == "ready"]
        self.table.setRowCount(len(ready_sources))
        for row, source in enumerate(ready_sources):
            doc_id = str(source.get("doc_id"))
            target = self._role_item(doc_id, doc_id in target_doc_ids)
            control = self._role_item(doc_id, doc_id in control_doc_ids)
            self.table.setItem(row, 0, target)
            self.table.setItem(row, 1, control)
            self.table.setItem(row, 2, QTableWidgetItem(str(source.get("filename", ""))))
            if doc_id in target_doc_ids:
                original = "目标"
            elif doc_id in control_doc_ids:
                original = "对照"
            else:
                original = "新增"
            self.table.setItem(row, 3, QTableWidgetItem(original))
        self.table.itemChanged.connect(self._role_changed)
        layout.addWidget(self.table, 1)
        self.status_label = QLabel("")
        self.status_label.setObjectName("mutedText")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._strategy_changed()

    def selection(self) -> UpgradeSelection:
        return UpgradeSelection(
            target_doc_ids=frozenset(self._checked_ids(0)),
            control_doc_ids=frozenset(self._checked_ids(1)),
            domain=self.domain_input.text().strip(),
            strategy=str(self.strategy_combo.currentData() or "incremental"),
        )

    @staticmethod
    def _role_item(doc_id: str, checked: bool) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        item.setData(Qt.ItemDataRole.UserRole, doc_id)
        return item

    def _strategy_changed(self) -> None:
        incremental = self.strategy_combo.currentData() == "incremental"
        self._updating = True
        try:
            for row in range(self.table.rowCount()):
                doc_id = str(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
                original = doc_id in self._original_targets or doc_id in self._original_controls
                if incremental and original:
                    self.table.item(row, 0).setCheckState(
                        Qt.CheckState.Checked
                        if doc_id in self._original_targets
                        else Qt.CheckState.Unchecked
                    )
                    self.table.item(row, 1).setCheckState(
                        Qt.CheckState.Checked
                        if doc_id in self._original_controls
                        else Qt.CheckState.Unchecked
                    )
                for column in (0, 1):
                    item = self.table.item(row, column)
                    flags = Qt.ItemFlag.ItemIsEnabled
                    if not (incremental and original):
                        flags |= Qt.ItemFlag.ItemIsUserCheckable
                    item.setFlags(flags)
        finally:
            self._updating = False
        self._validate()

    def _role_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() not in (0, 1):
            return
        if item.checkState() == Qt.CheckState.Checked:
            other = self.table.item(item.row(), 1 - item.column())
            self._updating = True
            other.setCheckState(Qt.CheckState.Unchecked)
            self._updating = False
        self._validate()

    def _checked_ids(self, column: int) -> set[str]:
        identifiers: set[str] = set()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, column)
            if item.checkState() == Qt.CheckState.Checked:
                identifiers.add(str(item.data(Qt.ItemDataRole.UserRole)))
        return identifiers

    def _validate(self) -> None:
        target = self._checked_ids(0)
        control = self._checked_ids(1)
        incremental = self.strategy_combo.currentData() == "incremental"
        valid = bool(target)
        message = f"目标 {len(target)} 篇 · 对照 {len(control)} 篇"
        if incremental and self._missing_originals:
            valid = False
            message = "原版本有语料已被删除，请改用“重新构建版本”"
        elif incremental and not (
            (target | control) - (self._original_targets | self._original_controls)
        ):
            valid = False
            message = "请至少加入一篇新语料"
        elif control and not self.domain_input.text().strip():
            valid = False
            message = "使用对照语料时必须填写内容领域"
        self.status_label.setText(message)
        button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if button is not None:
            button.setEnabled(valid)
