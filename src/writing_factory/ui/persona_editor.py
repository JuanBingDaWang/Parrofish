"""独立的 PersonaSpec JSON 编辑与 Markdown 预览窗口。"""

from __future__ import annotations

import json
from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from writing_factory.distill.models import PersonaSpec
from writing_factory.distill.runtime import RuntimePersonaSpec
from writing_factory.ui.time_format import format_china_datetime

PersonaLoader = Callable[[str], tuple[PersonaSpec, str] | None]
PersonaSaver = Callable[[str, PersonaSpec], tuple[PersonaSpec, str]]
RuntimePersonaLoader = Callable[[str], RuntimePersonaSpec | None]
PersonaVersionLoader = Callable[[str], list[dict[str, object]]]


class PersonaEditorWindow(QMainWindow):
    """编辑完整结构化档案，并保持派生 Markdown 与 JSON 一致。"""

    saved = pyqtSignal(str)

    def __init__(
        self,
        persona_id: str,
        *,
        load_persona: PersonaLoader,
        save_persona: PersonaSaver,
        load_runtime_persona: RuntimePersonaLoader | None = None,
        list_persona_versions: PersonaVersionLoader | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.persona_id = persona_id
        self._load_persona = load_persona
        self._save_persona = save_persona
        self._load_runtime_persona = load_runtime_persona
        self._list_persona_versions = list_persona_versions
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("档案详情")
        self.setMinimumSize(760, 560)
        self.resize(920, 720)
        self._build_ui()
        self.reload()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        toolbar = QHBoxLayout()
        title = QLabel("档案详情")
        title.setObjectName("pageTitle")
        toolbar.addWidget(title)
        toolbar.addStretch(1)
        self.reload_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload),
            "重新载入",
        )
        self.reload_button.setToolTip("放弃未保存修改并重新载入")
        self.reload_button.clicked.connect(self.reload)
        toolbar.addWidget(self.reload_button)
        self.save_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "保存",
        )
        self.save_button.clicked.connect(self.save)
        toolbar.addWidget(self.save_button)
        layout.addLayout(toolbar)

        self.tabs = QTabWidget()
        self.json_editor = QPlainTextEdit()
        self.json_editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.json_editor.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.markdown_preview = QPlainTextEdit()
        self.markdown_preview.setReadOnly(True)
        self.markdown_preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.runtime_preview = QPlainTextEdit()
        self.runtime_preview.setReadOnly(True)
        self.runtime_preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.version_preview = QPlainTextEdit()
        self.version_preview.setReadOnly(True)
        self.tabs.addTab(self.json_editor, "结构化档案")
        self.tabs.addTab(self.markdown_preview, "Markdown")
        self.tabs.addTab(self.runtime_preview, "运行时档案")
        self.tabs.addTab(self.version_preview, "版本历史")
        layout.addWidget(self.tabs, 1)

        self.status_label = QLabel()
        self.status_label.setObjectName("mutedText")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.setCentralWidget(root)

    def reload(self) -> None:
        """从 SQLite 重新载入当前可用档案。"""

        loaded = self._load_persona(self.persona_id)
        if loaded is None:
            self.json_editor.clear()
            self.markdown_preview.clear()
            self.runtime_preview.clear()
            self.version_preview.clear()
            self.json_editor.setEnabled(False)
            self.save_button.setEnabled(False)
            self.status_label.setText("档案不存在或尚未完成")
            return
        persona, markdown = loaded
        payload = json.dumps(persona.model_dump(mode="json"), ensure_ascii=False, indent=2)
        self.json_editor.setPlainText(payload)
        self.markdown_preview.setPlainText(markdown)
        self._reload_derived_tabs()
        self.json_editor.setEnabled(True)
        self.save_button.setEnabled(True)
        self.status_label.setText(f"{persona.name} · {len(persona.mental_models)} 个心智模型")

    def save(self) -> None:
        """校验并保存 JSON；错误显示在窗口内，不弹阻塞对话框。"""

        try:
            persona = PersonaSpec.model_validate_json(self.json_editor.toPlainText())
            if persona.id != self.persona_id:
                raise ValueError("档案 id 不允许修改")
            saved, markdown = self._save_persona(self.persona_id, persona)
        except Exception as exc:
            self.status_label.setText(f"保存失败：{str(exc)[:500]}")
            return
        payload = json.dumps(saved.model_dump(mode="json"), ensure_ascii=False, indent=2)
        self.json_editor.setPlainText(payload)
        self.markdown_preview.setPlainText(markdown)
        self._reload_derived_tabs()
        self.status_label.setText("已保存；旧自检分数已失效")
        self.saved.emit(self.persona_id)

    def _reload_derived_tabs(self) -> None:
        runtime = (
            self._load_runtime_persona(self.persona_id)
            if self._load_runtime_persona is not None
            else None
        )
        self.runtime_preview.setPlainText(
            json.dumps(runtime.model_dump(mode="json"), ensure_ascii=False, indent=2)
            if runtime is not None
            else "尚无运行时安全投影"
        )
        versions = (
            self._list_persona_versions(self.persona_id)
            if self._list_persona_versions is not None
            else []
        )
        self.version_preview.setPlainText(
            "\n".join(
                f"v{item.get('version_number')} · {item.get('status')} · "
                f"{format_china_datetime(item.get('research_date') or item.get('updated_at'))}"
                for item in versions
            )
            or "暂无版本历史"
        )
