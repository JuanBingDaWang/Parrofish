"""Reference dialog for task-level writing quality steps."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

QUALITY_STEP_HELP = (
    ("HyDE", "先生成假设性文本再检索，提高本地知识库中抽象问题的召回率。", "每次检索约 5–20 秒"),
    ("查询改写", "把任务扩展为多个子查询并融合本地知识库结果，减少漏检。", "每次检索约 5–20 秒"),
    ("选题锐化", "结合作者档案和知识库，明确文体、受众、目的和中心信息。", "约 10–60 秒"),
    ("内容规划", "按目标文体生成带内容功能、单元关系和证据候选的内容规划。", "约 30 秒–5 分钟"),
    (
        "事实核验",
        "由不带作者档案的中性模型逐项核对事实论断，必要时触发修订。",
        "每单元约 10–60 秒",
    ),
    (
        "单元打磨",
        "事实核验后按作者表达 DNA 打磨各内容单元，不改变冻结事实。",
        "每单元约 20–120 秒",
    ),
    (
        "打磨防漂移",
        "中性模型检查内容单元打磨是否改变事实，发现漂移时回退。",
        "每单元约 5–30 秒",
    ),
    ("术语审查", "检查全文术语的定义、称谓和使用是否一致。", "全文约 10–60 秒"),
    ("结构审查", "按目标文体检查内容单元衔接、结构缺口和中心信息的一致性。", "全文约 10–60 秒"),
    ("全局打磨", "在各内容单元完成后统一过渡、语气和全文表达。", "约 30 秒–3 分钟"),
    ("全局防漂移", "中性模型检查全局打磨是否改动已核验事实，异常时回退。", "全文约 10–60 秒"),
)


class QualityStepsHelpDialog(QDialog):
    """Explain what each quality switch costs and protects."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("质量步骤说明")
        self.setMinimumSize(680, 430)
        self.resize(780, 540)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "耗时是单次正常请求的经验范围；文稿长度、内容单元数、网络状况和自动重试都会使总耗时增加。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(len(QUALITY_STEP_HELP), 3)
        self.table.setHorizontalHeaderLabels(("步骤", "负责内容", "预计耗时"))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        for row, values in enumerate(QUALITY_STEP_HELP):
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.resizeRowsToContents()
        layout.addWidget(self.table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
