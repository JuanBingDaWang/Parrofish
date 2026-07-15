"""Typed failures that stop a writing graph before committing a bad node."""

from __future__ import annotations


class PipelineNodeError(RuntimeError):
    """Report the failed node while preserving the original exception as cause."""

    def __init__(self, node_label: str, detail: str) -> None:
        self.node_label = node_label
        self.detail = detail.strip() or "未知错误"
        super().__init__(f"{self.node_label}失败：{self.detail}")
