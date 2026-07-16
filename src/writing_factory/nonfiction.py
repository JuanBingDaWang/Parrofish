"""Shared nonfiction genre vocabulary and task-level genre inference."""

from __future__ import annotations

import re
from typing import Literal, cast

NonfictionGenre = Literal[
    "general_nonfiction",
    "academic_paper",
    "research_report",
    "policy_brief",
    "commentary",
    "review",
    "popular_science",
    "speech",
    "public_article",
    "news_analysis",
    "instructional",
    "summary",
    "other_nonfiction",
]

GENRE_OPTIONS: tuple[tuple[NonfictionGenre, str], ...] = (
    ("general_nonfiction", "通用非虚构文本"),
    ("academic_paper", "学术论文"),
    ("research_report", "研究 / 调研报告"),
    ("policy_brief", "政策建议 / 决策简报"),
    ("commentary", "评论 / 观点文章"),
    ("review", "书评 / 作品评论"),
    ("popular_science", "科普 / 知识解释"),
    ("speech", "演讲 / 发言稿"),
    ("public_article", "公众文章 / 公众号文章"),
    ("news_analysis", "新闻分析 / 特稿"),
    ("instructional", "教程 / 指南"),
    ("summary", "摘要 / 内容提要"),
    ("other_nonfiction", "其他非虚构文本"),
)

GENRE_LABELS: dict[NonfictionGenre, str] = dict(GENRE_OPTIONS)

GENRE_GUIDANCE: dict[NonfictionGenre, str] = {
    "general_nonfiction": "根据用户目的和受众组织清晰、连贯、事实受控的非虚构文本。",
    "academic_paper": "形成可检验的中心论点、明确的概念边界和层层推进的论证结构。",
    "research_report": "围绕调查问题组织方法、发现、解释和建议，区分观察结果与分析判断。",
    "policy_brief": "面向决策者先呈现问题与结论，再给出证据、选项、权衡和可执行建议。",
    "commentary": "明确提出判断，以事实和推理推进观点，同时认真处理反方意见和适用边界。",
    "review": "先交代评价对象与标准，再分析关键特征、贡献、局限及其更广泛意义。",
    "popular_science": "从受众可理解的问题切入，用准确解释、例证和递进层次降低理解门槛。",
    "speech": "适合口头表达，以清晰主线、听众意识、节奏变化和可记忆的收束组织内容。",
    "public_article": "兼顾可读性与信息密度，用明确切口、自然过渡和具体解释维持阅读推进。",
    "news_analysis": "区分已知事实、背景和分析，按重要性与因果关系组织，不把推测写成事实。",
    "instructional": "以读者任务为中心，按前提、步骤、检查点和常见错误组织可执行说明。",
    "summary": "忠实压缩原材料，保留目的、核心信息、关键限定与结论，不引入新判断。",
    "other_nonfiction": "严格服从用户指定的非虚构文体、受众、目的和格式约束。",
}


def genre_label(genre: NonfictionGenre) -> str:
    """Return the stable Simplified Chinese display label for a genre."""

    return GENRE_LABELS[genre]


def infer_nonfiction_genre(task: str) -> NonfictionGenre:
    """Infer a conservative nonfiction genre from explicit task wording."""

    patterns: tuple[tuple[NonfictionGenre, str], ...] = (
        ("academic_paper", r"论文|学术文章|文献综述|期刊文章|学位论文"),
        ("research_report", r"研究报告|调研报告|调查报告|评估报告|分析报告|白皮书"),
        ("policy_brief", r"政策建议|政策简报|决策简报|资政报告|内参|咨政"),
        ("speech", r"演讲稿|发言稿|讲话稿|致辞|讲座稿"),
        ("review", r"书评|影评|剧评|作品评论|评介"),
        ("popular_science", r"科普|知识解释|通俗解释|面向公众解释"),
        ("news_analysis", r"新闻分析|新闻特稿|深度报道|新闻评论"),
        ("instructional", r"教程|指南|操作说明|使用说明|步骤说明|手册"),
        ("summary", r"摘要|内容提要|概要|梗概|要点提炼"),
        ("public_article", r"公众号|推文|专栏文章|公众文章|博客文章"),
        ("commentary", r"评论文章|观点文章|短评|时评|社论|评论"),
    )
    for genre, pattern in patterns:
        if re.search(pattern, task, flags=re.IGNORECASE):
            return genre
    return cast(NonfictionGenre, "general_nonfiction")
