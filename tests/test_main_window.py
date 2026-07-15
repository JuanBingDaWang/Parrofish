"""Desktop shell interaction tests."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtCore import QItemSelectionModel, Qt
from PyQt6.QtWidgets import QApplication, QLineEdit, QScrollArea

from tests.test_distill_pipeline import _persona
from writing_factory.distill.serialization import render_persona_markdown
from writing_factory.kb.models import FusedHit, IngestResult, RetrievalResult
from writing_factory.llm.configuration import STEP_DEFINITIONS
from writing_factory.llm.models import ChatResult, TokenUsage
from writing_factory.ui.help_ui import PageHelpDialog
from writing_factory.ui.main_window import MainWindow
from writing_factory.ui.quality_steps_help import QUALITY_STEP_HELP, QualityStepsHelpDialog
from writing_factory.ui.settings_dialogs import ProviderSettingsDialog, StepSettingsDialog


def test_connection_button_runs_check_in_background(qtbot) -> None:
    def check_connection() -> ChatResult:
        time.sleep(0.05)
        return ChatResult(
            content="OK",
            model="test-model",
            usage=TokenUsage(total_tokens=3),
        )

    window = MainWindow(check_connection)
    qtbot.addWidget(window)
    window.show()

    qtbot.mouseClick(window.check_button, Qt.MouseButton.LeftButton)
    assert not window.check_button.isEnabled()
    qtbot.waitUntil(lambda: window.siliconflow_status.text() == "可用", timeout=2000)

    assert window.check_button.isEnabled()
    assert "3 tokens" in window.statusBar().currentMessage()
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_settings_page_applies_shared_siliconflow_concurrency(qtbot) -> None:
    changed: list[int] = []
    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        get_siliconflow_concurrency=lambda: 5,
        set_siliconflow_concurrency=changed.append,
    )
    qtbot.addWidget(window)

    assert window.concurrency_input.value() == 5
    window.concurrency_input.setValue(6)

    assert changed == [6]


def test_settings_page_persists_global_siliconflow_request_timeout(qtbot) -> None:
    changed: list[int] = []
    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        get_siliconflow_request_timeout=lambda: 1200,
        set_siliconflow_request_timeout=changed.append,
    )
    qtbot.addWidget(window)

    assert window.siliconflow_timeout_input.value() == 1200
    window.siliconflow_timeout_input.setValue(1500)

    assert changed == [1500]


def test_settings_page_persists_author_chat_recent_rounds(qtbot) -> None:
    changed: list[int] = []
    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        get_author_chat_recent_rounds=lambda: 6,
        set_author_chat_recent_rounds=changed.append,
    )
    qtbot.addWidget(window)

    assert window.settings_page.chat_recent_rounds_input.value() == 6
    window.settings_page.chat_recent_rounds_input.setValue(8)

    assert changed == [8]


def test_settings_page_lists_all_steps_and_independent_editors(qtbot) -> None:
    window = MainWindow(lambda: ChatResult(content="OK", model="test"))
    qtbot.addWidget(window)
    page = window.settings_page

    assert sum(table.rowCount() for table in page.step_tables.values()) == len(STEP_DEFINITIONS)
    summary_header = page.step_tables["distill"].horizontalHeaderItem(1).text()
    assert summary_header.startswith("设置摘要\n温度")
    assert "思考｜强度" in summary_header
    assert "上限｜请求｜重试｜超时" in summary_header
    assert page.model_value_labels["embedding"].text() == "BAAI/bge-m3"
    assert page.model_value_labels["reranker"].text() == "BAAI/bge-reranker-v2-m3"

    definition = next(item for item in STEP_DEFINITIONS if item.step_id == "writing.draft")
    step_dialog = StepSettingsDialog(
        page.backend,
        definition,
        on_changed=page.refresh,
        parent=page,
    )
    qtbot.addWidget(step_dialog)
    step_dialog.temperature_input.setValue(0.9)
    step_dialog.thinking_combo.setCurrentIndex(step_dialog.thinking_combo.findData(True))
    step_dialog.effort_combo.setCurrentIndex(step_dialog.effort_combo.findData("max"))
    step_dialog.max_tokens_input.setValue(12288)
    step_dialog.stream_checkbox.setChecked(False)
    step_dialog.retry_input.setValue(4)
    step_dialog.timeout_input.setValue(1800)
    step_dialog._save()

    saved = page.backend.get_step_config("writing.draft")
    assert saved.temperature == 0.9
    assert saved.thinking is True
    assert saved.reasoning_effort == "max"
    assert saved.max_tokens == 12288
    assert not saved.stream
    assert saved.retry_count == 4
    assert saved.timeout_seconds == 1800

    provider_dialog = ProviderSettingsDialog(page.backend, "siliconflow", page)
    qtbot.addWidget(provider_dialog)
    assert provider_dialog.secret_input.echoMode() == QLineEdit.EchoMode.Password
    provider_dialog.secret_input.setText("local-test-secret")
    provider_dialog.base_url_input.setText("https://example.invalid/v1")
    provider_dialog._save()
    snapshot = page.backend.provider_snapshot("siliconflow")
    assert snapshot["configured"] is True
    assert snapshot["base_url"] == "https://example.invalid/v1"


def test_settings_credentials_help_explains_api_signup_and_allows_its_links(qtbot) -> None:
    window = MainWindow(lambda: ChatResult(content="OK", model="test"))
    qtbot.addWidget(window)
    page = window.settings_page
    dialog = PageHelpDialog("credentials", page)
    qtbot.addWidget(dialog)

    html = dialog.browser.toHtml()
    plain_text = dialog.browser.toPlainText()
    assert page.credentials_help_button.accessibleName() == "API 获取与配置帮助"
    assert page.credentials_help_button.toolTip() == "查看API 获取与配置功能介绍和操作教程"
    assert dialog.browser.openExternalLinks()
    assert "https://cloud.siliconflow.cn/i/j7F36Uco" in html
    assert "https://mineru.net/" in html
    assert "以 sk- 开头" in plain_text
    assert "API 密钥和 Token 相当于账号密码" in plain_text

    ordinary_help = PageHelpDialog("settings", page)
    qtbot.addWidget(ordinary_help)
    assert not ordinary_help.browser.openExternalLinks()


def test_writing_page_previews_isolated_target_sources_and_uses_scroll_regions(qtbot) -> None:
    documents = [
        {"doc_id": "target_a", "filename": "目标一.pdf"},
        {"doc_id": "target_b", "filename": "目标二.pdf"},
        {"doc_id": "control", "filename": "对照.pdf"},
    ]

    def preview(persona_id, selected, explicitly_allowed):
        assert persona_id == "persona"
        excluded = {"target_a", "target_b"} - explicitly_allowed
        return {
            "selected_count": len(selected),
            "isolated_count": len(selected & excluded),
            "usable_count": len(selected - excluded),
        }

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_documents=lambda: documents,
        list_personas=lambda: [{"persona_id": "persona", "name": "测试作者"}],
        run_writing_pipeline=lambda **_kwargs: {},
        preview_source_selection=preview,
    )
    qtbot.addWidget(window)
    page = window.writing_task_page
    page.persona_combo.setCurrentIndex(1)
    for index in range(page.document_list.count()):
        page.document_list.item(index).setCheckState(Qt.CheckState.Checked)

    assert page.source_summary_label.text() == "已选 3 篇 · 隔离 2 篇 · 实际可用 1 篇"
    document_row, _role = page.config_layout.getWidgetPosition(page.document_list)
    summary_row, _role = page.config_layout.getWidgetPosition(page.source_summary_label)
    isolation_row, _role = page.config_layout.getWidgetPosition(page.allow_persona_sources)
    assert document_row < summary_row < isolation_row
    assert page.start_button.isEnabled()
    assert page.main_splitter.orientation() == Qt.Orientation.Horizontal
    assert page.main_splitter.widget(0) is page.history_scroll
    assert isinstance(page.history_scroll, QScrollArea)
    assert page.history_scroll.minimumWidth() >= 250
    assert isinstance(page.progress_scroll, QScrollArea)
    assert page.workspace_tabs.count() == 3
    assert page.document_list.minimumHeight() >= 180
    assert page.task_table.maximumHeight() > 1000
    assert page.section_table.maximumHeight() > 1000
    window.navigation.setCurrentRow(4)
    window.resize(960, 640)
    window.show()
    qtbot.waitUntil(
        lambda: page.config_scroll.verticalScrollBar().maximum() > 0,
        timeout=2000,
    )
    page.workspace_tabs.setCurrentIndex(page._progress_workspace_index)
    assert page.progress_tabs.isVisible()
    page._pipeline_streamed("content", "实时内容")
    qtbot.waitUntil(
        lambda: page.progress_scroll.verticalScrollBar().value()
        == page.progress_scroll.verticalScrollBar().maximum(),
        timeout=2000,
    )

    page.document_list.item(2).setCheckState(Qt.CheckState.Unchecked)
    assert page.source_summary_label.text() == "已选 2 篇 · 隔离 2 篇 · 实际可用 0 篇"
    assert not page.start_button.isEnabled()

    page.allow_persona_sources.setChecked(True)
    assert page.source_summary_label.text() == "已选 2 篇 · 隔离 0 篇 · 实际可用 2 篇"
    assert page.start_button.isEnabled()


def test_loading_checkpoint_tracks_current_section(qtbot) -> None:
    state = {
        "status": "drafting",
        "current_section_index": 1,
        "sections": [
            {"section_id": "1", "heading": "第一节", "status": "polished"},
            {"section_id": "2", "heading": "第二节", "status": "drafting"},
        ],
    }
    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_projects=lambda: [{"project_id": "project", "title": "项目"}],
        list_writing_tasks=lambda _project_id: [
            {
                "task_id": "task",
                "title": "任务",
                "status": "error",
                "updated_at": "2026-07-14T06:17:37+00:00",
            }
        ],
        load_writing_task=lambda _task_id: {
            "task_id": "task",
            "title": "任务",
            "task_description": "要求",
            "domain": "领域",
            "selected_doc_ids": set(),
            "allowed_persona_doc_ids": set(),
            "generation_options": {},
            "state": state,
        },
    )
    qtbot.addWidget(window)
    page = window.writing_task_page
    page.task_table.selectRow(0)

    page._load_selected_task()

    assert page._current_section_index == 1
    assert page.section_table.currentRow() == 1


def test_selecting_history_task_loads_only_its_persisted_outputs(qtbot) -> None:
    records = [
        {
            "task_id": "done_task",
            "title": "测试：段落",
            "status": "done",
            "updated_at": "2026-07-14T15:44:06+00:00",
        },
        {
            "task_id": "partial_task",
            "title": "摘要：数字出版研究综述",
            "status": "running",
            "updated_at": "2026-07-14T14:45:51+00:00",
        },
        {
            "task_id": "empty_task",
            "title": "尚未起草",
            "status": "running",
            "updated_at": "2026-07-14T14:40:00+00:00",
        },
    ]
    polished_section = json.dumps(
        {
            "section_id": "1",
            "heading": "已经完成的第一节",
            "polished_text": "这是摘要任务已经持久化的部分正文。",
        },
        ensure_ascii=False,
    )
    states = {
        "done_task": {
            "status": "done",
            "sections": [],
            "final_draft_json": json.dumps(
                {
                    "title": "旧任务成稿",
                    "sections": [{"heading": "", "polished_text": "旧任务正文"}],
                },
                ensure_ascii=False,
            ),
            "outline_json": json.dumps(
                {"root_nodes": [{"heading": "旧任务规划", "children": []}]},
                ensure_ascii=False,
            ),
            "reference_list_json": json.dumps(
                {"style": "gb-t-7714", "items": [{"citation_text": "旧任务来源"}]},
                ensure_ascii=False,
            ),
        },
        "partial_task": {
            "status": "drafting",
            "sections": [
                {
                    "section_id": "1",
                    "heading": "第一节",
                    "status": "polished",
                    "polished_section_json": polished_section,
                }
            ],
        },
        "empty_task": {"status": "drafting", "sections": []},
    }
    evaluations = {
        "done_task": {"traceability": 1.0, "judge_rationale": "旧任务评估"},
    }
    loads: list[str] = []
    saved: list[tuple[str, str, str]] = []

    def load_task(task_id: str):
        loads.append(task_id)
        record = next(item for item in records if item["task_id"] == task_id)
        return {
            **record,
            "task_description": "测试要求",
            "domain": "",
            "selected_doc_ids": set(),
            "allowed_persona_doc_ids": set(),
            "generation_options": {},
            "state": states[task_id],
            "evaluation": evaluations.get(task_id),
        }

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_projects=lambda: [{"project_id": "project", "title": "项目"}],
        list_writing_tasks=lambda _project_id: records,
        load_writing_task=load_task,
        save_edited_draft=lambda task_id, draft, outline: saved.append(
            (task_id, draft, outline)
        ),
    )
    qtbot.addWidget(window)
    page = window.writing_task_page
    page.workspace_tabs.setCurrentIndex(page._results_workspace_index)

    page.task_table.selectRow(0)

    assert loads == ["done_task"]
    assert "旧任务正文" in page._draft_view.toPlainText()
    assert "旧任务规划" in page._outline_view.toPlainText()
    assert "旧任务来源" in page._ref_view.toPlainText()
    assert "旧任务评估" in page._eval_view.toPlainText()
    assert page.workspace_tabs.currentIndex() == page._results_workspace_index
    assert page.results_group.title() == "写作结果 · 测试：段落"

    page.task_table.selectRow(1)

    assert loads == ["done_task", "partial_task"]
    assert "摘要任务已经持久化" in page._draft_view.toPlainText()
    assert "旧任务正文" not in page._draft_view.toPlainText()
    assert page._outline_view.toPlainText() == ""
    assert page._ref_view.toPlainText() == ""
    assert page._eval_view.toPlainText() == ""
    assert page.results_group.title() == "写作结果 · 摘要：数字出版研究综述"

    page.task_table.selectRow(2)

    assert loads == ["done_task", "partial_task", "empty_task"]
    assert page._draft_view.toPlainText() == ""
    assert page._draft_view.placeholderText() == "该任务尚未生成正文"
    assert page._writing_task_id == "empty_task"
    assert page._displayed_task_id == "empty_task"
    assert page.save_draft_button.isEnabled()

    page._draft_view.setPlainText("新任务人工稿")
    page._save_draft()
    assert saved == [("empty_task", "新任务人工稿", "")]

    _select_additional_row(page.task_table, 0)
    assert not page.save_draft_button.isEnabled()
    page._displayed_task_id = "done_task"
    page._save_draft()
    assert saved == [("empty_task", "新任务人工稿", "")]
    assert "任务标识不一致" in window.statusBar().currentMessage()


def test_project_and_writing_times_display_as_east_eight(qtbot) -> None:
    project = {
        "project_id": "project",
        "title": "测试项目",
        "description": "",
        "task_count": 1,
        "updated_at": "2026-07-14T06:17:37+00:00",
    }
    task = {
        "task_id": "task",
        "title": "测试任务",
        "status": "error",
        "error": "超时",
        "updated_at": "2026-07-14T06:17:37+00:00",
    }
    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_projects=lambda: [project],
        list_writing_tasks=lambda _project_id: [task],
    )
    qtbot.addWidget(window)

    assert window.project_page.table.item(0, 4).text() == "2026-07-14 14:17:37"
    assert window.writing_task_page.task_table.item(0, 3).text() == "2026-07-14 14:17:37"
    window.writing_task_page.task_table.selectRow(0)
    assert window.writing_task_page.task_error_label.toPlainText() == "失败详情：超时"
    assert window.writing_task_page.task_error_label.maximumHeight() <= 96


def test_writing_failure_refreshes_history_without_green_full_progress(qtbot) -> None:
    task = {
        "task_id": "task",
        "title": "测试任务",
        "status": "running",
        "error": None,
        "updated_at": "2026-07-14T06:17:37+00:00",
    }
    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_projects=lambda: [{"project_id": "project", "title": "项目"}],
        list_writing_tasks=lambda _project_id: [task],
    )
    qtbot.addWidget(window)
    page = window.writing_task_page
    assert page.task_table.item(0, 2).text() == "运行中"
    page.progress_bar.setValue(100)
    task["status"] = "error"
    task["error"] = "框架生成超时"

    page._pipeline_failed("框架生成超时")

    assert page.task_table.item(0, 2).text() == "失败"
    assert page.progress_bar.value() == 99
    assert page.progress_bar.format() == "失败 · %p%"
    assert "#b42318" in page.progress_bar.styleSheet()


def test_writing_page_shows_elapsed_time_and_public_stream_only(qtbot) -> None:
    window = MainWindow(lambda: ChatResult(content="OK", model="test"))
    qtbot.addWidget(window)
    page = window.writing_task_page
    page._start_run_clock()
    page._run_started_at = time.monotonic() - 65
    page._step_started_at = time.monotonic() - 5

    page._pipeline_streamed("reasoning", "不应显示的推理文本")
    assert "不应显示" not in page.live_output_view.toPlainText()
    assert "模型最近活动" in page.activity_label.text()

    page._pipeline_streamed("content", '{"title":"')
    assert page.progress_tabs.currentIndex() == page._live_output_tab_index
    page.progress_tabs.setCurrentIndex(0)
    page._pipeline_streamed("content::全文结构审查", "结构清晰")
    assert page.progress_tabs.currentIndex() == 0
    page._pipeline_streamed("status::全文结构审查", "本次流式输出中断，正在重试")
    content = page.live_output_view.toPlainText()
    assert "正在准备流水线" in content
    assert "全文结构审查" in content
    assert "结构清晰" not in content
    assert "本次流式输出中断" in content
    page._update_elapsed_display()
    assert "本次运行 01:05" in page.elapsed_label.text()
    page._stop_run_clock()


def test_live_output_popout_shares_document_and_controls(qtbot) -> None:
    window = MainWindow(lambda: ChatResult(content="OK", model="test"))
    qtbot.addWidget(window)
    page = window.writing_task_page
    page.live_output_view.setPlainText("第一段输出")

    page._show_live_output_window()
    popout = page._live_output_window
    assert popout is not None
    assert popout.isVisible()
    assert popout.output_view.document() is page.live_output_view.document()
    assert popout.output_view.toPlainText() == "第一段输出"

    page.live_output_view.insertPlainText("\n第二段输出")
    assert "第二段输出" in popout.output_view.toPlainText()
    popout.auto_scroll_checkbox.setChecked(False)
    assert not page.auto_scroll_checkbox.isChecked()
    page.auto_scroll_checkbox.setChecked(True)
    assert popout.auto_scroll_checkbox.isChecked()
    page._copy_live_output()
    assert QApplication.clipboard().text() == page.live_output_view.toPlainText()
    popout.close()


def test_inline_live_output_respects_disabled_auto_scroll(qtbot) -> None:
    window = MainWindow(lambda: ChatResult(content="OK", model="test"))
    qtbot.addWidget(window)
    window.navigation.setCurrentRow(4)
    page = window.writing_task_page
    page.workspace_tabs.setCurrentIndex(page._progress_workspace_index)
    page.progress_tabs.setCurrentIndex(page._live_output_tab_index)
    window.show()
    page._pipeline_streamed("content::测试流", "\n".join(f"第 {i} 行" for i in range(120)))
    qtbot.waitUntil(
        lambda: page.live_output_view.verticalScrollBar().maximum() > 0,
        timeout=2000,
    )
    page.auto_scroll_checkbox.setChecked(False)
    scrollbar = page.live_output_view.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum() // 3)
    preserved = scrollbar.value()

    page._pipeline_streamed("content::测试流", "\n关闭自动滚动后的新内容")
    QApplication.processEvents()

    assert scrollbar.value() == preserved

    page._pipeline_streamed("content::重试流", "即将被清除的失败片段")
    scrollbar.setValue(scrollbar.maximum() // 3)
    preserved = scrollbar.value()
    page._pipeline_streamed("status::重试流", "本次流式输出中断，正在重试")
    QApplication.processEvents()
    assert scrollbar.value() == min(preserved, scrollbar.maximum())

    page.auto_scroll_checkbox.setChecked(True)
    page._pipeline_streamed("content::测试流", "\n重新跟随末尾")
    QApplication.processEvents()
    assert scrollbar.value() == scrollbar.maximum()


def test_writing_quality_presets_and_automatic_length_detection(qtbot) -> None:
    window = MainWindow(lambda: ChatResult(content="OK", model="test"))
    qtbot.addWidget(window)
    page = window.writing_task_page
    assert [checkbox.text() for checkbox in page._quality_checkboxes] == [
        "HyDE",
        "查询改写",
        "选题锐化",
        "内容规划",
        "事实核验",
        "单元打磨",
        "打磨防漂移",
        "术语审查",
        "结构审查",
        "全局打磨",
        "全局防漂移",
    ]
    for index, checkbox in enumerate(page._quality_checkboxes):
        layout_index = page.quality_grid.indexOf(checkbox)
        row, column, _row_span, _column_span = page.quality_grid.getItemPosition(layout_index)
        assert (row, column) == (index // 3, index % 3)
    assert all(checkbox.isChecked() for checkbox in page._quality_checkboxes)

    page.quality_preset_combo.setCurrentIndex(
        page.quality_preset_combo.findData("balanced")
    )
    assert all(checkbox.isChecked() for checkbox in page._quality_checkboxes[:5])
    assert all(not checkbox.isChecked() for checkbox in page._quality_checkboxes[5:])

    page.task_input.setPlainText("写一篇1500字左右的数字出版综述")
    page.target_length_spin.setValue(0)
    page.quality_preset_combo.setCurrentIndex(
        page.quality_preset_combo.findData("fast_draft")
    )

    options = page._generation_options()

    assert options.target_length_chars == 1500
    assert options.document_form == "short_text"
    assert not options.use_hyde
    assert not options.use_query_rewrite
    assert not options.topic_refinement
    assert not options.framework_generation
    assert not options.fact_verification
    assert not options.section_polish
    assert all(not checkbox.isEnabled() for checkbox in page._quality_checkboxes)

    page.quality_preset_combo.setCurrentIndex(page.quality_preset_combo.findData("custom"))
    page.fact_verification_checkbox.setChecked(True)
    page.section_polish_checkbox.setChecked(False)
    custom = page._generation_options()
    assert custom.preset == "custom"
    assert custom.fact_verification
    assert not custom.section_polish

    page._set_generation_options(
        {
            **custom.model_dump(mode="json"),
            "use_hyde": False,
            "use_query_rewrite": True,
            "topic_refinement": False,
            "framework_generation": True,
        }
    )
    restored = page._generation_options()
    assert not restored.use_hyde
    assert restored.use_query_rewrite
    assert not restored.topic_refinement
    assert restored.framework_generation

    page.document_form_combo.setCurrentIndex(page.document_form_combo.findData("auto"))
    page.target_length_spin.setValue(0)
    page.task_input.setPlainText("请写一个段落，说明数字阅读的公共价值")
    paragraph = page._generation_options()
    assert paragraph.document_form == "paragraph"
    assert paragraph.target_length_chars == 500

    page.task_input.setPlainText("生成这项研究的摘要")
    summary = page._generation_options()
    assert summary.document_form == "short_text"
    assert summary.target_length_chars == 1500

    page.document_form_combo.setCurrentIndex(page.document_form_combo.findData("paper"))
    paper = page._generation_options()
    assert paper.document_form == "paper"
    assert paper.target_length_chars == 5000

    page.genre_combo.setCurrentIndex(page.genre_combo.findData("auto"))
    page.citation_display_combo.setCurrentIndex(
        page.citation_display_combo.findData("internal_only")
    )
    page.task_input.setPlainText("面向社区居民写一篇演讲稿，介绍数字阅读服务")
    speech = page._generation_options()
    assert speech.genre == "speech"
    assert speech.citation_display == "internal_only"

    page.genre_combo.setCurrentIndex(page.genre_combo.findData("academic_paper"))
    academic = page._generation_options()
    assert academic.genre == "academic_paper"


def test_pipeline_state_updates_sections_partial_draft_and_diagnostics(qtbot) -> None:
    window = MainWindow(lambda: ChatResult(content="OK", model="test"))
    qtbot.addWidget(window)
    page = window.writing_task_page
    state = {
        "status": "drafting",
        "current_section_index": 1,
        "sections": [
            {
                "section_id": "1",
                "heading": "第一节",
                "status": "polished",
                "revision_count": 1,
                "elapsed_seconds": 125,
                "polished_text": "已经保存的第一节正文。",
            },
            {
                "section_id": "2",
                "heading": "第二节",
                "status": "revising",
                "revision_count": 2,
                "recovery_revision_count": 1,
                "elapsed_seconds": 30,
            },
        ],
    }

    page._pipeline_streamed("pipeline_state", json.dumps(state, ensure_ascii=False))
    page._pipeline_streamed("content::HyDE 假设文档", "检索辅助文本")

    assert page.section_table.rowCount() == 2
    assert page.section_table.item(0, 2).text() == "✓ 完成"
    assert page.section_table.item(0, 3).text() == "1"
    assert page.section_table.item(1, 3).text() == "2+恢复1/2"
    assert page.section_table.item(0, 4).text() == "02:05"
    assert "已经保存的第一节正文" in page._draft_view.toPlainText()
    assert "检索辅助文本" in page.diagnostic_output_view.toPlainText()
    assert "检索辅助文本" not in page.live_output_view.toPlainText()
    assert page._resume_progress(state) > 14


def test_retrieval_enhancement_switches_are_task_scoped(qtbot) -> None:
    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
    )
    qtbot.addWidget(window)

    assert not hasattr(window, "hyde_checkbox")
    assert not hasattr(window, "rewrite_checkbox")
    assert window.writing_task_page.hyde_checkbox.isChecked()
    assert window.writing_task_page.query_rewrite_checkbox.isChecked()


def test_quality_steps_help_and_closed_combos_ignore_wheel(qtbot) -> None:
    window = MainWindow(lambda: ChatResult(content="OK", model="test"))
    qtbot.addWidget(window)
    page = window.writing_task_page
    dialog = QualityStepsHelpDialog(page)
    qtbot.addWidget(dialog)

    assert dialog.table.rowCount() == len(QUALITY_STEP_HELP) == 11
    assert dialog.table.item(0, 0).text() == "HyDE"
    assert dialog.table.item(10, 0).text() == "全局防漂移"
    assert "预计耗时" == dialog.table.horizontalHeaderItem(2).text()
    assert page.quality_help_button.accessibleName() == "质量步骤说明"

    ignored = False

    class FakeWheelEvent:
        def ignore(self) -> None:
            nonlocal ignored
            ignored = True

    page.project_combo.wheelEvent(FakeWheelEvent())  # type: ignore[arg-type]
    assert ignored



def test_document_import_updates_table_without_blocking(qtbot, tmp_path: Path) -> None:
    documents: list[dict[str, object]] = []
    source = tmp_path / "资料.txt"
    source.write_text("测试", encoding="utf-8")

    def ingest(path: Path, context) -> IngestResult:
        assert path == source
        context.report_progress(50, "索引中")
        time.sleep(0.05)
        documents.append(
            {
                "doc_id": "doc",
                "filename": source.name,
                "status": "ready",
                "chunk_count": 1,
                "ingest_date": "2026-07-12T10:00:00+00:00",
            }
        )
        return IngestResult(
            job_id="job",
            kb_id="kb",
            doc_id="doc",
            child_chunk_count=1,
        )

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        ingest_document=ingest,
        list_documents=lambda: documents,
    )
    qtbot.addWidget(window)
    window.show()

    page = window.knowledge_page
    page.start_ingestion(source)
    assert not page.import_button.isEnabled()
    qtbot.waitUntil(lambda: page.document_table.rowCount() == 1, timeout=2000)

    assert page.document_table.item(0, 1).text() == "资料.txt"
    assert page.document_table.item(0, 2).text() == "可检索"
    assert window.persona_page.source_table.rowCount() == 1
    assert page.import_button.isEnabled()
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_batch_document_import_runs_sequentially(qtbot, tmp_path: Path) -> None:
    documents: list[dict[str, object]] = []
    sources = [tmp_path / "一.pdf", tmp_path / "二.pdf"]
    for source in sources:
        source.write_bytes(b"fixture")
    active = 0
    max_active = 0
    received: list[Path] = []

    def ingest(path: Path, context) -> IngestResult:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        received.append(path)
        context.report_progress(50, "解析中")
        time.sleep(0.03)
        documents.append(
            {
                "doc_id": f"doc_{len(documents)}",
                "filename": path.name,
                "status": "ready",
                "chunk_count": 2,
                "ingest_date": "2026-07-12T10:00:00+00:00",
            }
        )
        active -= 1
        return IngestResult(
            job_id=f"job_{len(documents)}",
            kb_id="kb",
            doc_id=str(documents[-1]["doc_id"]),
            child_chunk_count=2,
        )

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        ingest_document=ingest,
        list_documents=lambda: documents,
    )
    qtbot.addWidget(window)
    window.show()

    window.knowledge_page.start_ingestions(sources)
    qtbot.waitUntil(
        lambda: window.knowledge_page.document_table.rowCount() == 2,
        timeout=3000,
    )

    assert received == sources
    assert max_active == 1
    assert window.persona_page.source_table.rowCount() == 2
    assert "2 个文件" in window.statusBar().currentMessage()
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_hybrid_retrieval_runs_in_background_and_displays_filename(qtbot) -> None:
    received: list[tuple[str, bool, bool]] = []

    def retrieve(*, query, use_rewrite, use_hyde, context):
        received.append((query, use_rewrite, use_hyde))
        context.report_progress(50, "重排中")
        time.sleep(0.05)
        return RetrievalResult(
            query=query,
            hits=(
                FusedHit(
                    chunk_id="parent_one",
                    doc_id="doc_long_identifier",
                    text="这是可追溯的完整父级上下文。",
                    source="hybrid",
                    rrf_score=0.1,
                    rerank_score=0.9,
                    final_rank=1,
                    section_heading="引言",
                    page_start=2,
                    expanded_from_child=True,
                    matched_child_ids=("child_one",),
                ),
            ),
        )

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_documents=lambda: [
            {
                "doc_id": "doc_long_identifier",
                "filename": "叶芃论文.pdf",
                "status": "ready",
                "chunk_count": 1,
            }
        ],
        retrieve=retrieve,
    )
    qtbot.addWidget(window)
    window.show()
    page = window.knowledge_page
    page.query_input.setText("自主知识体系的实践性")

    qtbot.mouseClick(page.retrieve_button, Qt.MouseButton.LeftButton)
    assert not page.retrieve_button.isEnabled()
    qtbot.waitUntil(lambda: page.retrieval_table.rowCount() == 1, timeout=2000)

    assert received == [("自主知识体系的实践性", True, True)]
    assert page.retrieval_table.item(0, 2).text() == "叶芃论文.pdf"
    assert page.retrieval_table.item(0, 2).toolTip() == "doc_long_identifier"
    assert page.retrieval_table.item(0, 3).text() == "引言 · 2"
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_hybrid_retrieval_can_be_cancelled(qtbot) -> None:
    def retrieve(*, context, **_kwargs):
        for _index in range(100):
            time.sleep(0.01)
            context.check_cancelled()
        return RetrievalResult(query="不会完成")

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        retrieve=retrieve,
    )
    qtbot.addWidget(window)
    window.show()
    panel = window.knowledge_page.retrieval_panel
    panel.query_input.setText("待取消的问题")

    qtbot.mouseClick(panel.retrieve_button, Qt.MouseButton.LeftButton)
    assert panel.cancel_button.isVisible()
    qtbot.mouseClick(panel.cancel_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)

    assert panel.retrieve_button.isEnabled()
    assert not panel.cancel_button.isVisible()
    assert "任务已取消" in window.statusBar().currentMessage()


def test_entering_persona_page_refreshes_external_document_changes(qtbot) -> None:
    documents: list[dict[str, object]] = []
    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_documents=lambda: documents,
    )
    qtbot.addWidget(window)
    window.show()
    assert window.persona_page.source_table.rowCount() == 0
    documents.append(
        {
            "doc_id": "doc_new",
            "filename": "新论文.pdf",
            "status": "ready",
            "chunk_count": 3,
        }
    )

    window.navigation.setCurrentRow(2)

    assert window.persona_page.source_table.rowCount() == 1
    assert window.persona_page.source_table.item(0, 2).text() == "新论文.pdf"


def test_persona_page_distills_checked_sources_in_background(qtbot) -> None:
    profiles: list[dict[str, object]] = []
    received: list[tuple[str, str, set[str]]] = []

    def distill(name, mode, doc_ids, control_doc_ids, domain, context):
        received.append((name, mode, doc_ids))
        assert control_doc_ids == set()
        assert domain == ""
        context.report_progress(60, "归并")
        time.sleep(0.05)
        profiles.append(
            {
                "name": name,
                "mode": mode,
                "status": "ready",
                "model_count": 3,
                "research_date": "2026-07-12",
            }
        )
        return SimpleNamespace(persona=SimpleNamespace(mental_models=[1, 2, 3]))

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_documents=lambda: [
            {
                "doc_id": "doc_one",
                "filename": "论文.pdf",
                "status": "ready",
                "chunk_count": 4,
            }
        ],
        distill_persona=distill,
        list_personas=lambda: profiles,
    )
    qtbot.addWidget(window)
    window.show()
    window.navigation.setCurrentRow(2)
    page = window.persona_page
    page.name_input.setText("叶芃")

    assert page.distill_button.isEnabled()
    page.start_distillation()
    qtbot.waitUntil(lambda: page.profile_table.rowCount() == 1, timeout=2000)

    assert received == [("叶芃", "person", {"doc_one"})]
    assert page.profile_table.item(0, 1).text() == "叶芃"
    assert page.profile_table.item(0, 5).text() == "3"
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_topic_mode_ignores_checked_control_sources_and_domain(qtbot) -> None:
    received: list[tuple[set[str], set[str], str]] = []

    def distill(_name, _mode, doc_ids, control_doc_ids, domain, _context):
        received.append((doc_ids, control_doc_ids, domain))
        return SimpleNamespace(persona=SimpleNamespace(mental_models=[1, 2, 3]))

    documents = [
        {"doc_id": "doc_1", "filename": "一.pdf", "status": "ready", "chunk_count": 1},
        {"doc_id": "doc_2", "filename": "二.pdf", "status": "ready", "chunk_count": 1},
    ]
    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        distill_persona=distill,
        list_documents=lambda: documents,
    )
    qtbot.addWidget(window)
    page = window.persona_page
    page.name_input.setText("测试主题")
    page.domain_input.setText("不应传入")
    page.source_table.item(0, 1).setCheckState(Qt.CheckState.Checked)
    assert page.source_table.item(0, 0).checkState() == Qt.CheckState.Unchecked
    page.topic_button.setChecked(True)

    qtbot.mouseClick(page.distill_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: bool(received), timeout=2000)

    assert received == [({"doc_2"}, set(), "")]
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_persona_fidelity_check_runs_in_background_and_refreshes_score(qtbot) -> None:
    profiles = [
        {
            "persona_id": "persona_one",
            "name": "叶芃",
            "mode": "person",
            "status": "ready",
            "model_count": 3,
            "fidelity_score": None,
            "research_date": "2026-07-12",
        }
    ]
    received: list[str] = []

    def evaluate(persona_id, context):
        received.append(persona_id)
        context.report_progress(50, "中性评分")
        time.sleep(0.05)
        profiles[0]["fidelity_score"] = 88
        return SimpleNamespace(total=88)

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        evaluate_persona=evaluate,
        list_personas=lambda: profiles,
    )
    qtbot.addWidget(window)
    window.show()
    page = window.persona_page
    page.profile_table.selectRow(0)

    assert page.evaluate_button.isEnabled()
    qtbot.mouseClick(page.evaluate_button, Qt.MouseButton.LeftButton)
    assert not page.evaluate_button.isEnabled()
    qtbot.waitUntil(lambda: page.profile_table.item(0, 6).text() == "88/100", timeout=2000)

    assert received == ["persona_one"]
    assert page.evaluate_button.isEnabled()
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def _select_additional_row(table, row: int) -> None:
    selection = table.selectionModel()
    assert selection is not None
    selection.select(
        table.model().index(row, 1),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )


def test_knowledge_page_batch_deletes_selected_rows(qtbot) -> None:
    documents = [
        {"doc_id": "doc_a", "filename": "甲.pdf", "status": "ready", "chunk_count": 2},
        {"doc_id": "doc_b", "filename": "乙.pdf", "status": "ready", "chunk_count": 3},
    ]
    received: list[set[str]] = []

    def delete(doc_ids, context):
        received.append(set(doc_ids))
        context.report_progress(50, "删除中")
        documents[:] = [item for item in documents if item["doc_id"] not in doc_ids]
        return SimpleNamespace(removed_count=len(doc_ids), cleanup_failures=0)

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_documents=lambda: documents,
        delete_documents=delete,
    )
    qtbot.addWidget(window)
    window.show()
    page = window.knowledge_page
    page.document_table.selectRow(0)
    _select_additional_row(page.document_table, 1)

    assert page.delete_button.isEnabled()
    qtbot.mouseClick(page.delete_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page.document_table.rowCount() == 0, timeout=2000)

    assert received == [{"doc_a", "doc_b"}]
    assert window.persona_page.source_table.rowCount() == 0
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_persona_page_batch_deletes_selected_profiles(qtbot) -> None:
    profiles = [
        {"persona_id": "one", "name": "甲", "mode": "person", "status": "ready"},
        {"persona_id": "two", "name": "乙", "mode": "person", "status": "ready"},
    ]
    received: list[set[str]] = []

    def delete(persona_ids, context):
        received.append(set(persona_ids))
        profiles[:] = [item for item in profiles if item["persona_id"] not in persona_ids]
        return len(persona_ids)

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_personas=lambda: profiles,
        delete_personas=delete,
    )
    qtbot.addWidget(window)
    window.show()
    page = window.persona_page
    page.profile_table.selectRow(0)
    _select_additional_row(page.profile_table, 1)

    assert page.delete_button.isEnabled()
    qtbot.mouseClick(page.delete_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page.profile_table.rowCount() == 0, timeout=2000)

    assert received == [{"one", "two"}]
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_knowledge_checkboxes_take_precedence_over_highlighted_rows(qtbot) -> None:
    documents = [
        {"doc_id": "doc_a", "filename": "甲.pdf", "status": "ready"},
        {"doc_id": "doc_b", "filename": "乙.pdf", "status": "ready"},
        {"doc_id": "doc_c", "filename": "丙.pdf", "status": "ready"},
    ]
    received: list[set[str]] = []

    def delete(doc_ids, _context):
        received.append(set(doc_ids))
        documents[:] = [item for item in documents if item["doc_id"] not in doc_ids]
        return SimpleNamespace(removed_count=len(doc_ids), cleanup_failures=0)

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_documents=lambda: documents,
        delete_documents=delete,
    )
    qtbot.addWidget(window)
    window.show()
    page = window.knowledge_page
    page.document_table.selectRow(0)
    page.document_table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    page.document_table.item(2, 0).setCheckState(Qt.CheckState.Checked)

    qtbot.mouseClick(page.delete_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page.document_table.rowCount() == 1, timeout=2000)

    assert received == [{"doc_b", "doc_c"}]
    assert page.document_table.item(0, 1).text() == "甲.pdf"
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_persona_checkboxes_take_precedence_over_highlighted_rows(qtbot) -> None:
    profiles = [
        {"persona_id": "one", "name": "甲", "mode": "person", "status": "ready"},
        {"persona_id": "two", "name": "乙", "mode": "person", "status": "ready"},
        {"persona_id": "three", "name": "丙", "mode": "person", "status": "ready"},
    ]
    received: list[set[str]] = []

    def delete(persona_ids, _context):
        received.append(set(persona_ids))
        profiles[:] = [item for item in profiles if item["persona_id"] not in persona_ids]
        return len(persona_ids)

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_personas=lambda: profiles,
        delete_personas=delete,
    )
    qtbot.addWidget(window)
    window.show()
    page = window.persona_page
    page.profile_table.selectRow(0)
    page.profile_table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    page.profile_table.item(2, 0).setCheckState(Qt.CheckState.Checked)

    qtbot.mouseClick(page.delete_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page.profile_table.rowCount() == 1, timeout=2000)

    assert received == [{"two", "three"}]
    assert page.profile_table.item(0, 1).text() == "甲"
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_double_click_opens_persona_editor_and_saves_valid_json(qtbot) -> None:
    persona = _persona("persona_one")
    markdown = render_persona_markdown(persona)
    profiles = [
        {
            "persona_id": persona.id,
            "name": persona.name,
            "mode": persona.mode,
            "status": "ready",
            "model_count": len(persona.mental_models),
        }
    ]
    saved = []

    def save(persona_id, edited):
        assert persona_id == persona.id
        saved.append(edited)
        profiles[0]["name"] = edited.name
        return edited, render_persona_markdown(edited)

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_personas=lambda: profiles,
        load_persona=lambda _persona_id: (persona, markdown),
        save_persona=save,
    )
    qtbot.addWidget(window)
    window.show()
    window.navigation.setCurrentRow(2)
    page = window.persona_page
    page.profile_table.cellDoubleClicked.emit(0, 0)
    qtbot.waitUntil(lambda: persona.id in page._editor_windows, timeout=1000)
    editor = page._editor_windows[persona.id]
    qtbot.addWidget(editor)
    payload = json.loads(editor.json_editor.toPlainText())
    payload["name"] = "编辑后的叶芃"
    payload["mental_models"][0]["description"] = "这是经过人工校订的中文模型描述。"
    editor.json_editor.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

    qtbot.mouseClick(editor.save_button, Qt.MouseButton.LeftButton)

    assert saved[0].name == "编辑后的叶芃"
    assert saved[0].mental_models[0].description == "这是经过人工校订的中文模型描述。"
    assert "编辑后的叶芃" in editor.markdown_preview.toPlainText()
    assert page.profile_table.item(0, 1).text() == "编辑后的叶芃"
