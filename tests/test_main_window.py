"""Desktop shell interaction tests."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtCore import QItemSelectionModel, Qt
from PyQt6.QtWidgets import QApplication, QScrollArea

from tests.test_distill_pipeline import _persona
from writing_factory.distill.serialization import render_persona_markdown
from writing_factory.kb.models import FusedHit, IngestResult, RetrievalResult
from writing_factory.llm.models import ChatResult, TokenUsage
from writing_factory.ui.main_window import MainWindow


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
    window.navigation.setCurrentRow(3)
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


def test_writing_quality_presets_and_automatic_length_detection(qtbot) -> None:
    window = MainWindow(lambda: ChatResult(content="OK", model="test"))
    qtbot.addWidget(window)
    page = window.writing_task_page
    page.task_input.setPlainText("写一篇1500字左右的数字出版综述")
    page.target_length_spin.setValue(0)
    page.quality_preset_combo.setCurrentIndex(
        page.quality_preset_combo.findData("fast_draft")
    )

    options = page._generation_options()

    assert options.target_length_chars == 1500
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
                "elapsed_seconds": 30,
            },
        ],
    }

    page._pipeline_streamed("pipeline_state", json.dumps(state, ensure_ascii=False))
    page._pipeline_streamed("content::HyDE 假设文档", "检索辅助文本")

    assert page.section_table.rowCount() == 2
    assert page.section_table.item(0, 2).text() == "✓ 完成"
    assert page.section_table.item(0, 3).text() == "1"
    assert page.section_table.item(0, 4).text() == "02:05"
    assert "已经保存的第一节正文" in page._draft_view.toPlainText()
    assert "检索辅助文本" in page.diagnostic_output_view.toPlainText()
    assert "检索辅助文本" not in page.live_output_view.toPlainText()
    assert page._resume_progress(state) > 14


def test_settings_page_persists_retrieval_enhancement_switches(qtbot) -> None:
    stored = {"use_hyde": False, "use_rewrite": True}
    changed: list[tuple[str, bool]] = []
    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        get_retrieval_option=lambda key, default: stored.get(key, default),
        set_retrieval_option=lambda key, value: changed.append((key, value)),
    )
    qtbot.addWidget(window)

    assert not window.hyde_checkbox.isChecked()
    assert window.rewrite_checkbox.isChecked()
    window.hyde_checkbox.click()
    window.rewrite_checkbox.click()

    assert changed == [("use_hyde", True), ("use_rewrite", False)]


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
