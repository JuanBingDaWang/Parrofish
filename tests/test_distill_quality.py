"""Independent static Nüwa quality gate tests."""

from __future__ import annotations

from tests.test_distill_pipeline import _persona
from writing_factory.distill.quality import run_static_quality_check


def test_static_quality_passes_hard_contract_and_reports_thin_dimensions() -> None:
    report = run_static_quality_check(_persona("persona"))

    assert report.passed
    assert report.checks["triple_validation"]
    assert any("核心张力" in warning for warning in report.warnings)
    assert any("决策启发式" in warning for warning in report.warnings)
