"""Static contracts for Windows packaging and frozen runtime behavior."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyinstaller_build_is_windowed_and_uses_packaged_icon() -> None:
    spec = (ROOT / "packaging" / "parrofish.spec").read_text(encoding="utf-8")

    assert 'name="Parrofish"' in spec
    assert "console=False" in spec
    assert "parrofish.ico" in spec
    assert "THIRD_PARTY_NOTICES.md" in spec
    assert 'collect_data_files("citeproc")' in spec
    assert "china-national-standard-gb-t-7714-2015-numeric.csl" in spec
    assert 'collect_data_files("citeproc_styles")' not in spec


def test_installer_is_per_user_and_preserves_runtime_data() -> None:
    installer = (ROOT / "packaging" / "parrofish.iss").read_text(encoding="utf-8")

    assert "DefaultDirName={localappdata}\\Programs\\{#AppName}" in installer
    assert "PrivilegesRequired=lowest" in installer
    assert "UninstallDelete" not in installer
    assert "Parrofish-Setup-{#AppVersion}-x64" in installer
