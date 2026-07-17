"""Static contracts for Windows packaging and frozen runtime behavior."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyinstaller_build_is_windowed_and_uses_packaged_icon() -> None:
    spec = (ROOT / "packaging" / "parrofish.spec").read_text(encoding="utf-8")
    branding = (ROOT / "src/writing_factory/ui/branding.py").read_text(encoding="utf-8")

    assert 'name="Parrofish"' in spec
    assert "console=False" in spec
    assert "parrofish.ico" in spec
    assert "parrofish-64.png" in spec
    assert "parrofish-256.png" in spec
    assert '"writing_factory.assets.icons"' in spec
    assert "collect_data_files(\"writing_factory\"" not in spec
    assert "from writing_factory.assets import icons as icon_assets" in branding
    assert "files(icon_assets)" in branding
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


def test_windows_file_version_matches_release_version() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    major, minor, patch = (int(item) for item in version.split("."))
    version_info = (ROOT / "packaging" / "windows_version_info.txt").read_text(
        encoding="utf-8"
    )

    assert f"filevers=({major}, {minor}, {patch}, 0)" in version_info
    assert f"prodvers=({major}, {minor}, {patch}, 0)" in version_info
    assert f"FileVersion', '{version}'" in version_info
    assert f"ProductVersion', '{version}'" in version_info


def test_build_script_requires_the_real_frozen_main_window() -> None:
    script = (ROOT / "packaging" / "build_windows.ps1").read_text(encoding="utf-8")
    main = (ROOT / "src/writing_factory/main.py").read_text(encoding="utf-8")

    assert "Test-FrozenApplication" in script
    assert 'PARROFISH_FROZEN_SMOKE_TEST = "1"' in script
    assert "smoke test did not exit within 30 seconds" in script
    assert 'os.environ.get("PARROFISH_FROZEN_SMOKE_TEST") == "1"' in main
    assert "QTimer.singleShot(100, application.quit)" in main
    assert main.index("window.show()") < main.index("QTimer.singleShot")
