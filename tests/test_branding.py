"""Brand identity, packaged icon, and license metadata tests."""

from __future__ import annotations

import tomllib
from importlib.resources import files
from pathlib import Path
from struct import unpack_from

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QImage

from writing_factory.llm.models import ChatResult
from writing_factory.ui.branding import (
    APP_NAME,
    APP_WINDOW_TITLE,
    ORGANIZATION_NAME,
    configure_application_identity,
)
from writing_factory.ui.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_brand_icons_have_expected_formats_and_sizes() -> None:
    icon_root = files("writing_factory.assets.icons")
    for size in (64, 256):
        payload = icon_root.joinpath(f"parrofish-{size}.png").read_bytes()
        image = QImage.fromData(payload, "PNG")
        assert not image.isNull()
        assert image.size() == QSize(size, size)

    ico_payload = icon_root.joinpath("parrofish.ico").read_bytes()
    windows_icon = QImage.fromData(ico_payload, "ICO")
    assert not windows_icon.isNull()
    reserved, icon_type, image_count = unpack_from("<HHH", ico_payload)
    assert (reserved, icon_type) == (0, 1)
    sizes = {
        (
            ico_payload[6 + index * 16] or 256,
            ico_payload[7 + index * 16] or 256,
        )
        for index in range(image_count)
    }
    assert {(64, 64), (256, 256)} <= sizes


def test_main_window_uses_parrofish_title_and_icon(qtbot) -> None:
    window = MainWindow(lambda: ChatResult(content="OK", model="test"))
    qtbot.addWidget(window)

    sizes = {(size.width(), size.height()) for size in window.windowIcon().availableSizes()}
    assert window.windowTitle() == APP_WINDOW_TITLE
    assert {(64, 64), (256, 256)} <= sizes


def test_application_identity_does_not_duplicate_windows_title(qapp) -> None:
    original_name = qapp.applicationName()
    original_display_name = qapp.applicationDisplayName()
    original_organization = qapp.organizationName()
    original_icon = qapp.windowIcon()
    try:
        configure_application_identity(qapp)
        assert qapp.applicationName() == APP_NAME
        assert qapp.applicationDisplayName() == ""
        assert qapp.organizationName() == ORGANIZATION_NAME
        assert not qapp.windowIcon().isNull()
    finally:
        qapp.setApplicationName(original_name)
        qapp.setApplicationDisplayName(original_display_name)
        qapp.setOrganizationName(original_organization)
        qapp.setWindowIcon(original_icon)


def test_project_declares_full_gplv3_or_later_license() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert metadata["project"]["license"] == "GPL-3.0-or-later"
    assert metadata["project"]["authors"] == [
        {"name": "叶芃", "email": "yp.work@foxmail.com"}
    ]
    assert "GNU GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3, 29 June 2007" in license_text
    assert "GPL-3.0-or-later" in readme
    assert "用户导入的文档、知识库、作者档案、项目数据" in readme
