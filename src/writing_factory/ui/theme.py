"""Cross-platform application font selection for reliable Chinese text."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication


def configure_application_font(application: QApplication) -> str:
    """Select or register a Chinese-capable UI font and return its family."""

    preferred = (
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "DengXian",
        "SimHei",
    )
    available = {family.casefold(): family for family in QFontDatabase.families()}
    for family in preferred:
        match = available.get(family.casefold())
        if match is not None:
            application.setFont(QFont(match, 10))
            return match

    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = (
        windows_fonts / "NotoSansSC-VF.ttf",
        windows_fonts / "msyh.ttc",
        windows_fonts / "simhei.ttf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for path in candidates:
        if not path.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            continue
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            application.setFont(QFont(families[0], 10))
            return families[0]
    return application.font().family()
