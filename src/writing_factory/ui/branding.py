"""Central application identity and packaged icon loading."""

from __future__ import annotations

from importlib.resources import files

from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QApplication

from writing_factory.assets import icons as icon_assets

APP_NAME = "Parrofish"
ORGANIZATION_NAME = "Parrofish"
APP_WINDOW_TITLE = (
    "Parrofish by 叶芃  yp.work@foxmail.com  本项目采用 GPLv3+ 协议开源"
)


def configure_application_identity(application: QApplication) -> None:
    """Apply brand metadata without duplicating the app name in Windows titles."""

    application.setApplicationName(APP_NAME)
    application.setApplicationDisplayName("")
    application.setOrganizationName(ORGANIZATION_NAME)
    application.setWindowIcon(application_icon())


def application_icon() -> QIcon:
    """Load raster icon sizes eagerly so wheel resources remain self-contained."""

    icon = QIcon()
    icon_root = files(icon_assets)
    for filename in ("parrofish-64.png", "parrofish-256.png"):
        pixmap = QPixmap()
        pixmap.loadFromData(icon_root.joinpath(filename).read_bytes(), "PNG")
        if not pixmap.isNull():
            icon.addPixmap(pixmap)
    return icon
