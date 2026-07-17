# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-directory, windowed build for Parrofish."""

from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

PROJECT_ROOT = Path(SPECPATH).resolve().parent
PACKAGING_ROOT = PROJECT_ROOT / "packaging"
ICON_ASSET_ROOT = PROJECT_ROOT / "src" / "writing_factory" / "assets" / "icons"

datas = [
    (str(PROJECT_ROOT / "LICENSE"), "."),
    (str(PROJECT_ROOT / "README.md"), "."),
    (str(PACKAGING_ROOT / "THIRD_PARTY_NOTICES.md"), "."),
]
datas += [
    (str(ICON_ASSET_ROOT / filename), "writing_factory/assets/icons")
    for filename in ("parrofish-64.png", "parrofish-256.png", "parrofish.ico")
]
datas += collect_data_files("citeproc")
datas += collect_data_files("jieba")
datas += copy_metadata("keyring")

citeproc_styles_spec = find_spec("citeproc_styles")
if citeproc_styles_spec is None or citeproc_styles_spec.origin is None:
    raise RuntimeError("citeproc_styles package is unavailable")
citeproc_styles_root = Path(citeproc_styles_spec.origin).parent
datas.append(
    (
        str(
            citeproc_styles_root
            / "styles"
            / "china-national-standard-gb-t-7714-2015-numeric.csl"
        ),
        "citeproc_styles/styles",
    )
)

license_distributions = (
    "citeproc-py",
    "citeproc-py-styles",
    "keyring",
    "lancedb",
    "langgraph",
    "langgraph-checkpoint-sqlite",
    "pydantic",
    "PyQt6",
    "pyarrow",
    "rank-bm25",
    "tenacity",
)
for distribution_name in license_distributions:
    try:
        package_distribution = distribution(distribution_name)
    except PackageNotFoundError:
        continue
    for relative_file in package_distribution.files or ():
        filename = relative_file.name.casefold()
        if not filename.startswith(("license", "copying", "notice", "authors")):
            continue
        source = package_distribution.locate_file(relative_file)
        if source.is_file():
            datas.append((str(source), f"licenses/{distribution_name}"))

hiddenimports = collect_submodules("keyring.backends") + [
    "writing_factory.assets",
    "writing_factory.assets.icons",
]

analysis = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=(
        "IPython",
        "matplotlib",
        "notebook",
        "pytest",
        "tkinter",
    ),
    noarchive=False,
    optimize=1,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="Parrofish",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "src/writing_factory/assets/icons/parrofish.ico"),
    version=str(PACKAGING_ROOT / "windows_version_info.txt"),
    uac_admin=False,
)

collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Parrofish",
)
