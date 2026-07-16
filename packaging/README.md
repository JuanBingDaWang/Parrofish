# Windows packaging

Parrofish uses a PyInstaller `onedir` build with `console=False`, followed by Inno Setup 6. Ordinary users receive a GUI executable without a console window.

## Prerequisites

```powershell
uv sync --all-groups --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Install Inno Setup 6 and pass `ISCC.exe` explicitly if it is not in the default installation directory.
The project vendors Inno Setup's official simplified-Chinese messages file so the installer does not
depend on optional language files on the build machine.

## Build

```powershell
.\packaging\build_windows.ps1 `
  -IsccPath "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
```

Generated files are written to `packaging/output/`: the installer, portable ZIP, clean source
ZIP, and `SHA256SUMS.txt`. Temporary build and work files use the short
`%LOCALAPPDATA%\ParrofishBuild` path to avoid the legacy Windows path-length limit.

The installer uses a per-user location and does not require administrator privileges. Runtime databases, documents, logs and checkpoints are stored under `%LOCALAPPDATA%\Parrofish`; uninstalling the application intentionally preserves that user data.
