[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$IsccPath = "",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$PackagingRoot = $PSScriptRoot
$ProjectRoot = (Resolve-Path (Join-Path $PackagingRoot "..")).Path
$TemporaryRoot = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "ParrofishBuild"
$BuildRoot = Join-Path $TemporaryRoot "dist"
$WorkRoot = Join-Path $TemporaryRoot "work"
$OutputRoot = if ($OutputDirectory) {
    [IO.Path]::GetFullPath($OutputDirectory)
} else {
    Join-Path $PackagingRoot "output"
}

if (-not $Python) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

if (-not $IsccPath) {
    $candidates = @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe"
    )
    $IsccPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $IsccPath -or -not (Test-Path -LiteralPath $IsccPath -PathType Leaf)) {
    throw "Inno Setup 6 compiler ISCC.exe was not found"
}

function Reset-BuildDirectory([string]$Path) {
    $fullPath = [IO.Path]::GetFullPath($Path)
    $separator = [IO.Path]::DirectorySeparatorChar
    $allowedRoots = @($PackagingRoot, $TemporaryRoot)
    $isAllowed = $false
    foreach ($allowedRoot in $allowedRoots) {
        $expectedPrefix = [IO.Path]::GetFullPath($allowedRoot).TrimEnd($separator) + $separator
        if ($fullPath.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            $isAllowed = $true
            break
        }
    }
    if (-not $isAllowed) {
        throw "Refusing to clean a path outside the approved build directories: $fullPath"
    }
    if (Test-Path -LiteralPath $fullPath) {
        Remove-Item -LiteralPath $fullPath -Recurse -Force
    }
    New-Item -ItemType Directory -Path $fullPath | Out-Null
}

Reset-BuildDirectory $BuildRoot
Reset-BuildDirectory $WorkRoot
if (-not (Test-Path -LiteralPath $OutputRoot)) {
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
}

$versionLine = Select-String -LiteralPath (Join-Path $ProjectRoot "pyproject.toml") `
    -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $versionLine) {
    throw "Unable to read the version from pyproject.toml"
}
$Version = $versionLine.Matches[0].Groups[1].Value

& $Python -c "import PyInstaller" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed in the selected Python environment"
}

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --distpath $BuildRoot `
    --workpath $WorkRoot `
    (Join-Path $PackagingRoot "parrofish.spec")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed"
}

& $IsccPath "/DAppVersion=$Version" "/DBuildRoot=$BuildRoot" "/O$OutputRoot" `
    (Join-Path $PackagingRoot "parrofish.iss")
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup build failed"
}

$portableName = "Parrofish-Portable-$Version-x64"
$portableStage = Join-Path $TemporaryRoot "portable-stage"
Reset-BuildDirectory $portableStage
$portableRoot = Join-Path $portableStage $portableName
Copy-Item -LiteralPath (Join-Path $BuildRoot "Parrofish") -Destination $portableRoot -Recurse
$portableZip = Join-Path $OutputRoot "$portableName.zip"
if (Test-Path -LiteralPath $portableZip) {
    Remove-Item -LiteralPath $portableZip -Force
}
Compress-Archive -LiteralPath $portableRoot -DestinationPath $portableZip -CompressionLevel Optimal
Remove-Item -LiteralPath $portableStage -Recurse -Force

$sourceName = "Parrofish-Source-$Version"
$sourceZip = Join-Path $OutputRoot "$sourceName.zip"
if (Test-Path -LiteralPath $sourceZip) {
    Remove-Item -LiteralPath $sourceZip -Force
}
& git -C $ProjectRoot archive `
    --format=zip `
    "--prefix=$sourceName/" `
    "--output=$sourceZip" `
    HEAD
if ($LASTEXITCODE -ne 0) {
    throw "Source archive build failed"
}

$installer = Join-Path $OutputRoot "Parrofish-Setup-$Version-x64.exe"
$checksumPath = Join-Path $OutputRoot "SHA256SUMS.txt"
$releaseFiles = @($installer, $portableZip, $sourceZip)
foreach ($releaseFile in $releaseFiles) {
    if (-not (Test-Path -LiteralPath $releaseFile -PathType Leaf)) {
        throw "Expected release file is missing: $releaseFile"
    }
}
$checksumLines = $releaseFiles | ForEach-Object {
    $hash = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $([IO.Path]::GetFileName($_))"
}
[IO.File]::WriteAllLines(
    $checksumPath,
    $checksumLines,
    [Text.UTF8Encoding]::new($false)
)

Get-ChildItem -LiteralPath $OutputRoot -File | Select-Object Name, Length, LastWriteTime
