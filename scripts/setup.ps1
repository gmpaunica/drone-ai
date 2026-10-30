[CmdletBinding()]
param(
    [switch]$UseVersionRanges,
    [switch]$SkipModelDownloads,
    [string]$Config
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$venvDir = Join-Path $projectRoot ".venv"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE."
    }
}

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($null -eq $pyLauncher) {
            throw "Python launcher not found. Install 64-bit Python 3.11, then rerun setup."
        }
        Invoke-NativeCommand -FilePath $pyLauncher.Source -ArgumentList @("-3.11", "-m", "venv", $venvDir)
    }

    Invoke-NativeCommand -FilePath $pythonExe -ArgumentList @(
        "-c",
        "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
    )

    if ($UseVersionRanges) {
        Invoke-NativeCommand -FilePath $pythonExe -ArgumentList @(
            "-m", "pip", "install",
            "-r", (Join-Path $projectRoot "requirements-dev.txt"),
            "-r", (Join-Path $projectRoot "requirements-tello.txt")
        )
    }
    else {
        Invoke-NativeCommand -FilePath $pythonExe -ArgumentList @(
            "-m", "pip", "install",
            "-r", (Join-Path $projectRoot "requirements-lock.txt")
        )
    }

    Invoke-NativeCommand -FilePath $pythonExe -ArgumentList @("-m", "pip", "check")
    Invoke-NativeCommand -FilePath $pythonExe -ArgumentList @("-m", "pytest")

    if (-not $SkipModelDownloads) {
        Invoke-NativeCommand -FilePath $pythonExe -ArgumentList @(
            "-c",
            "from ultralytics import YOLO; YOLO('yolo26n.pt'); print('YOLO weights ready')"
        )
    }

    $readyScript = Join-Path $PSScriptRoot "ready.ps1"
    if ($Config) {
        & $readyScript -PullMissingModel -Config $Config
    }
    else {
        & $readyScript -PullMissingModel
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Readiness checks failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
