[CmdletBinding()]
param(
    [switch]$Tello,
    [switch]$Json,
    [switch]$SkipOllamaStart,
    [switch]$PullMissingModel,
    [string]$Config,
    [string]$OllamaModel = "qwen3:4b-instruct"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$mainScript = Join-Path $projectRoot "main.py"

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Virtual environment is missing. Run scripts\setup.ps1 first."
}

function Test-OllamaApi {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $pending = $client.BeginConnect("127.0.0.1", 11434, $null, $null)
        if (-not $pending.AsyncWaitHandle.WaitOne(250)) {
            return $false
        }
        $client.EndConnect($pending)
        return $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Find-OllamaExecutable {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    $installed = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path -LiteralPath $installed -PathType Leaf) {
        return $installed
    }
    return $null
}

$ollamaExe = Find-OllamaExecutable
if (-not (Test-OllamaApi) -and -not $SkipOllamaStart) {
    if ($null -eq $ollamaExe) {
        Write-Warning "Ollama is not installed or was not found. The doctor will report this failure."
    }
    else {
        Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
        $ollamaReady = $false
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            if (Test-OllamaApi) {
                $ollamaReady = $true
                break
            }
            Start-Sleep -Milliseconds 500
        }
        if (-not $ollamaReady) {
            Write-Warning "Ollama did not become ready within 15 seconds."
        }
    }
}

if ($PullMissingModel -and (Test-OllamaApi)) {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    $installedModels = @($tags.models | ForEach-Object { $_.name })
    if ($installedModels -notcontains $OllamaModel) {
        if ($null -eq $ollamaExe) {
            throw "Cannot pull $OllamaModel because the Ollama executable was not found."
        }
        & $ollamaExe pull $OllamaModel
        if ($LASTEXITCODE -ne 0) {
            throw "Ollama model download failed with exit code $LASTEXITCODE."
        }
    }
}

$doctorArgs = @($mainScript)
if ($Config) {
    $configPath = (Resolve-Path -LiteralPath $Config).Path
    $doctorArgs += @("--config", $configPath)
}
$doctorArgs += "doctor"
if ($Json) {
    $doctorArgs += "--json"
}
if ($Tello) {
    $doctorArgs += "--tello"
}

Push-Location $projectRoot
try {
    & $pythonExe @doctorArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Readiness checks failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
