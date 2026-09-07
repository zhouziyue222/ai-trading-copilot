param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload,
    [switch]$Internal
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$UiPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $UiPython)) {
    throw "Project virtual environment not found: $UiPython"
}
. (Join-Path $PSScriptRoot "enable-utf8.ps1")
$env:COPILOT_INTERNAL_UI = if ($Internal) { "1" } else { "0" }
$UiArguments = @("-X", "utf8", "-m", "ai_trading_copilot.copilot.ui.server", "--host", $HostName, "--port", [string]$Port)
if ($Reload) { $UiArguments += "--reload" }
& $UiPython @UiArguments
exit $LASTEXITCODE
