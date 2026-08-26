param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload,
    [string]$Endpoint = "http://127.0.0.1:4317",
    [string]$ServiceName = "ai-trading-copilot",
    [string]$JaegerUiUrl = "http://127.0.0.1:16686"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$UiExe = Join-Path $Root ".venv\Scripts\ai-trading-copilot-ui.exe"

$env:OTEL_EXPORTER_OTLP_ENDPOINT = $Endpoint
$env:OTEL_SERVICE_NAME = $ServiceName
$env:JAEGER_UI_URL = $JaegerUiUrl

$UiArgs = @("--host", $HostName, "--port", [string]$Port)
if ($Reload) {
    $UiArgs += "--reload"
}

Write-Host "OTel export enabled:"
Write-Host "  OTEL_EXPORTER_OTLP_ENDPOINT=$env:OTEL_EXPORTER_OTLP_ENDPOINT"
Write-Host "  OTEL_SERVICE_NAME=$env:OTEL_SERVICE_NAME"
Write-Host "  JAEGER_UI_URL=$env:JAEGER_UI_URL"

if (Test-Path -LiteralPath $UiExe) {
    & $UiExe @UiArgs
    exit $LASTEXITCODE
}

if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv run ai-trading-copilot-ui @UiArgs
    exit $LASTEXITCODE
}

& python -m ai_trading_copilot.copilot.ui.server @UiArgs
exit $LASTEXITCODE
