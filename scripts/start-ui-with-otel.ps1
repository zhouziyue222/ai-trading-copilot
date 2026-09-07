param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload,
    [string]$Endpoint = "http://127.0.0.1:4317",
    [string]$ServiceName = "ai-trading-copilot",
    [string]$JaegerUiUrl = "http://127.0.0.1:16686"
)

$ErrorActionPreference = "Stop"

$env:OTEL_EXPORTER_OTLP_ENDPOINT = $Endpoint
$env:OTEL_SERVICE_NAME = $ServiceName
$env:JAEGER_UI_URL = $JaegerUiUrl

Write-Host "OTel export enabled:"
Write-Host "  OTEL_EXPORTER_OTLP_ENDPOINT=$env:OTEL_EXPORTER_OTLP_ENDPOINT"
Write-Host "  OTEL_SERVICE_NAME=$env:OTEL_SERVICE_NAME"
Write-Host "  JAEGER_UI_URL=$env:JAEGER_UI_URL"

& (Join-Path $PSScriptRoot "start-ui.ps1") -HostName $HostName -Port $Port -Reload:$Reload
exit $LASTEXITCODE
