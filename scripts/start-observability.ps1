param(
    [switch]$FollowLogs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $Root "observability\docker-compose.otel.yml"

docker compose -f $ComposeFile up -d

Write-Host "OpenTelemetry Collector is listening on:"
Write-Host "  OTLP gRPC: http://127.0.0.1:4317"
Write-Host "  OTLP HTTP: http://127.0.0.1:4318"
Write-Host "Jaeger UI:"
Write-Host "  http://127.0.0.1:16686"

if ($FollowLogs) {
    docker compose -f $ComposeFile logs -f
}
