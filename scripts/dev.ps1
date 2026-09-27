param(
    [switch]$Seed
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not (Test-Path -LiteralPath ".env")) { Copy-Item .env.example .env }

if ($Seed) {
    conda run -n agentforge agentforge bootstrap
    conda run -n agentforge agentforge sync
}

Write-Host "Starting agentforge gateway on http://localhost:8000"
conda run -n agentforge agentforge serve --reload