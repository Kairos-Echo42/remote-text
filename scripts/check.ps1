$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"
conda run --no-capture-output -n agentforge ruff check --no-cache src tests benchmarks
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$cache = Join-Path $env:TEMP "agentforge-mypy"
conda run --no-capture-output -n agentforge mypy --cache-dir $cache src/agentforge
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
conda run --no-capture-output -n agentforge pytest -q -p no:cacheprovider
