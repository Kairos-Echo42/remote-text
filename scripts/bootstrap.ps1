param(
    [switch]$SkipEnvironment,
    [switch]$SkipCompose
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Find-Docker {
    if ($env:AGENTFORGE_DOCKER_EXE -and (Test-Path -LiteralPath $env:AGENTFORGE_DOCKER_EXE)) {
        return $env:AGENTFORGE_DOCKER_EXE
    }
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe",
        "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    $command = Get-Command docker -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw "Docker CLI was not found. Set AGENTFORGE_DOCKER_EXE."
}

$Docker = Find-Docker
$env:AGENTFORGE_DOCKER_EXE = $Docker
Write-Host "Docker CLI: $Docker"

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "Created .env from .env.example"
}

$hostWorkspaceRoot = ((Join-Path $Root "data\workspaces") -replace "\\", "/")
$envText = [IO.File]::ReadAllText((Join-Path $Root ".env"))
if ($envText -match "(?m)^AGENTFORGE_HOST_WORKSPACE_ROOT=.*$") {
    $envText = [regex]::Replace(
        $envText,
        "(?m)^AGENTFORGE_HOST_WORKSPACE_ROOT=.*$",
        "AGENTFORGE_HOST_WORKSPACE_ROOT=$hostWorkspaceRoot"
    )
} else {
    $envText = $envText.TrimEnd() + "`nAGENTFORGE_HOST_WORKSPACE_ROOT=$hostWorkspaceRoot`n"
}
[IO.File]::WriteAllText((Join-Path $Root ".env"), $envText, [Text.UTF8Encoding]::new($false))

if (-not $SkipEnvironment) {
    $envExists = conda env list | Select-String -Pattern "^\s*agentforge\s"
    if (-not $envExists) {
        conda create -n agentforge python=3.12 pip -y
    }
    conda run -n agentforge python -m pip install --upgrade pip
    conda run -n agentforge python -m pip install -r requirements-dev.txt
    conda run -n agentforge python -m pip install -e . --no-deps
}

if (-not $SkipCompose) {
    & $Docker compose up -d --build
    Write-Host "Waiting for AgentForge Gateway..."
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $health = Invoke-RestMethod -Uri "http://localhost:8000/health/ready" -TimeoutSec 2
            if ($health.status -eq "ready") {
                Write-Host "AgentForge is ready: http://localhost:8000"
                exit 0
            }
        } catch { Start-Sleep -Seconds 2 }
    }
    throw "Gateway did not become ready. Run: & '$Docker' compose logs gateway"
}

Write-Host "Environment ready. Start services with: conda run -n agentforge agentforge serve"