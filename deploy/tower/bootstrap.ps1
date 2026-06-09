<#
.SYNOPSIS
  Prepare a Windows host to run the structured-agents vLLM container with GPU.

.DESCRIPTION
  Idempotent. Does the parts that CAN be scripted, and clearly reports the parts that need a
  human (GUI installer / reboot). Safe to run repeatedly. Run in an ELEVATED PowerShell.

  Steps:
    1. Enable the WSL2 features (Microsoft-Windows-Subsystem-Linux + VirtualMachinePlatform).
    2. Ensure WSL default version 2.
    3. Check for an NVIDIA driver (nvidia-smi) and print the GPU + VRAM.
    4. Check for Docker (docker CLI / Docker Desktop). If missing, print the winget install line.
    5. Create the code directory (default E:\structured-agents-v2) and, if git is present and a
       repo URL is given, clone into it.
    6. Print a checklist of what remains (Docker Desktop WSL2 backend + NVIDIA Container Toolkit).

.PARAMETER CodeDir
  Where the repo lives on this host. Default E:\structured-agents-v2.

.PARAMETER RepoUrl
  Optional git URL to clone into CodeDir if it's empty.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1 -RepoUrl https://github.com/Bullish-Design/structured-agents-v2.git
#>
[CmdletBinding()]
param(
  [string]$CodeDir = 'E:\structured-agents-v2',
  [string]$RepoUrl = ''
)
$ErrorActionPreference = 'Continue'
$todo = New-Object System.Collections.Generic.List[string]
function Ok($m) { Write-Host "[OK]   $m" -ForegroundColor Green }
function Info($m) { Write-Host "[..]   $m" -ForegroundColor Cyan }
function Todo($m) { Write-Host "[TODO] $m" -ForegroundColor Yellow; $todo.Add($m) }

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  throw 'Run this in an elevated (Administrator) PowerShell.'
}

Write-Host "Windows: $((Get-CimInstance Win32_OperatingSystem).Caption) build $([Environment]::OSVersion.Version.Build)"

# 1. WSL2 features ---------------------------------------------------------------------
Info 'Checking WSL2 Windows features...'
$needReboot = $false
foreach ($feat in 'Microsoft-Windows-Subsystem-Linux', 'VirtualMachinePlatform') {
  $state = (Get-WindowsOptionalFeature -Online -FeatureName $feat -ErrorAction SilentlyContinue).State
  if ($state -eq 'Enabled') { Ok "$feat already enabled" }
  else {
    Info "Enabling $feat..."
    $r = Enable-WindowsOptionalFeature -Online -FeatureName $feat -NoRestart -ErrorAction SilentlyContinue
    if ($r.RestartNeeded) { $needReboot = $true }
    Ok "$feat enabled"
  }
}
if ($needReboot) { Todo 'REBOOT required to finish enabling WSL2, then re-run this script.' }

# 2. WSL default version 2 -------------------------------------------------------------
if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
  Info 'Setting WSL default version 2 + updating kernel...'
  wsl --set-default-version 2 2>$null | Out-Null
  wsl --update 2>$null | Out-Null
  Ok 'WSL present (default version 2)'
}
else { Todo 'wsl.exe not found yet (expected before first reboot/update).' }

# 3. NVIDIA driver / GPU ---------------------------------------------------------------
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  $gpu = (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null)
  Ok "NVIDIA GPU: $gpu"
}
else {
  Todo 'nvidia-smi not found — install the NVIDIA Windows GPU driver (game-ready/studio). WSL2 CUDA rides the Windows driver; do NOT install a driver inside WSL.'
}

# 4. Docker ----------------------------------------------------------------------------
if (Get-Command docker -ErrorAction SilentlyContinue) {
  $v = (docker --version 2>$null)
  Ok "Docker present: $v"
}
else {
  Todo 'Docker not found. Install Docker Desktop:  winget install -e --id Docker.DockerDesktop'
  Todo 'Then in Docker Desktop: Settings > General > "Use the WSL 2 based engine" (default), and Settings > Resources > WSL integration ON.'
}

# 5. Code directory --------------------------------------------------------------------
if (-not (Test-Path $CodeDir)) {
  New-Item -ItemType Directory -Force -Path $CodeDir | Out-Null
  Ok "Created $CodeDir"
}
else { Ok "$CodeDir exists" }

$hasContent = (Get-ChildItem -Path $CodeDir -Force -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0
if ($RepoUrl -and -not $hasContent) {
  if (Get-Command git -ErrorAction SilentlyContinue) {
    Info "Cloning $RepoUrl into $CodeDir..."
    git clone $RepoUrl $CodeDir
    Ok 'Repo cloned'
  }
  else { Todo "git not found — install (winget install -e --id Git.Git) then: git clone $RepoUrl `"$CodeDir`"" }
}
elseif (-not $RepoUrl) {
  Info "No -RepoUrl given; put the repo at $CodeDir yourself (clone, or copy from the dev box)."
}

# 6. Summary ---------------------------------------------------------------------------
Write-Host ''
Write-Host '== Remaining manual steps ==' -ForegroundColor Magenta
if ($todo.Count -eq 0) {
  Write-Host 'None — prerequisites look ready. Next: NVIDIA Container Toolkit check + deploy.' -ForegroundColor Green
}
else { $todo | ForEach-Object { Write-Host " - $_" -ForegroundColor Yellow } }

Write-Host ''
Write-Host 'After Docker Desktop + GPU are up, verify GPU reaches containers (from PowerShell or WSL):' -ForegroundColor Cyan
Write-Host '  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi'
Write-Host 'If that lists the GPU, deploy:' -ForegroundColor Cyan
Write-Host "  cd `"$CodeDir\deploy\vllm`";  copy .env.example .env;  docker compose up --build -d"
Write-Host 'Then verify the endpoint (from the dev box, over Tailscale):' -ForegroundColor Cyan
Write-Host '  LLM_BASE_URL=http://tower:8000/v1 LLM_API_KEY=<key> deploy/vllm/verify.sh'
