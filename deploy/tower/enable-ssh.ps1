<#
.SYNOPSIS
  Enable the OpenSSH server on a Windows host and authorize a public key, so the box can be
  driven non-interactively over SSH (e.g. across a Tailscale network).

.DESCRIPTION
  Run ONCE, at the machine, in an ELEVATED PowerShell (Administrator). This is the bootstrap
  that can't be done remotely — there's no inbound shell until it runs. Afterwards:
      ssh <user>@<host>

.PARAMETER PublicKey
  The SSH public key to authorize. Defaults to the project's deploy key.

.PARAMETER AdminUser
  Set if you will log in as an ADMIN account (default). Admin logins use the machine-wide
  administrators_authorized_keys with locked-down ACLs. For a NON-admin login, pass
  -NonAdminUser <name> instead to write that user's ~\.ssh\authorized_keys.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\enable-ssh.ps1
#>
[CmdletBinding()]
param(
  [string]$PublicKey = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICesVdfKGESibctJ+Au8HQ+6exX3BpLdPm192bBsCec9 BullishDesignLLC@gmail.com',
  [string]$NonAdminUser = ''
)
$ErrorActionPreference = 'Stop'

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  throw 'Run this in an elevated (Administrator) PowerShell.'
}

Write-Host '== Installing + starting OpenSSH Server ==' -ForegroundColor Cyan
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

Write-Host '== Firewall: allow inbound TCP 22 ==' -ForegroundColor Cyan
if (-not (Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
    -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
}

Write-Host '== Default shell -> PowerShell (friendlier than cmd for scripting) ==' -ForegroundColor Cyan
New-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell `
  -Value 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -PropertyType String -Force | Out-Null

Write-Host '== Authorizing public key ==' -ForegroundColor Cyan
if ($NonAdminUser) {
  $dir = "C:\Users\$NonAdminUser\.ssh"
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $akf = Join-Path $dir 'authorized_keys'
  Add-Content -Path $akf -Value $PublicKey
  Write-Host "Wrote key to $akf (user $NonAdminUser)"
}
else {
  $akf = "$env:ProgramData\ssh\administrators_authorized_keys"
  Add-Content -Path $akf -Value $PublicKey
  # admin key file MUST be readable only by Administrators + SYSTEM, or sshd ignores it
  icacls $akf /inheritance:r /grant 'Administrators:F' /grant 'SYSTEM:F' | Out-Null
  Write-Host "Wrote key to $akf (admin login)"
}

Restart-Service sshd
Write-Host ''
Write-Host 'Done. From the remote box:  ssh <user>@<this-host>' -ForegroundColor Green
Write-Host 'Tip: on a Tailscale network, <this-host> is the tailnet name (e.g. tower).'
