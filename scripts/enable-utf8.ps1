param([switch]$Persist)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $OutputEncoding
[Console]::OutputEncoding = $OutputEncoding
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Add-CopilotUtf8Profile {
    param([Parameter(Mandatory=$true)][string]$Path)
    $Marker = "# BEGIN ai_trading_copilot UTF-8"
    $Block = @'
# BEGIN ai_trading_copilot UTF-8
chcp 65001 > $null
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $OutputEncoding
[Console]::OutputEncoding = $OutputEncoding
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$PSDefaultParameterValues["Out-File:Encoding"] = "utf8"
# END ai_trading_copilot UTF-8
'@
    [byte[]]$Original = @()
    $Encoding = [System.Text.UTF8Encoding]::new($true)
    if (Test-Path -LiteralPath $Path) {
        $Original = [System.IO.File]::ReadAllBytes($Path)
        if ($Original.Length -ge 4 -and $Original[0] -eq 255 -and $Original[1] -eq 254 -and $Original[2] -eq 0 -and $Original[3] -eq 0) {
            $Encoding = [System.Text.UTF32Encoding]::new($false, $true)
        } elseif ($Original.Length -ge 4 -and $Original[0] -eq 0 -and $Original[1] -eq 0 -and $Original[2] -eq 254 -and $Original[3] -eq 255) {
            $Encoding = [System.Text.UTF32Encoding]::new($true, $true)
        } elseif ($Original.Length -ge 2 -and $Original[0] -eq 255 -and $Original[1] -eq 254) {
            $Encoding = [System.Text.Encoding]::Unicode
        } elseif ($Original.Length -ge 2 -and $Original[0] -eq 254 -and $Original[1] -eq 255) {
            $Encoding = [System.Text.Encoding]::BigEndianUnicode
        }
        if ($Encoding.GetString($Original).Contains($Marker)) {
            Write-Output "Already configured: $Path"
            return
        }
        $Backup = "$Path.copilot-backup-" + [guid]::NewGuid().ToString("N")
        [System.IO.File]::WriteAllBytes($Backup, $Original)
        Write-Output "Backup: $Backup"
    }
    [System.IO.Directory]::CreateDirectory((Split-Path -Parent $Path)) | Out-Null
    # The appended block is ASCII, so existing ANSI/UTF-8 bytes stay intact.
    # UTF-16/32 BOMs select matching encodings; never transcode old profile text.
    if ($Original.Length -eq 0) { $Original = $Encoding.GetPreamble() }
    [byte[]]$Updated = $Original + $Encoding.GetBytes("`r`n" + $Block + "`r`n")
    [System.IO.File]::WriteAllBytes($Path, $Updated)
    Write-Output "Configured: $Path"
}

if ($Persist) {
    $Documents = [Environment]::GetFolderPath("MyDocuments")
    $ProfilePaths = @(
        $PROFILE.CurrentUserAllHosts,
        (Join-Path $Documents "WindowsPowerShell\profile.ps1"),
        (Join-Path $Documents "PowerShell\profile.ps1")
    ) | Select-Object -Unique
    foreach ($ProfilePath in $ProfilePaths) { Add-CopilotUtf8Profile -Path $ProfilePath }
    [Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
    [Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")
    Write-Output "UTF-8 configured. Open a new PowerShell session to apply the profiles."
}
