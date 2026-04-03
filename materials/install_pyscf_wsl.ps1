# Run install_pyscf_wsl.sh inside WSL from Windows PowerShell.
# Usage (from anywhere):
#   & "D:\Yaozheng\QuantumChemistry\LearningPlan\learning_materials\install_pyscf_wsl.ps1" 2.6.2
# Or cd to learning_materials first:
#   .\install_pyscf_wsl.ps1
#   .\install_pyscf_wsl.ps1 2.6.2

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$sh = Join-Path $here "install_pyscf_wsl.sh"
if (-not (Test-Path -LiteralPath $sh)) {
    Write-Error "Missing: $sh"
}

# WSL path for the same folder (adjust drive letter if repo is not on D:)
$drive = $here.Substring(0, 1).ToLowerInvariant()
$rest = $here.Substring(2) -replace "\\", "/"
$wslDir = "/mnt/$drive$rest"

$argList = @("-e", "bash", "-lc", "cd `"$wslDir`" && chmod +x install_pyscf_wsl.sh && ./install_pyscf_wsl.sh $($args -join ' ')")
& wsl.exe @argList
exit $LASTEXITCODE
