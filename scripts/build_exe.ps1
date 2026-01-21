$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (!(Test-Path ".\\venv\\Scripts\\Activate.ps1")) {
    Write-Host "Virtual environment not found. Please create venv first." -ForegroundColor Yellow
    exit 1
}

. .\\venv\\Scripts\\Activate.ps1

pip install -q pyinstaller

$exePath = Join-Path $root "dist\\ImageTrust.exe"

# Stop running EXE if any
Get-Process -Name "ImageTrust" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Remove old EXE to avoid permission errors
if (Test-Path $exePath) {
    try { Remove-Item $exePath -Force } catch {}
}

$distPath = Join-Path $root "dist"
$buildPath = Join-Path $root "build"

pyinstaller --noconsole --onefile --name ImageTrust --distpath $distPath --workpath $buildPath src\\imagetrust\\desktop_app.py

if (Test-Path $exePath) {
    Write-Host "EXE generated in: $exePath" -ForegroundColor Green
} else {
    Write-Host "EXE not found. Check build logs in $buildPath." -ForegroundColor Red
    Write-Host "Try running: pyinstaller --clean --noconsole --onefile --name ImageTrust src\\imagetrust\\desktop_app.py"
}
