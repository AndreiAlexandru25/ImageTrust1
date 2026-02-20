# Cyber Scout Windows EXE Build Script
#
# This script builds the Cyber Scout desktop application as a Windows executable.
#
# Prerequisites:
#   - Python 3.10+ with pip
#   - Virtual environment activated
#   - PyInstaller: pip install pyinstaller
#   - pywebview: pip install pywebview
#
# Usage:
#   .\scripts\build_exe.ps1
#   .\scripts\build_exe.ps1 -Clean      # Clean build directories first
#
# Output:
#   dist\CyberScout\CyberScout.exe

param(
    [switch]$Clean,
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Cyber Scout Windows EXE Builder" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Get project root
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "Project root: $root"

# Check for venv
if (!(Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate venv
. .\venv\Scripts\Activate.ps1

# Install required packages
Write-Host ""
Write-Host "Installing build dependencies..." -ForegroundColor Yellow
pip install -q pyinstaller pywebview

# Clean if requested
if ($Clean) {
    Write-Host ""
    Write-Host "Cleaning build directories..." -ForegroundColor Yellow

    @("build", "dist") | ForEach-Object {
        $path = Join-Path $root $_
        if (Test-Path $path) {
            Remove-Item -Recurse -Force $path
            Write-Host "  Removed: $_"
        }
    }

    # Remove old .spec cache
    Get-ChildItem -Path $root -Filter "*.spec.bak" -ErrorAction SilentlyContinue | Remove-Item -Force
}

# Stop any running instances
Write-Host ""
Write-Host "Stopping any running instances..." -ForegroundColor Yellow
Get-Process -Name "CyberScout" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name "ImageTrust" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Build
Write-Host ""
Write-Host "Building Cyber Scout..." -ForegroundColor Yellow
Write-Host ""

$specFile = Join-Path $root "CyberScout.spec"

if ($OneFile) {
    # One-file build (slower startup, single exe)
    Write-Host "Building ONE-FILE executable..." -ForegroundColor Cyan
    pyinstaller --noconfirm --clean `
        --noconsole `
        --onefile `
        --name CyberScout `
        --add-data "assets;assets" `
        --add-data "src\imagetrust\frontend;imagetrust\frontend" `
        --hidden-import streamlit `
        --hidden-import webview `
        --hidden-import imagetrust.forensics `
        --hidden-import imagetrust.frontend.cyber_app `
        src\imagetrust\frontend\desktop_launcher.py

    $exePath = Join-Path $root "dist\CyberScout.exe"
} else {
    # One-folder build (faster startup, folder with exe)
    if (Test-Path $specFile) {
        Write-Host "Using spec file: $specFile" -ForegroundColor Cyan
        pyinstaller --noconfirm $specFile
    } else {
        Write-Host "Building ONE-FOLDER executable..." -ForegroundColor Cyan
        pyinstaller --noconfirm --clean `
            --noconsole `
            --name CyberScout `
            --add-data "assets;assets" `
            --add-data "src\imagetrust\frontend;imagetrust\frontend" `
            --hidden-import streamlit `
            --hidden-import webview `
            --hidden-import imagetrust.forensics `
            --hidden-import imagetrust.frontend.cyber_app `
            src\imagetrust\frontend\desktop_launcher.py
    }

    $exePath = Join-Path $root "dist\CyberScout\CyberScout.exe"
}

# Verify output
Write-Host ""
if (Test-Path $exePath) {
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output: $exePath" -ForegroundColor White

    $exeSize = [math]::Round((Get-Item $exePath).Length / 1MB, 2)
    Write-Host "Size: $exeSize MB" -ForegroundColor White
    Write-Host ""
    Write-Host "To run:" -ForegroundColor Cyan
    Write-Host "  $exePath"
    Write-Host ""
    Write-Host "NOTE: Target machine requires WebView2 Runtime" -ForegroundColor Yellow
    Write-Host "Download: https://developer.microsoft.com/microsoft-edge/webview2/"
} else {
    Write-Host "============================================" -ForegroundColor Red
    Write-Host "  BUILD FAILED!" -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Expected output: $exePath" -ForegroundColor Red
    Write-Host "Check build logs above for errors."
    exit 1
}
