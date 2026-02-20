<#
.SYNOPSIS
    ImageTrust v2.0 - Phase 1 Pipeline Launcher for Windows (PowerShell)

.DESCRIPTION
    Launches the Phase 1 pipeline with process priority control for
    resource-friendly background execution.

.PARAMETER Priority
    Process priority level: Low, BelowNormal, Normal
    Default: BelowNormal

.PARAMETER InputDir
    Input directory containing training images
    Default: data\train

.PARAMETER OutputBase
    Base output directory for Phase 1 outputs
    Default: data\phase1

.PARAMETER NumWorkers
    Number of CPU workers for synthetic generation
    Default: 6

.PARAMETER BatchSize
    GPU batch size for embedding extraction
    Default: 256

.PARAMETER GpuMemory
    GPU memory fraction (0.5-0.95)
    Default: 0.85

.PARAMETER SkipSynthetic
    Skip synthetic data generation phase

.PARAMETER SkipEmbedding
    Skip embedding extraction phase

.EXAMPLE
    .\launch_phase1_windows.ps1
    Run with default settings

.EXAMPLE
    .\launch_phase1_windows.ps1 -Priority Low
    Run at low priority for minimal system impact

.EXAMPLE
    .\launch_phase1_windows.ps1 -Priority Low -NumWorkers 4 -BatchSize 128
    Run at low priority with reduced resource usage

.NOTES
    Hardware Target: RTX 5080 (16GB VRAM), AMD 7800X3D (8 cores)
#>

param(
    [ValidateSet("Low", "BelowNormal", "Normal")]
    [string]$Priority = "BelowNormal",

    [string]$InputDir = "",
    [string]$OutputBase = "",
    [int]$NumWorkers = 6,
    [int]$BatchSize = 256,
    [double]$GpuMemory = 0.85,
    [switch]$SkipSynthetic,
    [switch]$SkipEmbedding,
    [switch]$NoNiqe,
    [string]$Webhook = ""
)

# Set strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Get script paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$PipelineScript = Join-Path $ScriptDir "run_phase1_pipeline.py"

# Set default paths if not provided
if ([string]::IsNullOrEmpty($InputDir)) {
    $InputDir = Join-Path $ProjectRoot "data\train"
}
if ([string]::IsNullOrEmpty($OutputBase)) {
    $OutputBase = Join-Path $ProjectRoot "data\phase1"
}

# Banner
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  IMAGETRUST v2.0 - PHASE 1 PIPELINE (WINDOWS POWERSHELL)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Display configuration
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Priority:     $Priority"
Write-Host "  Input:        $InputDir"
Write-Host "  Output:       $OutputBase"
Write-Host "  CPU Workers:  $NumWorkers"
Write-Host "  GPU Batch:    $BatchSize"
Write-Host "  GPU Memory:   $($GpuMemory * 100)%"
Write-Host "  NIQE Scoring: $(-not $NoNiqe)"
Write-Host ""

# Validate input directory
if (-not (Test-Path $InputDir)) {
    Write-Host "ERROR: Input directory not found: $InputDir" -ForegroundColor Red
    exit 1
}

# Check Python availability
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please ensure Python is installed and in PATH." -ForegroundColor Red
    exit 1
}

# Check for CUDA availability
try {
    $cudaCheck = python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1
    Write-Host $cudaCheck -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not verify CUDA availability" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting pipeline at $Priority priority..." -ForegroundColor Green
Write-Host "Press Ctrl+C to interrupt. Progress will be saved and can be resumed." -ForegroundColor Gray
Write-Host ""

# Build arguments
$arguments = @(
    "`"$PipelineScript`"",
    "--input_dir", "`"$InputDir`"",
    "--output_base", "`"$OutputBase`"",
    "--priority", $Priority.ToLower(),
    "--num_workers", $NumWorkers,
    "--batch_size", $BatchSize,
    "--gpu_memory", $GpuMemory
)

if ($SkipSynthetic) {
    $arguments += "--skip_synthetic"
}
if ($SkipEmbedding) {
    $arguments += "--skip_embedding"
}
if ($NoNiqe) {
    $arguments += "--no_niqe"
}
if (-not [string]::IsNullOrEmpty($Webhook)) {
    $arguments += "--webhook"
    $arguments += "`"$Webhook`""
}

# Map priority to ProcessPriorityClass
$priorityClass = switch ($Priority) {
    "Low" { [System.Diagnostics.ProcessPriorityClass]::Idle }
    "BelowNormal" { [System.Diagnostics.ProcessPriorityClass]::BelowNormal }
    "Normal" { [System.Diagnostics.ProcessPriorityClass]::Normal }
}

# Start the process
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = "python"
$startInfo.Arguments = $arguments -join " "
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardOutput = $false
$startInfo.RedirectStandardError = $false
$startInfo.WorkingDirectory = $ProjectRoot

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $startInfo

try {
    # Start the process
    $null = $process.Start()

    # Set priority after start
    $process.PriorityClass = $priorityClass
    Write-Host "Process started with PID: $($process.Id) at $Priority priority" -ForegroundColor Cyan

    # Wait for completion
    $process.WaitForExit()

    $exitCode = $process.ExitCode

    if ($exitCode -eq 0) {
        Write-Host ""
        Write-Host "Pipeline completed successfully!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "Pipeline completed with errors (exit code: $exitCode)" -ForegroundColor Yellow
    }

    exit $exitCode

} catch {
    Write-Host "ERROR: Failed to start pipeline: $_" -ForegroundColor Red
    exit 1
} finally {
    if ($process -and -not $process.HasExited) {
        Write-Host "Terminating process..." -ForegroundColor Yellow
        $process.Kill()
    }
}
