@echo off
REM ============================================================================
REM ImageTrust v2.0 - Phase 1 Pipeline Launcher for Windows
REM ============================================================================
REM
REM This script launches the Phase 1 pipeline with low priority to allow
REM concurrent PC usage (gaming, browsing, etc.) while processing runs.
REM
REM Priority Levels:
REM   - LOW: Minimal CPU usage, longest runtime (best for overnight runs)
REM   - BELOWNORMAL: Reduced CPU usage, moderate runtime (good for background)
REM   - NORMAL: Full CPU usage, fastest runtime (dedicated processing)
REM
REM Usage:
REM   launch_phase1_windows.bat                   - Run with default settings
REM   launch_phase1_windows.bat --priority low    - Run at low priority
REM   launch_phase1_windows.bat --help            - Show all options
REM
REM ============================================================================

setlocal EnableDelayedExpansion

REM Set default paths
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."
set "PYTHON_EXE=python"
set "PIPELINE_SCRIPT=%SCRIPT_DIR%run_phase1_pipeline.py"

REM Default configuration
set "INPUT_DIR=%PROJECT_ROOT%\data\train"
set "OUTPUT_BASE=%PROJECT_ROOT%\data\phase1"
set "PRIORITY=belownormal"
set "NUM_WORKERS=6"
set "BATCH_SIZE=256"
set "GPU_MEMORY=0.85"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :run_pipeline
if /i "%~1"=="--priority" (
    set "PRIORITY=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--input" (
    set "INPUT_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--output" (
    set "OUTPUT_BASE=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--workers" (
    set "NUM_WORKERS=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--batch_size" (
    set "BATCH_SIZE=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--gpu_memory" (
    set "GPU_MEMORY=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    goto :show_help
)
shift
goto :parse_args

:show_help
echo.
echo ImageTrust v2.0 - Phase 1 Pipeline Launcher
echo ============================================
echo.
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --priority LEVEL    Process priority (low, belownormal, normal)
echo                       Default: belownormal
echo   --input DIR         Input directory with training images
echo                       Default: data\train
echo   --output DIR        Output base directory
echo                       Default: data\phase1
echo   --workers N         Number of CPU workers (1-8)
echo                       Default: 6
echo   --batch_size N      GPU batch size
echo                       Default: 256
echo   --gpu_memory FRAC   GPU memory fraction (0.5-0.95)
echo                       Default: 0.85
echo   --help              Show this help message
echo.
echo Examples:
echo   %~nx0
echo   %~nx0 --priority low
echo   %~nx0 --priority low --workers 4 --batch_size 128
echo.
exit /b 0

:run_pipeline
echo.
echo ============================================================================
echo   IMAGETRUST v2.0 - PHASE 1 PIPELINE (WINDOWS)
echo ============================================================================
echo.
echo Configuration:
echo   Priority:     %PRIORITY%
echo   Input:        %INPUT_DIR%
echo   Output:       %OUTPUT_BASE%
echo   CPU Workers:  %NUM_WORKERS%
echo   GPU Batch:    %BATCH_SIZE%
echo   GPU Memory:   %GPU_MEMORY%
echo.

REM Check if input directory exists
if not exist "%INPUT_DIR%" (
    echo ERROR: Input directory not found: %INPUT_DIR%
    echo Please specify a valid input directory with --input
    exit /b 1
)

REM Check if Python is available
%PYTHON_EXE% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please ensure Python is installed and in PATH.
    exit /b 1
)

REM Map priority to Windows priority class
set "WIN_PRIORITY="
if /i "%PRIORITY%"=="low" set "WIN_PRIORITY=/LOW"
if /i "%PRIORITY%"=="belownormal" set "WIN_PRIORITY=/BELOWNORMAL"
if /i "%PRIORITY%"=="normal" set "WIN_PRIORITY="

echo Starting pipeline at %PRIORITY% priority...
echo.
echo Press Ctrl+C to interrupt. Progress will be saved and can be resumed.
echo.

REM Start the pipeline with specified priority
if defined WIN_PRIORITY (
    start "ImageTrust Phase 1" /WAIT %WIN_PRIORITY% %PYTHON_EXE% "%PIPELINE_SCRIPT%" ^
        --input_dir "%INPUT_DIR%" ^
        --output_base "%OUTPUT_BASE%" ^
        --priority %PRIORITY% ^
        --num_workers %NUM_WORKERS% ^
        --batch_size %BATCH_SIZE% ^
        --gpu_memory %GPU_MEMORY%
) else (
    %PYTHON_EXE% "%PIPELINE_SCRIPT%" ^
        --input_dir "%INPUT_DIR%" ^
        --output_base "%OUTPUT_BASE%" ^
        --priority %PRIORITY% ^
        --num_workers %NUM_WORKERS% ^
        --batch_size %BATCH_SIZE% ^
        --gpu_memory %GPU_MEMORY%
)

if errorlevel 1 (
    echo.
    echo Pipeline completed with errors. Check logs for details.
    exit /b 1
) else (
    echo.
    echo Pipeline completed successfully!
    exit /b 0
)

endlocal
