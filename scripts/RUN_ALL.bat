@echo off
REM ================================================================
REM  ImageTrust - Run All Remaining Steps
REM  Phase 3 (figures + tables) + Build .exe
REM ================================================================
REM
REM  Prerequisite: Phase 2 completed (models/phase2/ populated)
REM  Cross-generator evaluation completed (models/phase2/cross_generator/)
REM
REM  Usage: Double-click this file or run from cmd
REM

cd /d D:\disertatie\imagetrust
set PYTHON=D:\disertatie\imagetrust\.venv\Scripts\python.exe

echo.
echo ============================================================
echo  STEP 1/2: Phase 3 - Publication Pipeline (figures + tables)
echo ============================================================
echo  Estimated: 5-15 minutes
echo.

%PYTHON% scripts/orchestrator/run_phase3_publication.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Phase 3 failed! Check output above.
    echo Press any key to continue to .exe build anyway...
    pause
)

echo.
echo ============================================================
echo  STEP 2/2: Build ImageTrust.exe
echo ============================================================
echo  Estimated: 3-10 minutes
echo.

%PYTHON% scripts/build_desktop.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed! Check output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  ALL DONE!
echo ============================================================
echo.
echo  Phase 3 output:  outputs\phase3\
echo  Figures:         outputs\phase3\figures\
echo  LaTeX tables:    outputs\phase3\tables\
echo  Filled paper:    outputs\phase3\main_filled.tex
echo.
echo  Desktop app:     dist\ImageTrust\ImageTrust.exe
echo.
pause
