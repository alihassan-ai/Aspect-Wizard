@echo off
echo ================================
echo Installing Python and Git...
echo ================================

REM --- Install Python 3.11
echo Installing Python...
winget install -e --id Python.Python.3.11 -h
echo.

REM --- Install Git
echo Installing Git...
winget install -e --id Git.Git -h
echo.

REM --- Check Python
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH.
    echo Alternative solution:
    echo 1. Download manually from https://www.python.org/downloads/
    echo 2. During setup, tick "Add Python to PATH".
) else (
    python --version
    echo [OK] Python is installed.
)
echo.

REM --- Check Git
echo Checking Git installation...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git not found in PATH.
    echo Alternative solution:
    echo 1. Download manually from https://git-scm.com/download/win
    echo 2. During setup, tick "Add Git to PATH".
) else (
    git --version
    echo [OK] Git is installed.
)
echo.

echo ================================
echo Installation process finished!
echo ================================
pause
