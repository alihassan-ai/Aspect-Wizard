@echo off
setlocal

echo ============================================
echo   Aspect-Wizard Full Installer
echo ============================================

:: Check for Git
git --version >nul 2>&1
if errorlevel 1 (
    echo Git not found. Please run install_python_git.bat first and re-run.
    pause
    exit /b
)

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please run install_python_git.bat first and re-run.
    pause
    exit /b
)

:: Create venv if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate venv
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo No requirements.txt found. Skipping package install.
)

echo.
echo Installation complete! Run launch.bat to start Aspect-Wizard.
pause
