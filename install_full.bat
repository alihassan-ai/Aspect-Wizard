@echo off
setlocal

echo ============================================
echo   Aspect-Wizard Full Installer
echo ============================================

:: Check for Git
git --version >nul 2>&1
if errorlevel 1 (
    echo Git not found. Please install Git and re-run.
    pause
    exit /b
)

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.10+ and re-run.
    pause
    exit /b
)

:: Clone repo if not already present
if not exist Aspect-Wizard (
    echo Cloning Aspect-Wizard repository...
    git clone https://github.com/alihassan-ai/Aspect-Wizard.git
) else (
    echo Repo already exists. Updating...
    cd Aspect-Wizard
    git pull
    cd ..
)

:: Go inside project
cd Aspect-Wizard

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
