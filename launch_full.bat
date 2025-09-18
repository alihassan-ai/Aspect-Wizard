@echo off
setlocal

echo ============================================
echo   Aspect-Wizard Launcher
echo ============================================


:: Activate venv
if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo Virtual environment not found. Please run install.bat first.
    pause
    exit /b
)

:: Run app
python app.py

pause
