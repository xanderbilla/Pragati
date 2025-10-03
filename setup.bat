@echo off
echo ========================================
echo   PRAGATI - Setup Script
echo ========================================
echo.

echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)
echo Python found!

echo.
echo [2/4] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo [3/4] Activating virtual environment and installing packages...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo [4/4] Checking environment file...
if not exist ".env" (
    echo Creating .env file template...
    echo # NVIDIA API Key for accessing NVIDIA AI endpoints > .env
    echo # You need to get this from https://build.nvidia.com/ >> .env
    echo NVIDIA_API_KEY=your_nvidia_api_key_here >> .env
    echo. >> .env
    echo # Add your actual API key here >> .env
    echo # NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx >> .env
    echo.
    echo IMPORTANT: Please edit the .env file and add your NVIDIA API key!
) else (
    echo .env file already exists.
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To run the application:
echo   1. Edit .env file and add your NVIDIA API key
echo   2. Run: .venv\Scripts\python.exe voice.py
echo.
echo Press any key to continue...
pause >nul