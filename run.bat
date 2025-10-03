@echo off
echo ========================================
echo   PRAGATI - Voice Assistant
echo ========================================
echo.

if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

if not exist ".env" (
    echo ERROR: .env file not found!
    echo Please create .env file with your NVIDIA API key.
    pause
    exit /b 1
)

echo Starting PRAGATI Voice Assistant...
echo.
echo Instructions:
echo - Select your language from the startup screen
echo - Click the microphone button to record
echo - Speak your query (include Aadhaar number for scheme status)
echo - Click microphone again to stop recording
echo - Listen to the AI response
echo.
echo Close this window to stop the application.
echo.

.venv\Scripts\python.exe voice.py

echo.
echo Application closed.
pause