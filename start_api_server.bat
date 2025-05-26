@echo off
echo Starting Warden FastAPI Server with Web UI...

:: Try to activate the virtual environment if it exists
if exist "env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call env\Scripts\activate.bat
)

:: Start the server with both API and GUI enabled
python warden.py --gui

:: Add a pause at the end to keep the window open if there's an error
pause
