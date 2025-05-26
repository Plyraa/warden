# PowerShell script to start the Warden API server with GUI
# This activates the virtual environment and starts the server

# Activate the virtual environment
$envPath = Join-Path $PSScriptRoot "env\Scripts\Activate.ps1"

if (Test-Path $envPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & $envPath
    
    if ($?) {
        # Start the server with both API and GUI enabled
        Write-Host "Starting Warden API server with GUI..." -ForegroundColor Cyan
        python warden.py --gui
    }
    else {
        Write-Host "Failed to activate virtual environment at: $envPath" -ForegroundColor Red
        Exit 1
    }
}
else {
    Write-Host "Virtual environment not found at: $envPath" -ForegroundColor Red
    Write-Host "Please run the following command to create the virtual environment:" -ForegroundColor Yellow
    Write-Host "python -m venv env" -ForegroundColor Yellow
    Write-Host "Then install the requirements:" -ForegroundColor Yellow
    Write-Host ".\env\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    
    # Try running without virtual environment as fallback
    Write-Host "Attempting to start without virtual environment..." -ForegroundColor Yellow
    python warden.py --gui
}
