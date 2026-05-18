@echo off
REM Create .venv and install SpotCheck (no PowerShell execution policy required).
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv ...
    python -m venv .venv
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip wheel
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" -m pip install -e ".[fast,dev]"
if errorlevel 1 exit /b 1

echo.
echo SpotCheck is ready.
echo   Run GUI:  .venv\Scripts\spot-check.exe
echo   Or:      .venv\Scripts\activate.bat  then  spot-check
exit /b 0
