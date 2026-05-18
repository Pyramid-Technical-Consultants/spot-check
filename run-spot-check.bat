@echo off
REM Launch SpotCheck GUI using the project venv (run scripts\setup.bat first).
cd /d "%~dp0"
if not exist ".venv\Scripts\spot-check.exe" (
    echo Run scripts\setup.bat first to create .venv and install dependencies.
    exit /b 1
)
".venv\Scripts\spot-check.exe" %*
