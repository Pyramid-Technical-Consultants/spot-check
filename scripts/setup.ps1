# Create a project virtualenv and install SpotCheck in editable mode.
# Required on Windows when using the Microsoft Store Python: its default user
# site-packages path is too long for PySide6 (MAX_PATH).
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv ..."
    & $Python -m venv .venv
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "venv python not found at $VenvPython"
}

& $VenvPython -m pip install --upgrade pip wheel
& $VenvPython -m pip install -e ".[fast,dev]"

Write-Host ""
Write-Host "SpotCheck is ready."
Write-Host "  Activate:  .\.venv\Scripts\Activate.ps1"
Write-Host "  Run GUI:   spot-check"
