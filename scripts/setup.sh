#!/usr/bin/env bash
# Create a project virtualenv and install SpotCheck in editable mode.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

if [[ ! -d .venv ]]; then
  echo "Creating .venv ..."
  "$PYTHON" -m venv .venv
fi

if [[ -f .venv/Scripts/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install --upgrade pip wheel
python -m pip install -e ".[fast,dev]"

if python -m pre_commit install >/dev/null 2>&1; then
  echo "Pre-commit hooks installed (ruff, same as CI)."
fi

echo ""
echo "SpotCheck is ready."
echo "  Activate:  source .venv/bin/activate  (or .venv/Scripts/activate on Windows)"
echo "  Run GUI:   spot-check"
