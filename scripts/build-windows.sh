#!/usr/bin/env bash
# Build a Windows one-folder SpotCheck distribution with PyInstaller.
#
# Usage (Git Bash on Windows, or CI):
#   ./scripts/build-windows.sh
#   SPOT_CHECK_VERSION=1.2.3 ./scripts/build-windows.sh
#
# Output:
#   dist/SpotCheck/SpotCheck.exe
#   dist/SpotCheck-<version>-windows-x64.zip
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
VENV="${VENV:-$ROOT/.build-venv}"
VERSION_FILE="$ROOT/src/spot_check/_version.py"

if [[ -n "${SPOT_CHECK_VERSION:-}" ]]; then
  echo "Setting version to ${SPOT_CHECK_VERSION}"
  cat >"$VERSION_FILE" <<EOF
"""Single source of truth for the SpotCheck release version."""

__version__ = "${SPOT_CHECK_VERSION}"
EOF
fi

if [[ ! -d "$VENV" ]]; then
  "$PYTHON" -m venv "$VENV"
fi

if [[ -f "$VENV/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/Scripts/activate"
elif [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "Could not find venv activate script under $VENV" >&2
  exit 1
fi

python -m pip install --upgrade pip wheel
python -m pip install -e ".[fast,build]"

VERSION="$(python -c "from spot_check._version import __version__; print(__version__)")"
echo "Building SpotCheck ${VERSION}"

rm -rf build dist
pyinstaller packaging/spot-check.spec --noconfirm --clean

OUT_DIR="$ROOT/dist/SpotCheck"
EXE="$OUT_DIR/SpotCheck.exe"
if [[ ! -f "$EXE" ]]; then
  echo "Expected executable not found: $EXE" >&2
  exit 1
fi

ARCHIVE="$ROOT/dist/SpotCheck-${VERSION}-windows-x64.zip"
rm -f "$ARCHIVE"
(
  cd "$ROOT/dist"
  if command -v zip >/dev/null 2>&1; then
    zip -r "$(basename "$ARCHIVE")" SpotCheck
  else
    powershell -NoProfile -Command "Compress-Archive -Path 'SpotCheck' -DestinationPath '$(basename "$ARCHIVE")' -Force"
  fi
)

echo "Built: $EXE"
echo "Archive: $ARCHIVE"
