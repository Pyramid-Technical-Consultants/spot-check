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
# Omit [fast]/SciPy from the frozen bundle (~100MB+); runtime falls back without cKDTree.
python -m pip install -e ".[build]"

VERSION="$(python -c "from spot_check._version import __version__; print(__version__)")"
echo "Building SpotCheck ${VERSION}"

rm -rf build dist
pyinstaller packaging/spot-check.spec --noconfirm --clean

OUT_DIR="$ROOT/dist/SpotCheck"
python "$ROOT/packaging/trim_windows_bundle.py" "$OUT_DIR"
python "$ROOT/packaging/trim_windows_bundle.py" "$OUT_DIR" --report | tail -n 12
EXE="$OUT_DIR/SpotCheck.exe"
if [[ ! -f "$EXE" ]]; then
  echo "Expected executable not found: $EXE" >&2
  exit 1
fi

INTERNAL="$OUT_DIR/_internal"
if [[ ! -d "$INTERNAL" ]]; then
  INTERNAL="$OUT_DIR"
fi
if [[ ! -d "$INTERNAL/pyvista" ]]; then
  echo "ERROR: PyVista was not bundled under $INTERNAL/pyvista (3D view will fail)." >&2
  exit 1
fi
if [[ ! -d "$INTERNAL/vtk.libs" ]]; then
  echo "ERROR: vtk.libs was not bundled (VTK import will fail in the frozen exe)." >&2
  exit 1
fi
if ! compgen -G "$INTERNAL/vtk.libs/"'vtkFiltersSources'*.dll > /dev/null; then
  echo "ERROR: vtkFiltersSources DLL missing from vtk.libs (PyVista import will fail)." >&2
  exit 1
fi

echo "Frozen VTK smoke test (SPOT_CHECK_SMOKE=1)..."
SMOKE_RC=0
if command -v timeout >/dev/null 2>&1; then
  timeout 120 env SPOT_CHECK_SMOKE=1 "$EXE" 2>&1 || SMOKE_RC=$?
else
  env SPOT_CHECK_SMOKE=1 "$EXE" 2>&1 || SMOKE_RC=$?
fi
if [[ "$SMOKE_RC" -ne 0 ]]; then
  echo "ERROR: Frozen executable failed VTK/PyVista smoke import (exit $SMOKE_RC)." >&2
  exit 1
fi
MPL_RC="$INTERNAL/matplotlib/mpl-data/matplotlibrc"
if [[ ! -f "$MPL_RC" ]]; then
  echo "ERROR: matplotlibrc missing from bundle ($MPL_RC)." >&2
  exit 1
fi
echo "Frozen VTK smoke test passed."

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

du_mb() {
  local path="$1"
  if command -v du >/dev/null 2>&1; then
    du -sm "$path" 2>/dev/null | awk '{print $1}'
  else
    powershell -NoProfile -Command "(Get-ChildItem -LiteralPath '$path' -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB" 2>/dev/null | awk '{printf "%.0f", $1}'
  fi
}
echo "Built: $EXE"
echo "Folder size: ~$(du_mb "$OUT_DIR") MB"
echo "Archive: $ARCHIVE ($(du_mb "$ARCHIVE" 2>/dev/null || echo "?") MB zip)"
