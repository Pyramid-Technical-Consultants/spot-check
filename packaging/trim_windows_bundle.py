"""Remove Qt/VTK payloads not needed by SpotCheck after PyInstaller COLLECT.

Safe to run on dist/SpotCheck/ (one-folder layout). Deletes only known-unused names;
re-run the GUI smoke test after changing this list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Qt modules SpotCheck does not use (widgets + QVTK only).
_QT_DLL_PREFIXES = (
    "Qt6Quick",
    "Qt6Qml",
    "Qt6Designer",
    "Qt6WebEngine",
    "Qt6Pdf",
    "Qt6Multimedia",
    "Qt6Bluetooth",
    "Qt6Positioning",
    "Qt6Location",
    "Qt6Sensors",
    "Qt6SerialPort",
    "Qt6SerialBus",
    "Qt6Test",
    "Qt6UiTools",
    "Qt6Charts",
    "Qt6DataVisualization",
    "Qt6NetworkAuth",
    "Qt6RemoteObjects",
    "Qt6Scxml",
    "Qt6StateMachine",
    "Qt6VirtualKeyboard",
    "Qt6HttpServer",
    "Qt6Graphs",
    "Qt6Labs",
)

_QT_EXE_NAMES = frozenset(
    {
        "qmlls.exe",
        "qmltyperegistrar.exe",
        "qmlimportscanner.exe",
        "qmlcachegen.exe",
        "designer.exe",
        "linguist.exe",
        "lrelease.exe",
        "lupdate.exe",
    }
)

# VTK Python extensions / DLL name fragments not required for basic OpenGL + Qt embed.
_VTK_NAME_FRAGMENTS = (
    "vtkIOFFMPEG",
    "vtkIOExportPDF",
    "vtkIOGeoJSON",
    "vtkIOParallel",
    "vtkRenderingVR",
    "vtkRenderingVtkJS",
    "vtkAcceleratorsVTKm",
    "vtkDomainsChemistry",
    "vtkFiltersFlowPaths",
    "vtkFiltersParallel",
    "vtkFiltersTemporal",
    "vtkIOAMR",
    "vtkIOCGNS",
    "vtkIOEnSight",
    "vtkIOExodus",
    "vtkIOFDS",
    "vtkIOHDF",
    "vtkIOIOSS",
    "vtkIOLANLX3D",
    "vtkIOMotionFX",
    "vtkIOMovie",
    "vtkIONetCDF",
    "vtkIOOCCT",
    "vtkIOOMF",
    "vtkIOParaview",
    "vtkIOPIO",
    "vtkIOTecplot",
    "vtkIOVeraOut",
    "vtkIOXMLParser",
    "vtkWeb",
    "vtkTesting",
)


def _should_remove(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    if lower in _QT_EXE_NAMES:
        return True
    if lower == "qml" and path.is_dir():
        return True
    for prefix in _QT_DLL_PREFIXES:
        if name.startswith(prefix):
            return True
    for frag in _VTK_NAME_FRAGMENTS:
        if frag in name:
            return True
    return False


def trim_bundle(root: Path, *, dry_run: bool = False) -> tuple[int, int]:
    """Return (files_removed, bytes_freed)."""
    if not root.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {root}")
    removed = 0
    freed = 0
    # Walk deepest paths first so we can remove files before empty dirs.
    paths = sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for path in paths:
        if not _should_remove(path):
            continue
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            continue
        if dry_run:
            print(f"would remove: {path.relative_to(root)}")
            removed += 1
            freed += size
            continue
        if path.is_file():
            path.unlink(missing_ok=True)
            removed += 1
            freed += size
        elif path.is_dir():
            try:
                path.rmdir()
                removed += 1
            except OSError:
                pass
    return removed, freed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle_dir",
        type=Path,
        nargs="?",
        default=Path("dist/SpotCheck"),
        help="PyInstaller COLLECT output (default: dist/SpotCheck)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    removed, freed = trim_bundle(args.bundle_dir.resolve(), dry_run=args.dry_run)
    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {removed} path(s), freed {freed / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
