"""Remove Qt/VTK/PIL payloads not needed by SpotCheck after PyInstaller COLLECT.

Safe to run on dist/SpotCheck/ (one-folder layout). Deletes only known-unused names;
re-run the GUI smoke test after changing this list.
"""

from __future__ import annotations

import argparse
import os
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
    "Qt6ShaderTools",
    "Qt6SpatialAudio",
    "Qt6SvgWidgets",
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

# VTK payloads required for PyVista import + SpotCheck 3D (never trim).
_VTK_KEEP_FRAGMENTS = (
    "vtkFiltersSources",
    "vtkCommonCore",
    "vtkCommonDataModel",
    "vtkCommonExecutionModel",
    "vtkCommonMath",
    "vtkCommonTransforms",
    "vtkCommonSystem",
    "vtkCommonMisc",
    "vtkCommonComputationalGeometry",
    "vtkFiltersGeneral",
    "vtkFiltersCore",
    "vtkFiltersGeometry",
    "vtkRenderingCore",
    "vtkRenderingOpenGL2",
    "vtkInteractionStyle",
    "vtkRenderingFreeType",
    "vtkRenderingUI",
    "vtkImagingCore",
    "vtkIOImage",
    "vtkIOCore",
    "vtkIOXML",
    "vtkIOXMLParser",
    "vtkzlib",
    "vtkdoubleconversion",
    "vtkexpat",
    "vtkfmt",
    "vtkglew",
    "vtkjpeg",
    "vtkjson",
    "vtkkissfft",
    "vtklz4",
    "vtklzma",
    "vtkmetaio",
    "vtkpng",
    "vtksys",
    "vtktiff",
    "vtkzlib",
    "vtkloguru",
    "vtkpugixml",
    "vtktoken",
    "vtkWrappingPythonCore",
)

# VTK DLL / PYD name fragments not required for OpenGL scatter + cube axes + Qt embed.
_VTK_NAME_FRAGMENTS = (
    "vtkIOFFMPEG",
    "vtkIOExportPDF",
    "vtkIOGeoJSON",
    "vtkIOParallel",
    "vtkRenderingVR",
    "vtkRenderingVtkJS",
    "vtkRenderingVolume",
    "vtkChartsCore",
    "vtkAcceleratorsVTKm",
    "viskores_",
    "vtkDomainsChemistry",
    "vtkFiltersFlowPaths",
    "vtkFiltersParallel",
    "vtkFiltersTemporal",
    "vtkFiltersProgrammable",
    "vtkFiltersReduction",
    "vtkFiltersTensor",
    "vtkFiltersTopology",
    "vtkFiltersVerdict",
    "vtkFiltersGeometryPreview",
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
    "vtkRenderingLIC",
    "vtkRenderingLabel",
    "vtkRenderingImage",
    "vtkRenderingContext",
    "vtkRenderingCellGrid",
    "vtkRenderingHyperTreeGrid",
    "vtkRenderingGridAxes",
    "vtkWeb",
    "vtkTesting",
    "vtkPythonContext",
)

# Pillow codecs not needed (matplotlib excluded; PyVista screenshots use PNG via VTK).
_PIL_CODEC_FRAGMENTS = (
    "_avif",
    "_imagingft",
    "_imagingcms",
    "_imagingtk",
    "_webp",
)

# Matplotlib payload if it was collected anyway (keep matplotlibrc + minimal fonts).
_MPL_DIR_NAMES = frozenset({"sample_data", "stylelib"})
_MPL_KEEP_FILES = frozenset({"matplotlibrc", "fontlist-v390.json", "fontlist-v330.json"})

_KEEP_MPL_FONT_PREFIXES = ("DejaVuSans", "DejaVuSansDisplay", "STIXGeneral")


def _bundle_root(root: Path) -> Path:
    internal = root / "_internal"
    return internal if internal.is_dir() else root


def _should_remove(path: Path, *, bundle: Path, aggressive: bool) -> bool:
    name = path.name
    lower = name.lower()
    rel = path.relative_to(bundle)
    rel_s = rel.as_posix()

    # Never trim VTK payloads: substring rules can delete required DLLs (e.g. vtkFiltersSources).
    if "vtk.libs" in rel.parts or rel_s.startswith("vtkmodules/"):
        return False

    if lower in _QT_EXE_NAMES:
        return True
    if path.is_dir() and name == "qml":
        return True
    if path.is_dir() and name == "translations" and rel_s.startswith("PySide6/"):
        return True
    for prefix in _QT_DLL_PREFIXES:
        if name.startswith(prefix):
            return True
    for frag in _VTK_NAME_FRAGMENTS:
        if frag in name:
            return True
    for frag in _PIL_CODEC_FRAGMENTS:
        if frag in name:
            return True

    if path.is_dir() and name in _MPL_DIR_NAMES and "matplotlib" in rel.parts:
        return True

    if path.is_file() and "matplotlib/mpl-data" in rel_s:
        if name in _MPL_KEEP_FILES:
            return False
    if path.is_file() and "matplotlib/mpl-data/fonts" in rel_s:
        if not any(name.startswith(p) for p in _KEEP_MPL_FONT_PREFIXES):
            return True

    if aggressive and name == "opengl32sw.dll":
        return True

    return False


def trim_bundle(
    root: Path,
    *,
    dry_run: bool = False,
    aggressive: bool | None = None,
) -> tuple[int, int]:
    """Return (files_removed, bytes_freed)."""
    if not root.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {root}")
    if aggressive is None:
        aggressive = os.environ.get("SPOT_CHECK_TRIM_AGGRESSIVE", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )
    bundle = _bundle_root(root)
    removed = 0
    freed = 0
    paths = sorted(bundle.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for path in paths:
        if not _should_remove(path, bundle=bundle, aggressive=aggressive):
            continue
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            continue
        if path.is_dir() and not path.is_file():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
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
                import shutil

                shutil.rmtree(path)
                removed += 1
                freed += size
            except OSError:
                pass
    return removed, freed


def report_bundle(root: Path, *, top_n: int = 25) -> None:
    bundle = _bundle_root(root)
    files: list[tuple[int, str]] = []
    for f in bundle.rglob("*"):
        if f.is_file():
            files.append((f.stat().st_size, f.relative_to(bundle).as_posix()))
    print(f"Bundle root: {bundle} ({sum(s for s, _ in files) / 1e6:.1f} MB, {len(files)} files)")
    for sz, name in sorted(files, reverse=True)[:top_n]:
        print(f"  {sz / 1e6:7.2f} MB  {name}")


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
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Also remove opengl32sw.dll (~21MB; needs working GPU OpenGL)",
    )
    parser.add_argument(
        "--no-aggressive",
        action="store_true",
        help="Keep opengl32sw.dll even if SPOT_CHECK_TRIM_AGGRESSIVE is set",
    )
    parser.add_argument("--report", action="store_true", help="Print largest files and exit")
    args = parser.parse_args(argv)
    root = args.bundle_dir.resolve()
    if args.report:
        report_bundle(root)
        return 0
    aggressive = None
    if args.aggressive:
        aggressive = True
    if args.no_aggressive:
        aggressive = False
    removed, freed = trim_bundle(root, dry_run=args.dry_run, aggressive=aggressive)
    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {removed} path(s), freed {freed / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
