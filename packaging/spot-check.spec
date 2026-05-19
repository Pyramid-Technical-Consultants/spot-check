# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for SpotCheck (Windows one-folder bundle).

Avoid collect_all() — it copies entire PySide6/VTK/SciPy trees (~2× bloat). PyInstaller
hooks + import tracing pull what the app needs; packaging/trim_windows_bundle.py drops
unused Qt/VTK payloads afterward.

Matplotlib is excluded; packaging/pyi_rth_skip_vtk_matplotlib.py stubs VTK's matplotlib
backend (PyVista imports it for LaTeX; SpotCheck uses FreeType axis labels).

SciPy is omitted from the frozen build (plan QA uses a slower fallback).
"""
from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

spec_dir = Path(SPECPATH)
root = spec_dir.parent
src = root / "src"
hooks_dir = spec_dir / "hooks"

block_cipher = None

hiddenimports: list[str] = [
    "pyvista",
    "pyvista.plotting",
    "pyvista.core",
    "pyvista.tracing",
    "vtkmodules",
    "vtkmodules.qt.QVTKRenderWindowInteractor",
    "vtkmodules.util",
    "vtkmodules.util.numpy_support",
    "vtkmodules.numpy_interface.dataset_adapter",
    "vtkmodules.vtkRenderingOpenGL2",
    "vtkmodules.vtkInteractionStyle",
    "vtkmodules.vtkRenderingFreeType",
    "pydicom",
]
hiddenimports += [
    m for m in collect_submodules("pyvista") if not m.startswith("pyvista.trame")
]
datas: list = collect_data_files("pyvista")

excludes: list[str] = [
    "matplotlib",
    "mpl_toolkits",
    "pytest",
    "IPython",
    "notebook",
    "pandas",
    "torch",
    "tensorflow",
    "sklearn",
    "skimage",
    "h5py",
    "numba",
    "bokeh",
    "dask",
    "distributed",
    "xarray",
    "zarr",
    "scipy",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "tkinter",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtBluetooth",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtGraphs",
    "PySide6.QtHelp",
    "PySide6.QtHttpServer",
    "PySide6.QtLocation",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNetworkAuth",
    "PySide6.QtNfc",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtPositioning",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtSerialBus",
    "PySide6.QtStateMachine",
    "PySide6.QtTest",
    "PySide6.QtUiTools",
    "PySide6.QtVirtualKeyboard",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets",
    "PySide6.QtXml",
]

a = Analysis(
    [str(spec_dir / "win_entry.py")],
    pathex=[str(src)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(hooks_dir)],
    hooksconfig={},
    runtime_hooks=[str(spec_dir / "pyi_rth_skip_vtk_matplotlib.py")],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SpotCheck",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SpotCheck",
)
