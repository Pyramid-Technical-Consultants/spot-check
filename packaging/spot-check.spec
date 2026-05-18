# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for SpotCheck (Windows one-folder bundle)."""
from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

spec_dir = Path(SPECPATH)
root = spec_dir.parent
src = root / "src"

block_cipher = None

hiddenimports: list[str] = [
    "scipy.spatial._ckdtree",
    "scipy.spatial.transform._rotation_groups",
]
hiddenimports += collect_submodules("vtkmodules")
hiddenimports += collect_submodules("pydicom")

datas: list = []
binaries: list = []

for pkg in ("pyvista", "vtkmodules", "PySide6", "pydicom", "scipy"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception:
        pass

a = Analysis(
    [str(spec_dir / "win_entry.py")],
    pathex=[str(src)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pytest",
        "IPython",
        "notebook",
        "pandas",
        "torch",
        "tensorflow",
    ],
    noarchive=False,
    optimize=0,
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
