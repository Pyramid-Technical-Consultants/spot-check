"""Bundle vtkmodules .pyd extensions and companion DLLs from vtk.libs (delvewheel layout).

PyInstaller's collect_dynamic_libs('vtkmodules') returns nothing; without vtk.libs the
frozen app fails at import (e.g. vtkFiltersSources DLL load failed).
"""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("vtkmodules")

_vtk_dir = Path(__import__("vtkmodules").__file__).resolve().parent
_binaries: list[tuple[str, str]] = []

for pyd in sorted(_vtk_dir.glob("*.pyd")):
    _binaries.append((str(pyd), "vtkmodules"))

_vtk_libs = _vtk_dir.parent / "vtk.libs"
if _vtk_libs.is_dir():
    for dll in sorted(_vtk_libs.glob("*.dll")):
        _binaries.append((str(dll), "vtk.libs"))

binaries = _binaries
