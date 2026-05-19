"""Tests for post-PyInstaller bundle trimming."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _trim_module():
    path = Path(__file__).resolve().parents[1] / "packaging" / "trim_windows_bundle.py"
    spec = importlib.util.spec_from_file_location("trim_windows_bundle", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trim_bundle_never_removes_vtk_libs(tmp_path: Path) -> None:
    trim_bundle = _trim_module().trim_bundle
    root = tmp_path / "SpotCheck" / "_internal"
    vtklibs = root / "vtk.libs"
    vtklibs.mkdir(parents=True)
    keep = vtklibs / "vtkFiltersSources-9.6.1.dll"
    keep.write_bytes(b"x" * 800)
    io_dll = vtklibs / "vtkIOFFMPEG-9.6.1.dll"
    io_dll.write_bytes(b"y" * 800)
    removed, _freed = trim_bundle(root.parent, aggressive=False)
    assert keep.exists()
    assert io_dll.exists()
    assert removed == 0


def test_trim_bundle_removes_qt_quick_dll(tmp_path: Path) -> None:
    trim_bundle = _trim_module().trim_bundle
    root = tmp_path / "SpotCheck"
    root.mkdir()
    quick = root / "Qt6Quick.dll"
    quick.write_bytes(b"x" * 1000)
    keep = root / "Qt6Widgets.dll"
    keep.write_bytes(b"y" * 500)
    removed, freed = trim_bundle(root, aggressive=False)
    assert removed >= 1
    assert freed >= 1000
    assert not quick.exists()
    assert keep.exists()
