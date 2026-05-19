"""Collect matplotlib for PyVista plotting (pyvista.plotting.colors imports it at load)."""

from __future__ import annotations

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("matplotlib")
