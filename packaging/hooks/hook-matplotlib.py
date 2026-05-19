"""Collect matplotlib for PyVista plotting (pyvista.plotting.colors imports it at load)."""

from __future__ import annotations

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = collect_submodules("matplotlib")
# mpl-data (matplotlibrc, fonts, etc.) is required at import time.
datas = collect_data_files("matplotlib")
