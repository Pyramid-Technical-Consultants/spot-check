"""Collect PyVista for frozen builds (optional try/except import in analysis skips it otherwise)."""

from __future__ import annotations

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = [
    m for m in collect_submodules("pyvista") if not m.startswith("pyvista.trame")
]
datas = collect_data_files("pyvista")
