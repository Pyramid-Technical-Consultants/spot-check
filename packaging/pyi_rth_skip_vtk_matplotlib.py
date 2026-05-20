"""PyInstaller runtime hook: skip VTK's matplotlib text backend.

PyVista imports ``vtkmodules.vtkRenderingMatplotlib`` for LaTeX/math labels. SpotCheck
uses FreeType axis labels (PyVista ``show_bounds`` / ``CubeAxesActor``) and does
not need matplotlib (~15MB+ in the bundle).
"""

from __future__ import annotations

import sys
import types

_NAME = "vtkmodules.vtkRenderingMatplotlib"
if _NAME not in sys.modules:
    _stub = types.ModuleType(_NAME)
    _stub.__doc__ = "Stubbed in frozen SpotCheck build (matplotlib not required)."
    sys.modules[_NAME] = _stub

# Load PyVista after the stub so analysis._core sees a working import (before win_entry).
try:
    import pyvista  # noqa: F401
except ImportError:
    pass
