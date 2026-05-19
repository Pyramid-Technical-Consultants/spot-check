"""PyInstaller entry point for the SpotCheck Windows distribution."""

# Runtime hook stubs vtkRenderingMatplotlib, then imports pyvista. Bind here too so
# spot_check.analysis._core.pv is set even if that module imported before pyvista loaded.
import pyvista

import spot_check.analysis._core as _analysis_core

_analysis_core.pv = pyvista

from spot_check.gui import main

if __name__ == "__main__":
    main()
