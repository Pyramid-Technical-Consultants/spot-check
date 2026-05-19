"""PyInstaller entry point for the SpotCheck Windows distribution."""

import os
import sys

# Post-build / CI smoke: SPOT_CHECK_SMOKE=1 SpotCheck.exe  -> import VTK and exit 0.
if os.environ.get("SPOT_CHECK_SMOKE", "").strip() in ("1", "true", "yes"):
    try:
        import pyvista as _pv

        _pl = _pv.Plotter(off_screen=True)
        _pl.close()
        print(f"SPOT_CHECK_SMOKE ok pyvista={_pv.__version__}", file=sys.stderr)
    except Exception as exc:
        print(f"SPOT_CHECK_SMOKE failed: {exc}", file=sys.stderr)
        # console=False builds can hang CI on an uncaught-exception dialog; _exit skips that.
        os._exit(1)
    os._exit(0)

# Runtime hook stubs vtkRenderingMatplotlib, then imports pyvista. Bind here too so
# spot_check.analysis._core.pv is set even if that module imported before pyvista loaded.
import pyvista

import spot_check.analysis._core as _analysis_core

_analysis_core.pv = pyvista

from spot_check.gui import main  # noqa: E402

if __name__ == "__main__":
    main()
