"""PyInstaller entry point for the SpotCheck Windows distribution."""

# Runtime hook stubs vtkRenderingMatplotlib before this runs. Import PyVista here so
# PyInstaller always bundles it (analysis uses try/except in spot_check.analysis._core).
import pyvista  # noqa: F401

from spot_check.gui import main

if __name__ == "__main__":
    main()
