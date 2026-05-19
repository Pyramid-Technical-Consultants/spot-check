"""3D comparison visualization (PyVista)."""

from spot_check.analysis.viz.data import prepare_comparison_3d_data
from spot_check.analysis.viz.embed import (
    apply_comparison_3d_camera_view,
    idle_slice_band_controls,
    idle_slice_band_controls_qt,
)
from spot_check.analysis.viz.plotter import show_comparison_3d_pyvista

__all__ = [
    "apply_comparison_3d_camera_view",
    "idle_slice_band_controls",
    "idle_slice_band_controls_qt",
    "prepare_comparison_3d_data",
    "show_comparison_3d_pyvista",
]
