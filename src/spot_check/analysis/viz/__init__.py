"""3D comparison visualization (PyVista)."""

from spot_check.analysis.viz.data import prepare_comparison_3d_data
from spot_check.analysis.viz.embed import (
    apply_comparison_3d_camera_view,
    apply_comparison_3d_projection_view,
    idle_slice_band_controls,
    idle_slice_band_controls_qt,
    idle_time_slice_controls,
    idle_time_slice_controls_qt,
)
from spot_check.analysis.viz.plotter import (
    refresh_comparison_3d_display,
    show_comparison_3d_pyvista,
)

__all__ = [
    "apply_comparison_3d_camera_view",
    "apply_comparison_3d_projection_view",
    "idle_slice_band_controls",
    "idle_slice_band_controls_qt",
    "idle_time_slice_controls",
    "idle_time_slice_controls_qt",
    "prepare_comparison_3d_data",
    "refresh_comparison_3d_display",
    "show_comparison_3d_pyvista",
]
