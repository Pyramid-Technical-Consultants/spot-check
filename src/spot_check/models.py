"""Domain models for plan vs acquisition comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spot_check.constants import BOUNDS_Z_TICK_MEV_DEFAULT, BOUNDS_Z_TICK_MM_DEFAULT


@dataclass(frozen=True)
class ZAxisDisplayConfig:
    """3D view Z mapping: water depth (mm) vs nominal MeV, WET, and cube tick steps."""

    use_water_depth_mm: bool
    upstream_wet_mm: float = 0.0
    z_depth_metric: str = "csda"
    tick_mm: float = BOUNDS_Z_TICK_MM_DEFAULT
    tick_mev: float = BOUNDS_Z_TICK_MEV_DEFAULT


@dataclass(frozen=True)
class CubeZAxisSpec:
    """Cube-axes Z mapping for the 3D scene (scene bounds + tick labels)."""

    zmin_scene: float
    zmax_scene: float
    z_label_at_min: float  # tick label at scene zmin (VTK index 0)
    z_label_at_max: float  # tick label at scene zmax
    n_zlabels: int
    ztitle: str


@dataclass
class Comparison3DData:
    """Prepared arrays and metadata for :func:`spot_check.viz.plot3d.show_comparison_3d_pyvista`."""

    plan_xyz: np.ndarray
    meas_xyz: np.ndarray
    xlab: str
    ylab: str
    e_hi: float
    e_lo: float
    meas_weight: np.ndarray | None = None
    meas_partial_raw: np.ndarray | None = None
    plan_fwhm_xy_mm: np.ndarray | None = None
    meas_sigma_xy_mm: np.ndarray | None = None


@dataclass(frozen=True)
class DetectorRigidAlign2D:
    """2D rigid map measured → plan: ``p ≈ R(θ) @ m + t`` with θ CCW in the plan X–Y plane."""

    theta_deg: float
    tx_mm: float
    ty_mm: float
    rms_nn_mm: float
    rms_residual_mm: float
    n_pairs: int
    ab_axes_swapped: bool = False
    icp_iterations: int = 0
    n_pairs_fit: int = 0
