"""Shared imports and module-level state for analysis submodules."""

from __future__ import annotations

import bisect
import csv
import importlib.util
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if importlib.util.find_spec("pydicom") is None:  # pragma: no cover
    raise ImportError("RT Ion analysis requires pydicom. Install with: pip install pydicom")

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore[assignment, misc]
    ttk = None  # type: ignore[assignment, misc]

from spot_check.constants import (
    _DEFAULT_PLAN_FWHM_MM_WHEN_MISSING,
    _MEASURED_COLOR_3D,
    _PARTIAL_AXIS_MEAS_COLOR_3D,
    _PLAN_COLOR_3D,
    _PLAN_FWHM_GLYPH_Z_SPAN_FRAC,
    _PLAN_QA_DOSE_UNDER_FAIL_HEX,
    _PLAN_QA_DOSE_UNDER_WARN_HEX,
    _PLAN_QA_FAIL_HEX,
    _PLAN_QA_PASS_HEX,
    _PLAN_QA_WARN_HEX,
    _SPOT_WEIGHT_MODES,
    AUTO_EPISODE_MERGE_DT_MM2_PER_S,
    AUTO_MIN_EPISODE_ROWS_DEFAULT,
    AUTO_MIN_ON_SPOT_WEIGHT_NA_DEFAULT,
    AUTO_SPOT_XY_JUMP_MM_DEFAULT,
    BOUNDS_XY_TICK_MM_DEFAULT,
    CHANNEL_SUM_KEY,
    DETECTOR_ALIGN_MAX_FIT_SAMPLES,
    DISPLAY_GLYPH_INSTANCE_CAP,
    DISPLAY_POINT_MESH_TARGET,
    FIT_AMPLITUDE_A_KEY,
    FIT_AMPLITUDE_B_KEY,
    GATE_COUNTER_KEY,
    MEASURED_SIGMA_GLYPH_FALLBACK_MM,
    MEASURED_SIGMA_GLYPH_MAX_MM,
    MEASURED_SIGMA_GLYPH_MIN_MM,
    MEASURED_SIGMA_GLYPH_SCALE_DEFAULT,
    PLAN_QA_DOSE_PASS_PP_DEFAULT,
    PLAN_QA_DOSE_WARN_PP_DEFAULT,
    PLAN_QA_PASS_MM_DEFAULT,
    PLAN_QA_WARN_MM_DEFAULT,
    REFILL_REJECT_EXTRA_MM,
    REFILL_REJECT_RATIO,
    REFILL_SAME_SPOT_XY_TOLERANCE_MM,
    REFILL_TRUST_TIME_GAP_STAY_DIST_MM,
    SIGMA_A_KEY,
    SIGMA_B_KEY,
    SPOT_WEIGHT_MODE_DEFAULT,
    TIME_LAYER_GAP_S_DEFAULT,
    VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT,
    project_root,
)
from spot_check.exceptions import (
    AcquisitionDataError,
    GeometryConfigError,
    PlanDataError,
)
from spot_check.geometry import (
    PYVISTA_CUBE_AXES_GRID,
    PYVISTA_CUBE_AXES_LOCATION,
    PYVISTA_CUBE_AXES_PADDING,
    PYVISTA_CUBE_AXES_TICKS,
    apply_pyvista_cube_axes_style,
    apply_z_display_to_comparison_clouds,
    cube_axes_ranges,
    disable_pyvista_cube_axes_label_lod,
    invert_z_cube_axis_tick_labels,
    n_cube_axis_labels_for_mm_step,
    nominal_depth_to_scene_z_cube,
    nominal_energy_to_scene_z,
    nominal_mev_to_scene_z_mev_cube,
    normalize_cube_axes_label_counts,
    pin_pyvista_cube_bounds,
    pin_xy_cube_axis_tick_endpoints,
    plan_depth_bounds_mm,
    pyvista_show_bounds_kwargs,
    refresh_pyvista_cube_axes,
)
from spot_check.geometry import (
    cube_z_axis_spec as _cube_z_axis_spec,
)
from spot_check.geometry import (
    cube_z_axis_spec_for_display as _cube_z_axis_spec_for_display,
)
from spot_check.models import (
    Comparison3DData,
    CubeZAxisSpec,
    DetectorRigidAlign2D,
    ZAxisDisplayConfig,
)

try:
    from scipy.spatial import cKDTree as _cKDTree
except ImportError:  # pragma: no cover
    _cKDTree = None

FOLDER = project_root()
logger = logging.getLogger(__name__)

_CubeZAxisSpec = CubeZAxisSpec

__all__ = [
    "AcquisitionDataError",
    "Any",
    "AUTO_EPISODE_MERGE_DT_MM2_PER_S",
    "AUTO_MIN_EPISODE_ROWS_DEFAULT",
    "AUTO_MIN_ON_SPOT_WEIGHT_NA_DEFAULT",
    "AUTO_SPOT_XY_JUMP_MM_DEFAULT",
    "BOUNDS_XY_TICK_MM_DEFAULT",
    "CHANNEL_SUM_KEY",
    "Comparison3DData",
    "CubeZAxisSpec",
    "ZAxisDisplayConfig",
    "DETECTOR_ALIGN_MAX_FIT_SAMPLES",
    "DISPLAY_GLYPH_INSTANCE_CAP",
    "DISPLAY_POINT_MESH_TARGET",
    "DetectorRigidAlign2D",
    "FIT_AMPLITUDE_A_KEY",
    "FIT_AMPLITUDE_B_KEY",
    "FOLDER",
    "GATE_COUNTER_KEY",
    "GeometryConfigError",
    "MEASURED_SIGMA_GLYPH_FALLBACK_MM",
    "MEASURED_SIGMA_GLYPH_MAX_MM",
    "MEASURED_SIGMA_GLYPH_MIN_MM",
    "MEASURED_SIGMA_GLYPH_SCALE_DEFAULT",
    "PLAN_QA_DOSE_PASS_PP_DEFAULT",
    "PLAN_QA_DOSE_WARN_PP_DEFAULT",
    "PLAN_QA_PASS_MM_DEFAULT",
    "PLAN_QA_WARN_MM_DEFAULT",
    "PYVISTA_CUBE_AXES_GRID",
    "PYVISTA_CUBE_AXES_LOCATION",
    "PYVISTA_CUBE_AXES_PADDING",
    "PYVISTA_CUBE_AXES_TICKS",
    "Path",
    "PlanDataError",
    "REFILL_REJECT_EXTRA_MM",
    "REFILL_REJECT_RATIO",
    "REFILL_SAME_SPOT_XY_TOLERANCE_MM",
    "REFILL_TRUST_TIME_GAP_STAY_DIST_MM",
    "SIGMA_A_KEY",
    "SIGMA_B_KEY",
    "SPOT_WEIGHT_MODE_DEFAULT",
    "Sequence",
    "TIME_LAYER_GAP_S_DEFAULT",
    "VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT",
    "_CubeZAxisSpec",
    "_DEFAULT_PLAN_FWHM_MM_WHEN_MISSING",
    "_MEASURED_COLOR_3D",
    "_PARTIAL_AXIS_MEAS_COLOR_3D",
    "_PLAN_COLOR_3D",
    "_PLAN_FWHM_GLYPH_Z_SPAN_FRAC",
    "_PLAN_QA_DOSE_UNDER_FAIL_HEX",
    "_PLAN_QA_DOSE_UNDER_WARN_HEX",
    "_PLAN_QA_FAIL_HEX",
    "_PLAN_QA_PASS_HEX",
    "_PLAN_QA_WARN_HEX",
    "_SPOT_WEIGHT_MODES",
    "_cKDTree",
    "_cube_z_axis_spec",
    "_cube_z_axis_spec_for_display",
    "apply_pyvista_cube_axes_style",
    "apply_z_display_to_comparison_clouds",
    "cube_axes_ranges",
    "disable_pyvista_cube_axes_label_lod",
    "invert_z_cube_axis_tick_labels",
    "bisect",
    "csv",
    "dataclass",
    "logger",
    "logging",
    "math",
    "n_cube_axis_labels_for_mm_step",
    "nominal_energy_to_scene_z",
    "nominal_depth_to_scene_z_cube",
    "nominal_mev_to_scene_z_mev_cube",
    "normalize_cube_axes_label_counts",
    "plan_depth_bounds_mm",
    "pyvista_show_bounds_kwargs",
    "pin_pyvista_cube_bounds",
    "pin_xy_cube_axis_tick_endpoints",
    "refresh_pyvista_cube_axes",
    "np",
    "tk",
    "ttk",
]
