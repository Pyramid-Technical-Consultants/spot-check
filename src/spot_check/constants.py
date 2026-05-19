"""
Central configuration for RT Ion plan vs acquisition analysis.

All distances are millimetres (mm) unless stated otherwise. Nominal beam energies are MeV.

**Not a medical device.** Constants affect heuristics only; validate any clinical workflow
independently before use.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Final

# --- Timing / layer-step heuristics (time-gap & auto episode gaps) -------------
TIME_LAYER_GAP_S_DEFAULT: Final[float] = 0.2

# After a long Δt: same-spot synchrotron refill only if post-gap XY is this close (mm, plan frame).
REFILL_SAME_SPOT_XY_TOLERANCE_MM: Final[float] = 3.0
REFILL_REJECT_EXTRA_MM: Final[float] = 5.0
REFILL_REJECT_RATIO: Final[float] = 1.4
REFILL_TRUST_TIME_GAP_STAY_DIST_MM: Final[float] = 35.0

# --- Viterbi layer assignment (plan_viterbi / auto centroids) -------------------
VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT: Final[float] = 400.0

# --- Auto layer mode (signal episodes + plan-assisted spot count) -------------
AUTO_MIN_ON_SPOT_WEIGHT_NA_DEFAULT: Final[float] = 1e-9
AUTO_SPOT_XY_JUMP_MM_DEFAULT: Final[float] = 3.0
AUTO_MIN_EPISODE_ROWS_DEFAULT: Final[int] = 1
# Delivery-edge segmentation: rows below rolling IX512 sum + fit-A baselines are deadtime.
AUTO_EDGE_DEAD_RATIO_DEFAULT: Final[float] = 0.60
AUTO_EDGE_TINY_MERGE_ROWS: Final[int] = 2
AUTO_EDGE_ROLLING_WINDOW: Final[int] = 15
AUTO_EDGE_ON_WEIGHT_FRAC: Final[float] = 0.45
# When raw episode count equals the plan, skip merge/split alignment.
AUTO_EDGE_COUNT_ALIGN_FRAC: Final[float] = 0.01
# Plan XY hint: search this many rows around each inter-spot boundary when refining spans.
AUTO_PLAN_BOUNDARY_REFINE_ROWS: Final[int] = 32
AUTO_PLAN_BOUNDARY_REFINE_PASSES: Final[int] = 4
# Blend plan vs signal merge cost when aligning episode count (0 = signal only, 1 = plan only).
AUTO_PLAN_MERGE_BLEND: Final[float] = 0.65
# Episode merge scoring: centroid XY delta² (mm²) + coeff × (Δt s)² when aligning M→N_plan.
AUTO_EPISODE_MERGE_DT_MM2_PER_S: Final[float] = 50.0

# --- Nominal energy → water-equivalent depth (display / approximate QA only) -----
# See :mod:`spot_check.geometry.proton_csda_water` (NIST PSTAR table + interpolation).
# **Not for clinical range verification.** Carbon / helium ions require different physics.
UPSTREAM_WET_SHIFTER_MM_DEFAULT: Final[float] = 0.0
UPSTREAM_WET_SHIFTER_MM_MAX: Final[float] = 500.0
# Distal falloff depths from PSTAR CSDA (mm): R90/R80 ≈ scale × R_CSDA − offset (50–230 MeV fits).
# Display only — see :func:`spot_check.geometry.proton_csda_water.proton_water_depth_mm`.
Z_DEPTH_METRIC_DEFAULT: Final[str] = "csda"
Z_DEPTH_METRICS: Final[frozenset[str]] = frozenset({"csda", "r90", "r80"})
PROTON_R90_FROM_CSDA_SCALE: Final[float] = 0.975
PROTON_R90_FROM_CSDA_OFFSET_MM: Final[float] = 0.56
PROTON_R80_FROM_CSDA_SCALE: Final[float] = 0.910
PROTON_R80_FROM_CSDA_OFFSET_MM: Final[float] = 1.14

# --- 3D view (display only; nominal-MeV Z uses stretch — not a physical calibration) ----------
_ENERGY_AXIS_VIEW_SCALE: Final[float] = 2.0
_PLAN_FWHM_GLYPH_Z_SPAN_FRAC: Final[float] = 0.004
_DEFAULT_PLAN_FWHM_MM_WHEN_MISSING: Final[float] = 4.0

# Measured spot σ ellipsoids (3D display): semiaxis = scale * CSV σ (mm); diameter = 2*scale*σ.
# Default 0.5 → 1σ diameter per A/B axis; Z extent shares plan glyph fraction (display only).
MEASURED_SIGMA_GLYPH_SCALE_DEFAULT: Final[float] = 0.5
MEASURED_SIGMA_GLYPH_FALLBACK_MM: Final[float] = 0.35
MEASURED_SIGMA_GLYPH_MIN_MM: Final[float] = 0.02
MEASURED_SIGMA_GLYPH_MAX_MM: Final[float] = 50.0
BOUNDS_XY_TICK_MM_DEFAULT: Final[float] = 5.0
# Max tick labels per cube axis (PyVista linspace count); keeps XY readable on large fields.
BOUNDS_XY_LABELS_MAX: Final[int] = 11
# Cube-axes Z tick step when Z is water-equivalent depth (mm).
BOUNDS_Z_TICK_MM_DEFAULT: Final[float] = 5.0
# Cube-axes Z when it shows nominal energy (MeV); linspace step target (~MeV, not mm).
BOUNDS_Z_TICK_MEV_DEFAULT: Final[float] = 5.0

# --- Large-scene 3D display (GPU / mesh build budget) -----------------------------
# Each FWHM/σ glyph instance expands to a full sphere mesh; cap avoids GPU melt.
DISPLAY_GLYPH_INSTANCE_CAP: Final[int] = 24_000
# Target max points uploaded for measured cloud after optional stride subsampling.
DISPLAY_POINT_MESH_TARGET: Final[int] = 1_250_000
# Max rows used to *fit* detector rigid XY (transform still applied to every row).
DETECTOR_ALIGN_MAX_FIT_SAMPLES: Final[int] = 3_000

# --- Plan QA colouring (XY distance to nearest plan spot on assigned layer) -----
PLAN_QA_PASS_MM_DEFAULT: Final[float] = 1.0
PLAN_QA_WARN_MM_DEFAULT: Final[float] = 3.0
# Dose QA: |measured_layer_% − plan_layer_%| thresholds in percentage points (pp).
PLAN_QA_DOSE_PASS_PP_DEFAULT: Final[float] = 1.0
PLAN_QA_DOSE_WARN_PP_DEFAULT: Final[float] = 3.0
_PLAN_QA_PASS_HEX: Final[str] = "#22c55e"
_PLAN_QA_WARN_HEX: Final[str] = "#eab308"
_PLAN_QA_FAIL_HEX: Final[str] = "#ef4444"
# Dose QA (signed layer %): over-dose reuses warn/fail; under-dose uses cool hues.
_PLAN_QA_DOSE_UNDER_WARN_HEX: Final[str] = "#38bdf8"
_PLAN_QA_DOSE_UNDER_FAIL_HEX: Final[str] = "#a855f7"
_PLAN_COLOR_3D: Final[str] = "#1f77b4"
_MEASURED_COLOR_3D: Final[str] = "#d62728"
_PARTIAL_AXIS_MEAS_COLOR_3D: Final[str] = "#daa520"

# --- CSV column names ----------------------------------------------------------
CHANNEL_SUM_KEY: Final[str] = "IX512 Channel Sum (nA)"
FIT_AMPLITUDE_A_KEY: Final[str] = "Fit Amplitude A (nA)"
FIT_AMPLITUDE_B_KEY: Final[str] = "Fit Amplitude B (nA)"
GATE_COUNTER_KEY: Final[str] = "Gate Counter"
SIGMA_A_KEY: Final[str] = "Fit Standard Deviation A (mm)"
SIGMA_B_KEY: Final[str] = "Fit Standard Deviation B (mm)"

SPOT_WEIGHT_MODE_DEFAULT: Final[str] = "channel_sum"
_SPOT_WEIGHT_MODES: Final[frozenset[str]] = frozenset(
    {"channel_sum", "fit_amplitude_a", "fit_amplitude_b"}
)

AGGREGATE_EVEN_ROWS_AFTER_ODD_DEFAULT: Final[int] = 0
AGGREGATE_EVEN_TAIL_MAX: Final[int] = 32


# --- Repository / project root (directory containing ``pyproject.toml``) ------------
def project_root() -> Path:
    """Project root in dev; directory containing the executable when frozen."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


CSV_LABEL_RE: Final[re.Pattern[str]] = re.compile(r"15186535_(T0G\d+(?:_\d+kHz)?)_", re.IGNORECASE)
