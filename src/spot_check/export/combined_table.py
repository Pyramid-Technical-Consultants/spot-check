"""Combined plan vs measured spot table for CSV export."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from spot_check import analysis

COMBINED_EXPORT_COLUMNS: tuple[str, ...] = (
    "spot_index",
    "aggregated",
    "layer_index",
    "nominal_energy_mev",
    "measured_fit_a_mm",
    "measured_fit_b_mm",
    "spot_weight",
    "partial_code",
    "sigma_a_mm",
    "sigma_b_mm",
    "expected_plan_x_mm",
    "expected_plan_y_mm",
    "expected_plan_energy_mev",
    "expected_plan_mu",
    "plan_xy_distance_mm",
    "plan_dose_fraction_pct",
    "meas_dose_fraction_pct",
    "dose_deviation_pp",
    "positions_aligned_to_plan",
)


def _cell(value: Any) -> str | int | float:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (np.floating,)):
        v = float(value)
        return "" if not math.isfinite(v) else v
    return value


def build_combined_export_rows(
    planned_xyz: list[tuple[float, float, float]],
    plan_mu: np.ndarray | None,
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool = False,
    aggregated: bool,
    positions_aligned_to_plan: bool = False,
) -> list[dict[str, Any]]:
    """One row per measured spot with NN plan position and meterset weight on its layer."""
    layer_e = analysis.nominal_layer_energies_mev(planned_xyz)
    dist, exp_xyz, exp_mu = analysis.layer_nn_plan_match_for_measured(
        planned_xyz, plan_mu, measured_rows, a_is_x=a_is_x
    )
    dev_pp, plan_frac, meas_frac, _dist_dose = analysis.plan_dose_fraction_deviation_pp(
        planned_xyz, plan_mu, measured_rows, a_is_x=a_is_x
    )
    rows: list[dict[str, Any]] = []
    for i, tup in enumerate(measured_rows):
        li = int(round(float(tup[2])))
        e_mev = float("nan")
        if layer_e and 0 <= li < len(layer_e):
            e_mev = float(layer_e[li])
        sig_a = float(tup[5]) if len(tup) > 5 else float("nan")
        sig_b = float(tup[6]) if len(tup) > 6 else float("nan")
        ex = ey = ee = float("nan")
        if i < exp_xyz.shape[0]:
            ex, ey, ee = (float(exp_xyz[i, 0]), float(exp_xyz[i, 1]), float(exp_xyz[i, 2]))
        rows.append(
            {
                "spot_index": i + 1,
                "aggregated": "yes" if aggregated else "no",
                "layer_index": li,
                "nominal_energy_mev": e_mev,
                "measured_fit_a_mm": float(tup[0]),
                "measured_fit_b_mm": float(tup[1]),
                "spot_weight": float(tup[3]) if len(tup) > 3 else float("nan"),
                "partial_code": int(tup[4]) if len(tup) > 4 else 0,
                "sigma_a_mm": sig_a,
                "sigma_b_mm": sig_b,
                "expected_plan_x_mm": ex,
                "expected_plan_y_mm": ey,
                "expected_plan_energy_mev": ee,
                "expected_plan_mu": float(exp_mu[i]) if i < exp_mu.shape[0] else float("nan"),
                "plan_xy_distance_mm": float(dist[i]) if i < dist.shape[0] else float("nan"),
                "plan_dose_fraction_pct": (
                    float(plan_frac[i]) * 100.0 if i < plan_frac.shape[0] else float("nan")
                ),
                "meas_dose_fraction_pct": (
                    float(meas_frac[i]) * 100.0 if i < meas_frac.shape[0] else float("nan")
                ),
                "dose_deviation_pp": float(dev_pp[i]) if i < dev_pp.shape[0] else float("nan"),
                "positions_aligned_to_plan": "yes" if positions_aligned_to_plan else "no",
            }
        )
    return rows


def write_combined_export_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write export rows; optional ``# key: value`` preamble lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        if metadata:
            for key, val in metadata.items():
                fh.write(f"# {key}: {val}\n")
        writer = csv.DictWriter(fh, fieldnames=list(COMBINED_EXPORT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k, "")) for k in COMBINED_EXPORT_COLUMNS})
