from pathlib import Path

import numpy as np

from spot_check.export import build_combined_export_rows, write_combined_export_csv


def test_build_combined_export_rows_nearest_plan() -> None:
    planned = [(0.0, 0.0, 100.0), (5.0, 0.0, 100.0), (0.0, 0.0, 120.0)]
    plan_mu = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    measured = [
        (0.1, 0.2, 0.0, 10.0, 0, float("nan"), float("nan")),
        (0.1, 5.1, 0.0, 8.0, 0, float("nan"), float("nan")),
    ]
    rows = build_combined_export_rows(
        planned,
        plan_mu,
        measured,
        aggregated=False,
        positions_aligned_to_plan=False,
    )
    assert len(rows) == 2
    assert rows[0]["expected_plan_x_mm"] == 0.0
    assert rows[0]["expected_plan_mu"] == 1.0
    assert rows[1]["expected_plan_x_mm"] == 5.0
    assert rows[1]["expected_plan_mu"] == 2.0
    assert rows[0]["aggregated"] == "no"


def test_write_combined_export_csv(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    write_combined_export_csv(
        out,
        [
            {
                "spot_index": 1,
                "aggregated": "yes",
                "layer_index": 0,
                "nominal_energy_mev": 100.0,
                "measured_fit_a_mm": 1.0,
                "measured_fit_b_mm": 2.0,
                "spot_weight": 3.0,
                "partial_code": 0,
                "sigma_a_mm": float("nan"),
                "sigma_b_mm": float("nan"),
                "expected_plan_x_mm": 0.0,
                "expected_plan_y_mm": 0.0,
                "expected_plan_energy_mev": 100.0,
                "expected_plan_mu": 1.5,
                "plan_xy_distance_mm": 0.1,
                "positions_aligned_to_plan": "no",
            }
        ],
        metadata={"test": "1"},
    )
    text = out.read_text(encoding="utf-8")
    assert "# test: 1" in text
    assert "expected_plan_mu" in text
    assert "1.5" in text
