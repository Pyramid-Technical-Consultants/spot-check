"""Integration tests for measured CSV layer modes and spatial import wiring."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from spot_check import analysis
from tests.conftest import MINIMAL_PLANNED_XYZ, minimal_measured_rows, write_measured_csv

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ANALYSIS_DIR = _REPO_ROOT / "src" / "spot_check" / "analysis"

_MODULE_SPATIAL_USAGE: dict[str, frozenset[str]] = {
    "measured.py": frozenset(
        {
            "_plan_xy_by_energy_layer",
            "_build_layer_kdtrees",
            "_emit_sqdist_to_layers_mm2",
        }
    ),
    "alignment.py": frozenset({"_plan_xy_by_energy_layer"}),
    "layers.py": frozenset({"_kdtree_query_k1", "_min_xy_dist_to_nominal_energy"}),
}


def _spatial_helper_names_used(module_path: Path, helpers: frozenset[str]) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in helpers:
            used.add(node.id)
    return used


def _names_imported_from_spatial(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "spot_check.analysis.spatial":
            continue
        for alias in node.names:
            imported.add(alias.asname or alias.name)
    return imported


@pytest.mark.parametrize(
    ("layer_mode", "needs_plan"),
    [
        ("gate_counter", True),
        ("unified", True),
        ("plan_viterbi", True),
        ("time_gap", False),
        ("time_gap", True),
    ],
    ids=[
        "gate_counter",
        "unified",
        "plan_viterbi",
        "time_gap_no_plan",
        "time_gap_with_plan",
    ],
)
def test_measured_spot_abc_from_csv_layer_modes(
    tmp_path: Path,
    layer_mode: str,
    needs_plan: bool,
) -> None:
    csv_path = write_measured_csv(tmp_path / "spots.csv", minimal_measured_rows())
    planned = list(MINIMAL_PLANNED_XYZ) if needs_plan else None
    rows = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=planned,
        layer_mode=layer_mode,
        a_is_x=False,
        layer_gap_s=0.2,
        refill_same_spot_xy_tol_mm=3.0,
        refill_trust_time_gap_stay_dist_mm=35.0,
        viterbi_advance_penalty_mm2=400.0,
    )
    assert len(rows) >= 1
    for row in rows:
        assert len(row) >= 4
        assert float(row[3]) > 0


def test_measured_spot_abc_gate_counter_aggregate_even_tail(tmp_path: Path) -> None:
    csv_path = write_measured_csv(tmp_path / "gc.csv", minimal_measured_rows())
    rows = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="gate_counter",
        aggregate_spots=True,
        aggregate_even_rows_after_odd=1,
        a_is_x=False,
    )
    assert len(rows) >= 1


def test_align_measured_to_plan_detector_xy_uses_plan_layers(
    measured_csv_writer,
) -> None:
    csv_path = measured_csv_writer()
    measured = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="gate_counter",
        a_is_x=False,
    )
    aligned, info = analysis.align_measured_to_plan_detector_xy(
        list(MINIMAL_PLANNED_XYZ),
        measured,
        a_is_x=False,
    )
    assert len(aligned) == len(measured)
    assert info is not None


@pytest.mark.parametrize("filename", sorted(_MODULE_SPATIAL_USAGE))
def test_analysis_modules_import_spatial_helpers_they_use(filename: str) -> None:
    """Catch missing spatial imports removed as 'unused' by the linter."""
    module_path = _ANALYSIS_DIR / filename
    expected = _MODULE_SPATIAL_USAGE[filename]
    used = _spatial_helper_names_used(module_path, expected)
    assert used == expected, f"{filename} should reference {expected}, got {used}"
    imported = _names_imported_from_spatial(module_path)
    missing = used - imported
    assert not missing, f"{filename} uses {missing} but does not import them from spatial"
