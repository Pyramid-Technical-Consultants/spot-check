"""GUI pipeline layer-mode resolution for acquisition CSVs."""

from __future__ import annotations

from pathlib import Path

from spot_check.gui.pipeline import aggregation_applies, resolve_csv_load_layer_mode
from tests.conftest import MINIMAL_PLANNED_XYZ, minimal_measured_rows, write_measured_csv


def test_aggregation_applies() -> None:
    assert aggregation_applies(layer_mode="auto", aggregate_spots=True) is True
    assert aggregation_applies(layer_mode="gate_counter", aggregate_spots=True) is True
    assert aggregation_applies(layer_mode="time_gap", aggregate_spots=True) is False
    assert aggregation_applies(layer_mode="auto", aggregate_spots=False) is False


def test_resolve_csv_auto_honors_aggregate(tmp_path: Path) -> None:
    csv_path = write_measured_csv(tmp_path / "g.csv", minimal_measured_rows())
    mode, agg = resolve_csv_load_layer_mode(
        layer_mode="auto",
        plan_path=tmp_path / "plan.dcm",
        csv_path=csv_path,
        aggregate_spots=True,
    )
    assert mode == "auto"
    assert agg is True


def test_resolve_csv_gate_counter_without_column_falls_back_to_time_gap(tmp_path: Path) -> None:
    plain = tmp_path / "no_gate.csv"
    plain.write_text(
        "time (s),IX512 Channel Sum (nA),Fit Amplitude A (nA),"
        "Fit Mean Position A (mm),Fit Mean Position B (mm),Gate Signal\n"  # ignored
        "0,1,0.5,1,2,0\n",
        encoding="utf-8",
    )
    mode, agg = resolve_csv_load_layer_mode(
        layer_mode="gate_counter",
        plan_path=tmp_path / "plan.dcm",
        csv_path=plain,
        aggregate_spots=True,
    )
    assert mode == "time_gap"
    assert agg is False


def test_resolve_csv_gate_counter_with_column_keeps_aggregate(tmp_path: Path) -> None:
    csv_path = write_measured_csv(tmp_path / "g.csv", minimal_measured_rows())
    mode, agg = resolve_csv_load_layer_mode(
        layer_mode="gate_counter",
        plan_path=tmp_path / "plan.dcm",
        csv_path=csv_path,
        aggregate_spots=True,
    )
    assert mode == "gate_counter"
    assert agg is True


def test_resolve_csv_csv_only_gate_counter_becomes_time_gap(tmp_path: Path) -> None:
    csv_path = write_measured_csv(tmp_path / "g.csv", minimal_measured_rows())
    mode, agg = resolve_csv_load_layer_mode(
        layer_mode="gate_counter",
        plan_path=None,
        csv_path=csv_path,
        aggregate_spots=True,
    )
    assert mode == "time_gap"
    assert agg is False


def test_auto_aggregate_false_yields_more_rows_than_true(tmp_path: Path) -> None:
    from spot_check import analysis

    csv_path = write_measured_csv(tmp_path / "many.csv", minimal_measured_rows() * 4)
    planned = list(MINIMAL_PLANNED_XYZ)
    agg = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=planned,
        layer_mode="auto",
        aggregate_spots=True,
        auto_infer_params=False,
        auto_episode_gap_s=0.2,
        auto_spot_xy_jump_mm=3.0,
    )
    raw = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=planned,
        layer_mode="auto",
        aggregate_spots=False,
        auto_infer_params=False,
        auto_episode_gap_s=0.2,
        auto_spot_xy_jump_mm=3.0,
    )
    assert len(raw) >= len(agg)
    assert len(agg) >= 1


def test_measured_probe_no_gate_counter_when_auto_aggregate_flag(tmp_path: Path) -> None:
    from spot_check import analysis

    plain = tmp_path / "no_gate.csv"
    plain.write_text(
        "time (s),IX512 Channel Sum (nA),Fit Amplitude A (nA),"
        "Fit Mean Position A (mm),Fit Mean Position B (mm),Gate Signal\n"  # ignored
        "0,1,0.5,1,2,0\n",
        encoding="utf-8",
    )
    rows = analysis.measured_spot_abc_from_csv(
        plain,
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="auto",
        aggregate_spots=True,
        auto_infer_params=False,
        auto_episode_gap_s=0.2,
        auto_spot_xy_jump_mm=3.0,
    )
    assert rows
