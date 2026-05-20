"""GUI layer-assignment mode normalization and pipeline mapping."""

from __future__ import annotations

from spot_check.gui.layer_assign import (
    normalize_layer_assign_mode,
    resolve_layer_assign_mode,
)


def test_normalize_layer_assign_legacy_aliases() -> None:
    assert normalize_layer_assign_mode("unified") == "auto"
    assert normalize_layer_assign_mode("time_gap") == "gate_counter"
    assert normalize_layer_assign_mode("auto_layer_em") == "auto"
    assert normalize_layer_assign_mode("bogus") == "gate_counter"


def test_normalize_auto_plan_sequential_aliases() -> None:
    assert normalize_layer_assign_mode("plan_sequential") == "auto_plan_sequential"
    assert normalize_layer_assign_mode("auto_plan_sequential") == "auto_plan_sequential"


def test_resolve_layer_assign_mode() -> None:
    assert resolve_layer_assign_mode("gate_counter") == ("gate_counter", "episodes", False)
    assert resolve_layer_assign_mode("auto") == ("auto", "episodes", True)
    assert resolve_layer_assign_mode("auto_plan_sequential") == (
        "auto",
        "plan_sequential",
        True,
    )
