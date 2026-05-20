from spot_check.gui.parsers import (
    parse_bounds_xy_tick_mm,
    parse_layer_gap_s,
    parse_plan_qa_thresholds,
    plan_qa_thresholds_input_in_progress,
    spot_weight_mode_from_saved,
)


def test_parse_layer_gap_s() -> None:
    assert parse_layer_gap_s("0.2") == 0.2
    assert parse_layer_gap_s("bad") is None


def test_parse_plan_qa_thresholds() -> None:
    assert parse_plan_qa_thresholds("1", "3") == (1.0, 3.0)
    assert parse_plan_qa_thresholds("0.5", "3") == (0.5, 3.0)
    assert parse_plan_qa_thresholds("3", "1") is None


def test_plan_qa_thresholds_input_in_progress() -> None:
    assert plan_qa_thresholds_input_in_progress("0.", "3") is True
    assert plan_qa_thresholds_input_in_progress("0.5", "3.") is True
    assert plan_qa_thresholds_input_in_progress("", "3") is True
    assert plan_qa_thresholds_input_in_progress("0.5", "3") is False


def test_parse_bounds_xy_tick_mm() -> None:
    assert parse_bounds_xy_tick_mm("0") == 0.0
    assert parse_bounds_xy_tick_mm("5") == 5.0


def test_spot_weight_mode_from_saved() -> None:
    assert spot_weight_mode_from_saved("channel_sum") == "channel_sum"
    assert spot_weight_mode_from_saved("fa") == "fit_amplitude_a"
