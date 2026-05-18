from spot_check.gui.parsers import (
    parse_aggregate_even_tail_n,
    parse_bounds_xy_tick_mm,
    parse_layer_gap_s,
    parse_plan_qa_thresholds,
    spot_weight_mode_from_saved,
)


def test_parse_layer_gap_s() -> None:
    assert parse_layer_gap_s("0.2") == 0.2
    assert parse_layer_gap_s("bad") is None


def test_parse_plan_qa_thresholds() -> None:
    assert parse_plan_qa_thresholds("1", "3") == (1.0, 3.0)
    assert parse_plan_qa_thresholds("3", "1") is None


def test_parse_bounds_xy_tick_mm() -> None:
    assert parse_bounds_xy_tick_mm("0") == 0.0
    assert parse_bounds_xy_tick_mm("5") == 5.0


def test_parse_aggregate_even_tail_n() -> None:
    assert parse_aggregate_even_tail_n("0") == 0
    assert parse_aggregate_even_tail_n("99") is None


def test_spot_weight_mode_from_saved() -> None:
    assert spot_weight_mode_from_saved("channel_sum") == "channel_sum"
    assert spot_weight_mode_from_saved("fa") == "fit_amplitude_a"
