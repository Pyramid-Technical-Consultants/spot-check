"""GUI parsers for upstream WET shifter."""

from __future__ import annotations

import pytest

from spot_check import constants as sc_const
from spot_check.gui.parsers import normalize_z_depth_metric, parse_upstream_wet_shifter_mm


def test_normalize_z_depth_metric_parser() -> None:
    assert normalize_z_depth_metric("r80") == "r80"
    assert normalize_z_depth_metric("CSDA") == "csda"
    assert normalize_z_depth_metric("") == "csda"


def test_parse_upstream_wet_shifter_mm() -> None:
    assert parse_upstream_wet_shifter_mm("0") == 0.0
    assert parse_upstream_wet_shifter_mm(" 12.5 ") == pytest.approx(12.5)
    assert parse_upstream_wet_shifter_mm(str(sc_const.UPSTREAM_WET_SHIFTER_MM_MAX)) == float(
        sc_const.UPSTREAM_WET_SHIFTER_MM_MAX
    )
    assert parse_upstream_wet_shifter_mm("-1") is None
    assert parse_upstream_wet_shifter_mm("501") is None
    assert parse_upstream_wet_shifter_mm("x") is None
