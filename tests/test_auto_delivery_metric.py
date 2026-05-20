"""Auto deadtime metric works across accelerator CSV scales (T0G10 vs IC256 cube)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spot_check.analysis.auto_columns import load_auto_fit_columns_from_csv
from spot_check.analysis.auto_params import infer_auto_layer_params
from spot_check.analysis.episodes import count_episodes_for_dead_ratio
from spot_check.analysis.measured import _PlanImputeLookup
from spot_check.constants import AUTO_EDGE_DEAD_RATIO_MAX, AUTO_EDGE_DEAD_RATIO_MIN

_RUN12 = (
    Path(__file__).resolve().parents[1]
    / "test_data"
    / "run12-cube-ic256-42-11377-data acquisition-2026-05-19-23-07-25.csv"
)
_T0G10 = (
    Path(__file__).resolve().parents[1]
    / "test_data"
    / "15186535_T0G10_ic256-45-9018-data acquisition-2026-05-06-16-27-25.csv"
)

pytestmark = pytest.mark.local_data


def _episode_count_at_ratio(cols, ratio: float) -> int:
    return count_episodes_for_dead_ratio(
        cols, ratio, min_episode_rows=1, tiny_merge_rows=2
    )


@pytest.mark.skipif(not _RUN12.is_file(), reason="run12 cube CSV not under test_data/")
def test_run12_reaches_more_episodes_at_high_ratio_than_legacy_cap() -> None:
    """IC256 cube CSV can segment more spots when calibration uses the widened ratio range."""
    lk = _PlanImputeLookup.from_xy(np.zeros((1, 2)))
    cols = load_auto_fit_columns_from_csv(
        _RUN12, global_lk=lk, a_is_x=False, spot_weight_mode="channel_sum"
    )
    at_old_cap = _episode_count_at_ratio(cols, 0.64)
    at_new_cap = _episode_count_at_ratio(cols, AUTO_EDGE_DEAD_RATIO_MAX)
    assert at_new_cap >= at_old_cap
    assert at_new_cap > 3000


@pytest.mark.skipif(not _RUN12.is_file(), reason="run12 cube CSV not under test_data/")
def test_calibrated_dead_ratio_uses_widened_range_on_run12() -> None:
    lk = _PlanImputeLookup.from_xy(np.zeros((1, 2)))
    cols = load_auto_fit_columns_from_csv(
        _RUN12, global_lk=lk, a_is_x=False, spot_weight_mode="channel_sum"
    )
    # Synthetic plan larger than legacy 0.64 cap could reach; calibration should use high end.
    n_plan = 3600
    p = infer_auto_layer_params(cols, [(0.0, 0.0, 100.0 + i) for i in range(n_plan)])
    assert AUTO_EDGE_DEAD_RATIO_MIN <= p.dead_ratio <= AUTO_EDGE_DEAD_RATIO_MAX
    n_ep = _episode_count_at_ratio(cols, p.dead_ratio)
    assert abs(n_ep - n_plan) < max(400, int(0.15 * n_plan))


@pytest.mark.skipif(not _T0G10.is_file(), reason="T0G10 acquisition CSV not under test_data/")
def test_t0g10_calibrated_ratio_still_near_plan_scale() -> None:
    lk = _PlanImputeLookup.from_xy(np.zeros((1, 2)))
    cols = load_auto_fit_columns_from_csv(
        _T0G10, global_lk=lk, a_is_x=False, spot_weight_mode="channel_sum"
    )
    n_plan = 12_779
    p = infer_auto_layer_params(cols, [(0.0, 0.0, 100.0 + i) for i in range(n_plan)])
    n_ep = _episode_count_at_ratio(cols, p.dead_ratio)
    assert abs(n_ep - n_plan) < max(800, int(0.08 * n_plan))
