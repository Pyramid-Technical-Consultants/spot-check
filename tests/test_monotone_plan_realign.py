"""Unit tests for banded monotone plan realignment."""

from __future__ import annotations

import numpy as np

from spot_check.analysis.episodes import _banded_monotone_plan_path


def test_banded_monotone_path_identity_when_aligned() -> None:
    n = 8
    plan = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    cents = plan.copy()
    path = _banded_monotone_plan_path(cents, plan, window=3)
    assert np.array_equal(path, np.arange(n))


def test_banded_monotone_path_stays_near_diagonal() -> None:
    n = 6
    plan = np.column_stack([np.arange(n, dtype=float) * 10.0, np.zeros(n)])
    cents = plan.copy()
    # Episode 2 centroid matches plan 3 (one-index slip).
    cents[2] = plan[3]
    path = _banded_monotone_plan_path(cents, plan, window=3, dup_penalty_mm2=500.0)
    assert int(path[2]) in (2, 3)
    assert np.all(np.diff(path) >= 0)
