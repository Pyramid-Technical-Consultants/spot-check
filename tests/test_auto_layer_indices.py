"""Auto nominal-layer assignment when episode count != plan spot count."""

from __future__ import annotations

import numpy as np

from spot_check.analysis.layers import auto_episode_layer_indices, delivery_layer_indices


def test_delivery_layer_indices_prefix_stays_on_layer_zero() -> None:
    """Gate-style mapping: fewer spots than layer 0 count => all layer 0."""
    layers = delivery_layer_indices(100, [5000, 5000, 5000])
    assert int(np.max(layers)) == 0


def test_auto_episode_layer_indices_spreads_across_layers() -> None:
    layers = auto_episode_layer_indices(3000, [5000, 5000, 5000])
    assert int(np.min(layers)) == 0
    assert int(np.max(layers)) == 2
    assert len(np.unique(layers)) >= 2


def test_auto_episode_layer_indices_respects_plan_weights() -> None:
    layers = auto_episode_layer_indices(1000, [100, 200, 700])
    assert int(np.sum(layers == 0)) < 200
    assert int(np.sum(layers == 2)) > 400
