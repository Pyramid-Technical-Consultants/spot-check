"""Backward-compatible re-exports for spot_check.analysis._core."""

from __future__ import annotations

from spot_check.analysis import (
    _imports as analysis_imports,
)
from spot_check.analysis import (
    alignment,
    colors,
    layers,
    measured,
    plan_qa,
    pyvista_backend,
    spatial,
)
from spot_check.analysis import (
    auto_params as auto_params_mod,
)
from spot_check.analysis import (
    episodes as episodes_mod,
)
from spot_check.analysis.viz import data as viz_data
from spot_check.analysis.viz import embed as viz_embed
from spot_check.analysis.viz import glyphs as viz_glyphs
from spot_check.analysis.viz import plotter as viz_plotter

_SUBMODULES = (
    analysis_imports,
    pyvista_backend,
    colors,
    spatial,
    plan_qa,
    alignment,
    layers,
    measured,
    auto_params_mod,
    episodes_mod,
    viz_data,
    viz_glyphs,
    viz_embed,
    viz_plotter,
)

for _sub in _SUBMODULES:
    for _name in dir(_sub):
        if _name.startswith("__"):
            continue
        globals()[_name] = getattr(_sub, _name)

del _sub, _name, _SUBMODULES
