#!/usr/bin/env python3
"""0..10 cube axes inside the real plotter (sanity mode).

    set SPOT_CHECK_CUBE_AXES_SANITY=1
    python scripts/cube_axes_sanity_check.py

Or pass data as usual; cube box is fixed 0..10 on X/Y/Z with ``bounds == axes_ranges``.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("SPOT_CHECK_CUBE_AXES_SANITY", "1")

import pyvista as pv

from spot_check.analysis.viz.plotter import show_comparison_3d_pyvista

# One plan spot inside the sanity box; measured optional.
PLANNED = [(5.0, 5.0, 100.0)]

pl = pv.Plotter(window_size=(900, 700), title="Cube axes sanity 0..10")
pl.set_background("#0d1117")
show_comparison_3d_pyvista(
    PLANNED,
    [],
    title="Sanity: cube axes 0..10",
    a_is_x=False,
    z_axis_use_proton_water_depth_mm=False,
    cube_axes_sanity=True,
    reuse_plotter=pl,
    reembed_qt=False,
)
actor = pl.renderer.cube_axes_actor
if actor is not None:
    zl = [actor.z_labels[i] for i in range(len(actor.z_labels))]
    print("Z tick labels:", zl)
    print("Expected ~0 .. 10 on X, Y, and Z")
else:
    print("No cube_axes_actor", file=sys.stderr)
    sys.exit(1)
pl.show()
