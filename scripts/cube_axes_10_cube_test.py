#!/usr/bin/env python3
"""Bare-bones 10×10×10 cube + cube axes. PyVista only — no spot_check.

Production 3D view uses the same rule: pass the same 6-tuple for ``bounds`` and
``axes_ranges`` (see ``show_comparison_3d_pyvista``).

    python scripts/cube_axes_10_cube_test.py
"""

from __future__ import annotations

import pyvista as pv

# Scene box 0..10 on X, Y, Z. bounds == axes_ranges so tick numbers match corner coords.
BOUNDS = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)

pl = pv.Plotter(window_size=(800, 600), title="10 cube — bare PyVista")
pl.set_background("black")

pl.add_mesh(
    pv.Box(bounds=BOUNDS),
    color="tan",
    opacity=0.4,
    show_edges=True,
)

# padding=0 is required: PyVista padding expands bounds but not axes_ranges (Z ticks skew).
pl.show_bounds(
    mesh=None,
    bounds=BOUNDS,
    axes_ranges=BOUNDS,
    location="outer",
    padding=0.0,
    xtitle="X",
    ytitle="Y",
    ztitle="Z",
    n_xlabels=6,
    n_ylabels=6,
    n_zlabels=6,
    fmt="%.0f",
    color="white",
)

pl.reset_camera(bounds=BOUNDS)
print("Box and axes: 0 .. 10 on X, Y, Z")
print("Close the window to exit.")
pl.show()
