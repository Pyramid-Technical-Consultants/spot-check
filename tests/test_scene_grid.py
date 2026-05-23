"""Custom scene grid (draft 1: XY zero axes + perimeter)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from spot_check.analysis.viz.scene_grid import (
    GridStyle,
    PlanSceneGridController,
    SceneFrame,
    plan_xy_grid,
    xy_boundary_perimeter,
    xy_zero_axes,
)
from spot_check.analysis.viz.scene_grid.geometry import (
    axis_lines_polydata,
    label_anchor_points,
    line_segments_polydata,
    line_segments_tube_mesh,
)
from spot_check.models import ZAxisDisplayConfig

pytest.importorskip("pyvista")


def test_xy_zero_axes_plan_endpoints() -> None:
    frame = SceneFrame(x_min=-10.0, x_max=20.0, y_min=-5.0, y_max=15.0, z_min=3.0, z_max=50.0)
    pad = 4.0
    plan = xy_zero_axes(frame, label_pad_mm=pad)
    assert len(plan.major_lines) == 2
    assert plan.minor_lines == ()
    assert len(plan.labels) == 4

    x_line, y_line = plan.major_lines
    assert x_line.start == (0.0, -10.0, 3.0)
    assert x_line.end == (0.0, 20.0, 3.0)
    assert y_line.start == (-10.0, 0.0, 3.0)
    assert y_line.end == (20.0, 0.0, 3.0)

    positions = {a.position for a in plan.labels}
    assert positions == {
        (0.0, -10.0 - pad, 3.0),
        (0.0, 20.0 + pad, 3.0),
        (-10.0 - pad, 0.0, 3.0),
        (20.0 + pad, 0.0, 3.0),
    }
    assert all(a.text == "0" for a in plan.labels)
    for anchor in plan.labels:
        assert anchor.position not in {
            x_line.start,
            x_line.end,
            y_line.start,
            y_line.end,
        }


def test_xy_boundary_perimeter() -> None:
    frame = SceneFrame(x_min=-10.0, x_max=20.0, y_min=-5.0, y_max=15.0, z_min=3.0, z_max=50.0)
    edges = xy_boundary_perimeter(frame)
    assert len(edges) == 4
    assert edges[0].start == (-10.0, -10.0, 3.0)
    assert edges[0].end == (20.0, -10.0, 3.0)
    assert edges[1].end == (20.0, 20.0, 3.0)
    assert edges[2].end == (-10.0, 20.0, 3.0)
    assert edges[3].end == (-10.0, -10.0, 3.0)


def test_plan_xy_grid_includes_major_and_minor() -> None:
    frame = SceneFrame(x_min=-1.0, x_max=1.0, y_min=-2.0, y_max=2.0, z_min=0.0, z_max=1.0)
    plan = plan_xy_grid(frame)
    assert len(plan.major_lines) == 2
    assert len(plan.minor_lines) == 8
    assert len(plan.labels) == 20


def test_xy_tick_grid_lines_and_labels() -> None:
    from spot_check.analysis.viz.scene_grid.planner import xy_tick_grid_lines_and_labels

    frame = SceneFrame(x_min=-25.0, x_max=25.0, y_min=-15.0, y_max=15.0, z_min=3.0, z_max=50.0)
    lines, labels = xy_tick_grid_lines_and_labels(frame, tick_mm=10.0, label_pad_mm=4.0)
    assert len(lines) == 10
    x_lines = [ln for ln in lines if ln.start[0] == ln.end[0]]
    y_lines = [ln for ln in lines if ln.start[1] == ln.end[1]]
    assert {ln.start[0] for ln in x_lines} == {-30.0, -20.0, -10.0, 10.0, 20.0, 30.0}
    assert {ln.start[1] for ln in y_lines} == {-20.0, -10.0, 10.0, 20.0}
    assert len(labels) == len(lines) * 2
    assert {a.text for a in labels} == {"-30", "-20", "-10", "10", "20", "30"}


def test_plan_xy_grid_includes_tick_grid() -> None:
    frame = SceneFrame(x_min=-25.0, x_max=25.0, y_min=-25.0, y_max=25.0, z_min=0.0, z_max=1.0)
    plan = plan_xy_grid(frame)
    assert len(plan.major_lines) == 2
    assert len(plan.minor_lines) == 4 + 12
    assert len(plan.labels) == 4 + 8 + 24


def test_xy_boundary_labels() -> None:
    frame = SceneFrame(x_min=-10.0, x_max=20.0, y_min=-5.0, y_max=15.0, z_min=3.0, z_max=50.0)
    pad = 4.0
    from spot_check.analysis.viz.scene_grid.planner import xy_boundary_labels

    edges = xy_boundary_perimeter(frame)
    labels = xy_boundary_labels(frame, edges, label_pad_mm=pad)
    assert len(labels) == 8
    texts = {a.text for a in labels}
    assert texts == {"-10", "20"}
    by_text_pos = {(a.text, a.position) for a in labels}
    z = 3.0
    assert ("-10", (-10.0, -10.0 - pad, z)) in by_text_pos
    assert ("20", (20.0, -10.0 - pad, z)) in by_text_pos
    assert ("-10", (20.0 + pad, -10.0, z)) in by_text_pos
    assert ("20", (20.0 + pad, 20.0, z)) in by_text_pos
    assert ("20", (20.0, 20.0 + pad, z)) in by_text_pos
    assert ("-10", (-10.0, 20.0 + pad, z)) in by_text_pos
    assert ("20", (-10.0 - pad, 20.0, z)) in by_text_pos
    assert ("-10", (-10.0 - pad, -10.0, z)) in by_text_pos


def test_grid_style_minor_opacity_scales_major() -> None:
    style = GridStyle(opacity=0.5)
    assert style.minor_opacity == pytest.approx(0.5 * style.minor_opacity_scale)


def test_anchor_beyond_line_end_zero_pad_keeps_endpoint() -> None:
    from spot_check.analysis.viz.scene_grid.label_layout import anchor_beyond_line_end

    end = (0.0, 10.0, 0.0)
    other = (0.0, 0.0, 0.0)
    assert anchor_beyond_line_end(end, other, 0.0) == end
    assert anchor_beyond_line_end(end, other, 2.0) == (0.0, 12.0, 0.0)


def test_axis_lines_polydata_six_segments() -> None:
    frame = SceneFrame(x_min=-1.0, x_max=1.0, y_min=-2.0, y_max=2.0, z_min=0.0, z_max=1.0)
    mesh = axis_lines_polydata(plan_xy_grid(frame))
    assert int(mesh.n_points) == 20
    assert int(mesh.n_lines) == 10


def test_line_segments_tube_mesh_has_no_line_cells() -> None:
    frame = SceneFrame(x_min=-1.0, x_max=1.0, y_min=-2.0, y_max=2.0, z_min=0.0, z_max=1.0)
    tubed = line_segments_tube_mesh(plan_xy_grid(frame).major_lines, radius_mm=0.12)
    assert int(tubed.n_lines) == 0
    assert int(tubed.n_cells) > 0


def test_line_segments_polydata_empty() -> None:
    mesh = line_segments_polydata(())
    assert int(mesh.n_points) == 0


def test_label_anchor_points() -> None:
    frame = SceneFrame(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, z_min=0.0, z_max=1.0)
    pts, texts = label_anchor_points(xy_zero_axes(frame).labels)
    assert pts.shape == (4, 3)
    assert texts == ["0", "0", "0", "0"]


def test_label_anchor_points_below_plane_offset() -> None:
    frame = SceneFrame(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, z_min=3.0, z_max=1.0)
    pts, _ = label_anchor_points(xy_zero_axes(frame).labels, below_plane_mm=1.0)
    assert np.all(pts[:, 2] == pytest.approx(2.0))


def test_billboard_label_text_is_center_aligned() -> None:
    from spot_check.analysis.viz.scene_grid.render import _configure_billboard_text_property

    try:
        from vtkmodules.vtkRenderingCore import vtkTextProperty
    except ImportError:  # pragma: no cover
        from pyvista import _vtk

        vtkTextProperty = _vtk.vtkTextProperty

    tp = vtkTextProperty()
    _configure_billboard_text_property(tp, style=GridStyle())
    assert tp.GetJustification() == 1
    assert tp.GetVerticalJustification() == 1
    assert tp.GetOpacity() == pytest.approx(GridStyle().label_opacity)


def test_plan_scene_grid_controller_show() -> None:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True

    cfg = ZAxisDisplayConfig(use_water_depth_mm=False)
    pl = pv.Plotter(off_screen=True)
    ctrl = PlanSceneGridController(
        xlab="X",
        ylab="Y",
        x_min=-5.0,
        x_max=5.0,
        y_min=-5.0,
        y_max=5.0,
        z_display_cfg=cfg,
    )
    z = np.array([100.0, 120.0], dtype=np.float64)
    e = np.array([100.0, 120.0], dtype=np.float64)
    ctrl.ready = True
    ctrl.show(pl, z, e)
    try:
        assert pl.renderer.cube_axes_actor is None
        assert ctrl._render.major_line_actor is not None
        assert ctrl._render.minor_line_actor is not None
        assert ctrl._render.label_actors
        assert ctrl._render.label_actor is not None
        bounds = ctrl.camera_bounds()
        assert bounds is not None
        assert bounds[0] == pytest.approx(-5.0)
        assert bounds[1] == pytest.approx(5.0)
    finally:
        ctrl.clear(pl)
        pl.close()


def test_plotter_sanity_mode_scene_grid() -> None:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    from spot_check.analysis.viz.plotter import show_comparison_3d_pyvista

    pl = pv.Plotter(off_screen=True)
    show_comparison_3d_pyvista(
        [(5.0, 5.0, 100.0)],
        [],
        title="sanity",
        a_is_x=False,
        z_axis_use_proton_water_depth_mm=False,
        cube_axes_sanity=True,
        reuse_plotter=pl,
        reembed_qt=False,
    )
    try:
        assert pl.renderer.cube_axes_actor is None
        grid = getattr(pl, "_spot_check_scene_grid", None)
        assert grid is not None
        assert grid._render.major_line_actor is not None
        assert grid._render.minor_line_actor is not None
        assert grid.camera_bounds() == (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
    finally:
        pl.close()
