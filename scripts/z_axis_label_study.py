#!/usr/bin/env python3
"""Visual study: Z cube-axis labels vs VTK ``SetAxisLabels`` / axis rebuilds.

Writes PNGs + ``report.txt`` under ``test_output/z_axis_label_study/``.

Key VTK nuance (confirmed here):
- ``SetAxisLabels`` on X/Y (``pin_xy_cube_axis_tick_endpoints``) rebuilds Z ticks too.
- ``update_bounds_axes`` / ``add_mesh`` after cube setup reset custom Z labels.
- Inverted Z must be reapplied *after* every such event.

    python scripts/z_axis_label_study.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import pyvista as pv  # noqa: E402

pv.OFF_SCREEN = True

from spot_check.geometry import (  # noqa: E402
    heal_plan_cube_axes,
    invert_z_cube_axis_tick_labels,
    pin_xy_cube_axis_tick_endpoints,
    plan_cube_scene_bounds_and_axes_ranges,
)
from spot_check.geometry.proton_csda_water import proton_water_depth_mm  # noqa: E402
from spot_check.geometry.z_axis import (  # noqa: E402
    cube_z_axis_label_endpoints,
    cube_z_axis_spec_for_display,
    label_at_scene_z,
    nominal_mev_column_to_scene_z,
)
from spot_check.models import CubeZAxisSpec, ZAxisDisplayConfig  # noqa: E402

OUT = ROOT / "test_output" / "z_axis_label_study"
ENERGIES = np.array([78.5, 100.0, 134.4], dtype=np.float64)
XY = (-40.0, 40.0, -40.0, 40.0)


@dataclass
class Snapshot:
    stage: str
    z_labels: list[float]
    vtk_z_strings: list[str]
    label_at_zmin: float | None
    label_at_zmax: float | None
    ok_inverted: bool
    png: str


def _spec(use_water: bool) -> CubeZAxisSpec:
    e_lo, e_hi = float(np.min(ENERGIES)), float(np.max(ENERGIES))
    cfg = ZAxisDisplayConfig(use_water_depth_mm=use_water, tick_mm=5.0, tick_mev=5.0)
    z = nominal_mev_column_to_scene_z(
        ENERGIES, plan_e_lo=e_lo, plan_e_hi=e_hi, config=cfg
    )
    return cube_z_axis_spec_for_display(z, ENERGIES, cfg)


def _bounds(spec: CubeZAxisSpec) -> tuple[float, float, float, float, float, float]:
    scene, _axes = plan_cube_scene_bounds_and_axes_ranges(*XY, spec)
    return scene


def _axes_ranges(spec: CubeZAxisSpec) -> tuple[float, float, float, float, float, float]:
    _scene, axes = plan_cube_scene_bounds_and_axes_ranges(*XY, spec)
    return axes


def _z_labels(actor: Any) -> list[float]:
    try:
        return [float(actor.z_labels[i]) for i in range(len(actor.z_labels))]
    except Exception:
        return []


def _vtk_z_strings(actor: Any) -> list[str]:
    try:
        arr = actor.GetAxisLabels(2)
        if arr is None:
            return []
        return [str(arr.GetValue(i)) for i in range(arr.GetNumberOfValues())]
    except Exception:
        return []


def _is_inverted(zl: list[float]) -> bool:
    return len(zl) >= 2 and zl[0] > zl[-1]


def _plotter_pipeline(actor: Any, bounds: tuple[float, ...], spec: CubeZAxisSpec) -> None:
    heal_plan_cube_axes(actor, bounds, z_spec=spec, apply_style=True)


def _guard_refresh(actor: Any, bounds: tuple[float, ...], spec: CubeZAxisSpec) -> None:
    heal_plan_cube_axes(actor, bounds, z_spec=spec, apply_style=True)


def _camera(pl: pv.Plotter, bounds: tuple[float, ...]) -> None:
    pl.set_background("#0d1117")
    pl.view_isometric()
    pl.reset_camera(bounds=bounds)
    pl.camera.zoom(1.2)


def _snap(
    pl: pv.Plotter,
    actor: Any,
    bounds: tuple[float, ...],
    stage: str,
    fname: str,
) -> Snapshot:
    _camera(pl, bounds)
    png = OUT / fname
    pl.screenshot(str(png))
    zl = _z_labels(actor)
    return Snapshot(
        stage=stage,
        z_labels=zl,
        vtk_z_strings=_vtk_z_strings(actor),
        label_at_zmin=label_at_scene_z(actor, float(bounds[4])),
        label_at_zmax=label_at_scene_z(actor, float(bounds[5])),
        ok_inverted=_is_inverted(zl),
        png=str(png.relative_to(ROOT)),
    )


def _run_sequence(name: str, spec: CubeZAxisSpec) -> list[Snapshot]:
    bounds = _bounds(spec)
    axes_ranges = _axes_ranges(spec)
    pl = pv.Plotter(off_screen=True, window_size=(960, 720))
    pl.add_mesh(
        pv.Box(bounds=bounds),
        color="#30363d",
        opacity=0.2,
        show_edges=True,
        edge_color="#484f58",
    )
    pts = np.column_stack(
        [
            np.zeros(len(ENERGIES)),
            np.zeros(len(ENERGIES)),
            np.linspace(bounds[4], bounds[5], len(ENERGIES)),
        ]
    )
    pl.add_mesh(pv.PolyData(pts), color="#f0883e", point_size=12)
    actor = pl.show_bounds(
        mesh=None,
        bounds=bounds,
        axes_ranges=axes_ranges,
        location="outer",
        padding=0.0,
        grid="back",
        ticks="inside",
        ztitle=spec.ztitle,
        n_xlabels=5,
        n_ylabels=5,
        n_zlabels=int(spec.n_zlabels),
        fmt="%.0f",
        color="white",
    )
    assert actor is not None
    snaps: list[Snapshot] = []
    tag = name
    snaps.append(
        _snap(pl, actor, bounds, "01 after show_bounds (split axes)", f"{tag}_01_vtk_default.png")
    )
    _plotter_pipeline(actor, bounds, spec)
    snaps.append(_snap(pl, actor, bounds, "02 after plotter pipeline", f"{tag}_02_plotter_ok.png"))
    pin_xy_cube_axis_tick_endpoints(actor)
    snaps.append(
        _snap(pl, actor, bounds, "03 after pin_xy ALONE (Z wiped)", f"{tag}_03_pin_xy_wipes_z.png")
    )
    invert_z_cube_axis_tick_labels(
        actor, z_scene_min=float(bounds[4]), z_scene_max=float(bounds[5])
    )
    snaps.append(_snap(pl, actor, bounds, "04 after re-invert", f"{tag}_04_reinvert_ok.png"))
    pl.renderer.update_bounds_axes()
    snaps.append(
        _snap(
            pl,
            actor,
            bounds,
            "05 after update_bounds_axes (no guard)",
            f"{tag}_05_update_bounds_broken.png",
        )
    )
    _guard_refresh(actor, bounds, spec)
    snaps.append(_snap(pl, actor, bounds, "06 after guard refresh", f"{tag}_06_guard_fixed.png"))
    pl.add_mesh(pv.Sphere(radius=3, center=(0, 0, float(bounds[4]) + 2)), color="#58a6ff")
    snaps.append(
        _snap(pl, actor, bounds, "07 after add_mesh (no guard)", f"{tag}_07_add_mesh_broken.png")
    )
    _guard_refresh(actor, bounds, spec)
    snaps.append(
        _snap(pl, actor, bounds, "08 after guard refresh again", f"{tag}_08_guard_fixed_again.png")
    )
    pl.close()
    return snaps


def _fmt_comparison() -> list[str]:
    """Production Z ticks always use ``PYVISTA_CUBE_Z_LABEL_FORMAT``, not ``show_bounds`` fmt."""
    bounds = (-40.0, 40.0, -40.0, 40.0, 50.14, 130.19)
    lines = ["## show_bounds fmt vs production Z labels (both use fixed-width %.0f)", ""]
    for fmt in ("%.4g", "%.0f"):
        pl = pv.Plotter(off_screen=True, window_size=(960, 720))
        pl.set_background("#0d1117")
        pl.add_mesh(pv.Box(bounds=bounds), opacity=0.2, show_edges=True)
        actor = pl.show_bounds(
            bounds=bounds,
            axes_ranges=bounds,
            padding=0.0,
            n_zlabels=11,
            fmt=fmt,
            grid="back",
            location="outer",
            ztitle="Water depth (mm)",
            color="white",
        )
        heal_plan_cube_axes(actor, bounds)
        _camera(pl, bounds)
        tag = fmt.replace("%", "pct").replace(".", "")
        png = OUT / f"fmt_{tag}.png"
        pl.screenshot(str(png))
        pl.close()
        lines.append(f"  show_bounds fmt={fmt!r} -> {png.relative_to(ROOT)}")
    lines.append("")
    return lines


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_snaps: dict[str, list[dict[str, Any]]] = {}
    lines = [
        "Z axis label study",
        "=" * 72,
        "",
        "Expected (MeV or depth mm): larger value at scene zmin, smaller at zmax.",
        "VTK default with bounds==axes_ranges: ascending labels (WRONG direction).",
        "invert_z_cube_axis_tick_labels: descending labels (CORRECT).",
        "",
        "VTK nuance 1: SetAxisLabels on X/Y rebuilds Z; update_bounds_axes resets Z.",
        "VTK nuance 2: inverted z_axis_range hides Z ticks; split axes_ranges breaks labels.",
        "VTK nuance 3: %.4g custom Z strings overlap visually; use %.0f (plotter default).",
        "",
    ]
    lines.extend(_fmt_comparison())

    for mode, use_water in (("mev", False), ("depth", True)):
        spec = _spec(use_water)
        deep, shallow = cube_z_axis_label_endpoints(spec)
        lines.append(f"## {mode.upper()}  endpoints zmin={deep:.4g}  zmax={shallow:.4g}")
        lines.append("")
        snaps = _run_sequence(mode, spec)
        all_snaps[mode] = [asdict(s) for s in snaps]
        for s in snaps:
            flag = "OK" if s.ok_inverted else "BAD"
            lines.append(f"  [{flag}] {s.stage}")
            lines.append(f"       z_labels[0..-1] = {s.z_labels[0]:.4g} .. {s.z_labels[-1]:.4g}")
            lines.append(f"       label_at_zmin={s.label_at_zmin}  label_at_zmax={s.label_at_zmax}")
            lines.append(f"       png: {s.png}")
            if s.vtk_z_strings and s.vtk_z_strings != [str(x) for x in s.z_labels]:
                lines.append(f"       vtk strings differ: {s.vtk_z_strings}")
            lines.append("")

    if use_water := True:
        deep_mm = float(proton_water_depth_mm(float(np.max(ENERGIES))))
        shallow_mm = float(proton_water_depth_mm(float(np.min(ENERGIES))))
        lines.extend(
            [
                "Depth reference (PSTAR CSDA):",
                f"  134.4 MeV -> {deep_mm:.2f} mm (expect at zmin tick)",
                f"   78.5 MeV -> {shallow_mm:.2f} mm (expect at zmax tick)",
                "",
            ]
        )

    (OUT / "report.json").write_text(json.dumps(all_snaps, indent=2), encoding="utf-8")
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    sys.stdout.buffer.write("\n".join(lines).encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(f"\nWrote study to {OUT}\n".encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
