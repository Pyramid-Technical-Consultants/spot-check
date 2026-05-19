"""One-off helper to split analysis/_core.py into submodules (dev maintenance)."""

from __future__ import annotations

import ast
import pathlib
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "src/spot_check/analysis/_core.py"
OUT = ROOT / "src/spot_check/analysis"

ASSIGN: dict[str, str] = {}


def _assign(prefix: str, module: str, names: list[str]) -> None:
    for n in names:
        ASSIGN[n] = module


_assign("pyvista_backend", "pyvista_backend", ["require_pyvista"])
_assign(
    "colors",
    "colors",
    [
        "_hex_to_rgb_u8",
        "_measured_alpha_u8_from_channel_weights",
        "measured_rgba_by_channel_weight",
    ],
)
_assign(
    "spatial",
    "spatial",
    [
        "_layer_xy_kdtrees_for_qa",
        "layer_nn_plan_xy_distances_and_expected_xyz",
        "layer_nn_plan_match_for_measured",
        "distances_measured_xy_to_layer_nn_plan_mm",
        "_layer_plan_mu_by_energy_layer",
        "layer_nn_local_spot_index_on_layer",
        "nominal_layer_energies_mev",
        "fit_ab_to_plan_xy",
        "_min_xy_dist_to_nominal_energy",
        "_plan_xy_from_optional_ab",
        "_ab_from_plan_xy",
        "_plan_xy_by_energy_layer",
        "_plan_xyz_by_energy_layer",
        "_build_layer_kdtrees",
        "_nearest_sqdist_sq_mm2_chunked",
        "_nearest_sqdist_sq_mm2_to_points",
        "_emit_sqdist_to_layers_mm2",
        "_kdtree_query_k1",
        "_layer_xy_kdtrees_2d",
        "_layer_nn_plan_targets",
        "_layer_nn_rms_mm",
        "_nearest_layer_index_from_plan_energy",
    ],
)
_assign(
    "plan_qa",
    "plan_qa",
    [
        "measured_rgba_by_plan_qa",
        "measured_rgba_by_plan_dose_qa",
        "plan_dose_qa_tier_counts",
        "_plan_qa_error_line_polylines",
        "plan_qa_pass_warn_fail_counts",
        "plan_qa_measured_spot_pass_warn_fail",
        "format_plan_qa_caption",
        "plan_dose_fraction_deviation_pp",
        "format_plan_dose_qa_caption",
    ],
)
_assign(
    "alignment",
    "alignment",
    [
        "_detector_align_coarse_angles_deg",
        "_rotation_matrix_2d",
        "_measured_xy_for_align",
        "_build_align_samples",
        "_subsample_align_indices",
        "_icp_rigid_layer_nn",
        "_detector_align_multistart_icp",
        "measured_plan_xy_from_row",
        "measured_row_with_plan_xy",
        "_kabsch_rigid_2d",
        "_apply_rigid_xy_to_measured_rows",
        "align_measured_to_plan_detector_xy",
        "format_detector_align_caption",
    ],
)
_assign(
    "layers",
    "layers",
    [
        "_layer_advance_plausible_vs_refill",
        "viterbi_monotone_layer_assign",
        "build_unified_advance_penalty_mm2",
        "energies_for_measured_time_layers",
        "_PlanImputeLookup",
        "_opt_float_cell",
        "_impute_plan_axis_fast",
        "_plan_impute_lookups_per_layer",
    ],
)
_assign(
    "measured",
    "measured",
    [
        "_channel_sum_na_from_row",
        "normalize_measured_spot_weight_mode",
        "measured_spot_weight_caption",
        "_sigma_cell_to_float",
        "_measured_row_with_sigma",
        "measured_charge_na_from_tuple",
        "measured_spot_weight_from_row",
        "_probe_csv_columns_for_measured_weights",
        "_gate_int_from_row",
        "_weighted_mean_masked",
        "_finalize_spot_channel_weighted",
        "_apply_gate_spot_aggregation",
        "_measured_tuple_for_spot_weighted_mean",
        "measured_spot_abc_from_csv",
    ],
)

VIZ_ASSIGN: dict[str, str] = {}
_assign(
    "viz_glyphs",
    "viz/glyphs",
    [
        "_plan_energy_bounds_mev",
        "_unit_sphere_glyph_template",
        "_disc_point_add_mesh_kwargs",
        "_instanced_axis_aligned_ellipsoids",
        "_plan_spot_fwhm_glyph_mesh",
        "_measured_spot_sigma_glyph_mesh",
    ],
)
_assign(
    "viz_data",
    "viz/data",
    [
        "prepare_comparison_3d_data",
        "_energy_slice_mask",
        "_nominal_layer_index_band_mev",
    ],
)
_assign(
    "viz_embed",
    "viz/embed",
    [
        "_vtk_rendering_tk_dll_present",
        "_stop_tk_vtk_event_pump",
        "_ensure_pyvista_iren_initialized",
        "_start_tk_vtk_event_pump",
        "_show_tk_vtk_fallback_panel",
        "idle_slice_band_controls",
        "_wire_slice_band_controls",
        "_embed_pyvista_plotter_in_tk",
        "_clear_qt_layout_items",
        "_embed_pyvista_plotter_in_qt",
        "apply_comparison_3d_camera_view",
        "idle_slice_band_controls_qt",
        "_wire_slice_band_controls_qt",
    ],
)
_assign("viz_plotter", "viz/plotter", ["show_comparison_3d_pyvista"])

for k, v in list(ASSIGN.items()):
    if v.startswith("viz/"):
        VIZ_ASSIGN[k] = v
        del ASSIGN[k]

MODULE_EXTRA: dict[str, str] = {
    "pyvista_backend": textwrap.dedent(
        """
        _pyvista_import_error: ImportError | None = None

        try:
            import pyvista as pv
        except ImportError as _pv_exc:  # pragma: no cover
            pv = None  # type: ignore[assignment]
            _pyvista_import_error = _pv_exc
        """
    ),
    "alignment": textwrap.dedent(
        """
        _DETECTOR_ALIGN_ICP_MAX_ITER = 25
        _DETECTOR_ALIGN_ICP_TOL_MM = 0.05
        _DETECTOR_ALIGN_COARSE_ANGLES_DEG: tuple[int, ...] = tuple(range(0, 360, 15))
        """
    ),
    "viz/glyphs": "_GLYPH_UNIT_SPHERE: dict[tuple[int, int], Any] = {}\n",
    "viz/embed": '_VTK_TK_PUMP: dict[str, Any] = {"after_id": None, "plotter": None}\n',
}

MODULE_IMPORTS: dict[str, str] = {
    "colors": "from spot_check.analysis._imports import *  # noqa: F403\n",
    "spatial": "from spot_check.analysis._imports import *  # noqa: F403\n",
    "plan_qa": textwrap.dedent(
        """
        from spot_check.analysis._imports import *  # noqa: F403
        from spot_check.analysis.colors import _hex_to_rgb_u8
        from spot_check.analysis.measured import measured_charge_na_from_tuple
        from spot_check.analysis.spatial import (
            _layer_plan_mu_by_energy_layer,
            distances_measured_xy_to_layer_nn_plan_mm,
            layer_nn_local_spot_index_on_layer,
            layer_nn_plan_xy_distances_and_expected_xyz,
            nominal_layer_energies_mev,
        )
        """
    ).strip()
    + "\n",
    "alignment": textwrap.dedent(
        """
        from spot_check.analysis._imports import *  # noqa: F403
        from spot_check.analysis.spatial import (
            _layer_nn_plan_targets,
            _layer_nn_rms_mm,
            _layer_xy_kdtrees_2d,
            fit_ab_to_plan_xy,
            layer_nn_plan_xy_distances_and_expected_xyz,
            nominal_layer_energies_mev,
        )
        """
    ).strip()
    + "\n",
    "layers": textwrap.dedent(
        """
        from spot_check.analysis._imports import *  # noqa: F403
        from spot_check.analysis.spatial import (
            _build_layer_kdtrees,
            _emit_sqdist_to_layers_mm2,
            _min_xy_dist_to_nominal_energy,
            _nearest_layer_index_from_plan_energy,
            _plan_xy_by_energy_layer,
            _plan_xyz_by_energy_layer,
            fit_ab_to_plan_xy,
            nominal_layer_energies_mev,
        )
        """
    ).strip()
    + "\n",
    "measured": textwrap.dedent(
        """
        from spot_check.analysis._imports import *  # noqa: F403
        from spot_check.analysis.layers import (
            _PlanImputeLookup,
            _impute_plan_axis_fast,
            _layer_advance_plausible_vs_refill,
            _opt_float_cell,
            _plan_impute_lookups_per_layer,
            build_unified_advance_penalty_mm2,
            energies_for_measured_time_layers,
            viterbi_monotone_layer_assign,
        )
        from spot_check.analysis.spatial import (
            _ab_from_plan_xy,
            _plan_xy_from_optional_ab,
            fit_ab_to_plan_xy,
            nominal_layer_energies_mev,
        )
        """
    ).strip()
    + "\n",
    "viz/glyphs": textwrap.dedent(
        """
        from spot_check.analysis._imports import *  # noqa: F403
        from spot_check.analysis.pyvista_backend import pv, require_pyvista
        """
    ).strip()
    + "\n",
    "viz/data": textwrap.dedent(
        """
        from spot_check.analysis._imports import *  # noqa: F403
        from spot_check.analysis.spatial import fit_ab_to_plan_xy
        from spot_check.analysis.viz.glyphs import _plan_energy_bounds_mev
        """
    ).strip()
    + "\n",
    "viz/embed": textwrap.dedent(
        """
        from spot_check.analysis._imports import *  # noqa: F403
        from spot_check.analysis.pyvista_backend import require_pyvista
        from spot_check.analysis.viz.data import _nominal_layer_index_band_mev
        """
    ).strip()
    + "\n",
    "viz/plotter": textwrap.dedent(
        """
        from spot_check.analysis._imports import *  # noqa: F403
        from spot_check.analysis.colors import (
            _hex_to_rgb_u8,
            _measured_alpha_u8_from_channel_weights,
            measured_rgba_by_channel_weight,
        )
        from spot_check.analysis.plan_qa import measured_rgba_by_plan_dose_qa, measured_rgba_by_plan_qa
        from spot_check.analysis.measured import measured_spot_weight_caption
        from spot_check.analysis.plan_qa import (
            _plan_qa_error_line_polylines,
            format_plan_dose_qa_caption,
            format_plan_qa_caption,
            plan_dose_fraction_deviation_pp,
            plan_dose_qa_tier_counts,
            plan_qa_pass_warn_fail_counts,
        )
        from spot_check.analysis.pyvista_backend import pv, require_pyvista
        from spot_check.analysis.spatial import (
            layer_nn_plan_xy_distances_and_expected_xyz,
            nominal_layer_energies_mev,
        )
        from spot_check.analysis.viz.data import (
            _energy_slice_mask,
            _nominal_layer_index_band_mev,
            prepare_comparison_3d_data,
        )
        from spot_check.analysis.viz.embed import (
            _embed_pyvista_plotter_in_qt,
            _embed_pyvista_plotter_in_tk,
            _ensure_pyvista_iren_initialized,
            _show_tk_vtk_fallback_panel,
            _start_tk_vtk_event_pump,
            _stop_tk_vtk_event_pump,
            _vtk_rendering_tk_dll_present,
            _wire_slice_band_controls,
            _wire_slice_band_controls_qt,
            idle_slice_band_controls,
            idle_slice_band_controls_qt,
        )
        from spot_check.analysis.viz.glyphs import (
            _disc_point_add_mesh_kwargs,
            _measured_spot_sigma_glyph_mesh,
            _plan_spot_fwhm_glyph_mesh,
        )
        """
    ).strip()
    + "\n",
}

HEADER_TMPL = '''\
"""{doc}."""

from __future__ import annotations

{imports}
{extra}
'''


def _module_path(mod: str) -> pathlib.Path:
    if mod.startswith("viz/"):
        return OUT / f"{mod}.py"
    return OUT / f"{mod}.py"


def main() -> None:
    src = CORE.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)

    # Skip scipy try block (moved to _imports)
    skip_lines = {1761, 1762, 1763, 1764}

    modules: dict[str, list[str]] = {}
    all_names: list[str] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        if node.lineno in skip_lines:
            continue
        mod = ASSIGN.get(node.name) or VIZ_ASSIGN.get(node.name)
        if mod is None:
            raise SystemExit(f"unassigned: {node.name} at line {node.lineno}")
        chunk = "".join(lines[node.lineno - 1 : node.end_lineno])
        modules.setdefault(mod, []).append(chunk)
        all_names.append(node.name)

    (OUT / "viz").mkdir(parents=True, exist_ok=True)

    for mod, chunks in sorted(modules.items()):
        path = _module_path(mod)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = mod.split("/")[-1].replace("_", " ").title()
        imports = MODULE_IMPORTS.get(mod, "from spot_check.analysis._imports import *  # noqa: F403\n")
        extra = MODULE_EXTRA.get(mod, "")
        body = HEADER_TMPL.format(doc=doc, imports=imports, extra=extra) + "\n".join(chunks)
        path.write_text(body, encoding="utf-8")
        print("wrote", path, len(chunks), "defs")

    # viz package __init__
    viz_init = '''\
"""3D comparison visualization (PyVista)."""

from spot_check.analysis.viz.data import prepare_comparison_3d_data
from spot_check.analysis.viz.embed import (
    apply_comparison_3d_camera_view,
    idle_slice_band_controls,
    idle_slice_band_controls_qt,
)
from spot_check.analysis.viz.plotter import show_comparison_3d_pyvista

__all__ = [
    "apply_comparison_3d_camera_view",
    "idle_slice_band_controls",
    "idle_slice_band_controls_qt",
    "prepare_comparison_3d_data",
    "show_comparison_3d_pyvista",
]
'''
    (OUT / "viz" / "__init__.py").write_text(viz_init, encoding="utf-8")

    # Facade _core.py
    facade_parts = [
        '"""Backward-compatible re-exports for spot_check.analysis._core."""\n',
        "from __future__ import annotations\n",
        "from spot_check.analysis.pyvista_backend import pv, require_pyvista\n",
        "from spot_check.analysis._imports import FOLDER, logger\n",
    ]
    mod_files = [
        "colors",
        "spatial",
        "plan_qa",
        "alignment",
        "layers",
        "measured",
    ]
    for mf in mod_files:
        facade_parts.append(f"from spot_check.analysis.{mf} import *  # noqa: F403\n")
    facade_parts.append("from spot_check.analysis.viz import *  # noqa: F403\n")
    facade_parts.append("from spot_check.analysis.viz.glyphs import *  # noqa: F403\n")
    facade_parts.append("from spot_check.analysis.viz.embed import *  # noqa: F403\n")
    facade_parts.append("from spot_check.analysis.viz.data import *  # noqa: F403\n")

    facade_path = OUT / "_core.py"
    # Backup original on first run only
    backup = OUT / "_core_monolith.py.bak"
    if not backup.exists():
        backup.write_text(src, encoding="utf-8")
    facade_path.write_text("".join(facade_parts), encoding="utf-8")
    print("wrote facade", facade_path)

    # Move docstring to __init__
    doc = ast.get_docstring(tree) or ""
    init_path = OUT / "__init__.py"
    init_src = init_path.read_text(encoding="utf-8")
    if 'RT Ion plan vs acquisition' not in init_src:
        new_init = f'"""\n{doc}\n"""\n\n' + init_src.lstrip()
        init_path.write_text(new_init, encoding="utf-8")
        print("updated", init_path)


if __name__ == "__main__":
    main()
