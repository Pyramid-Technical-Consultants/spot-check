"""Glyphs."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.colors import _hex_to_rgb_u8
from spot_check.analysis.pyvista_backend import pv, require_pyvista

_GLYPH_UNIT_SPHERE: dict[tuple[int, int], Any] = {}

def _plan_energy_bounds_mev(planned_xyz: list[tuple[float, float, float]]) -> tuple[float, float]:
    zs = [z for _, _, z in planned_xyz]
    if not zs:
        return 0.0, 0.0
    return max(zs), min(zs)

def _unit_sphere_glyph_template(phi_resolution: int, theta_resolution: int) -> Any:
    require_pyvista()
    key = (int(phi_resolution), int(theta_resolution))
    tpl = _GLYPH_UNIT_SPHERE.get(key)
    if tpl is None:
        tpl = pv.Sphere(
            radius=1.0,
            phi_resolution=int(phi_resolution),
            theta_resolution=int(theta_resolution),
        )
        _GLYPH_UNIT_SPHERE[key] = tpl
    return tpl

def _disc_point_add_mesh_kwargs(*, point_size: float) -> dict[str, Any]:
    """Sharp screen-space circular discs (VTK sphere impostors), flat and unlit."""
    return {
        "render_points_as_spheres": True,
        "lighting": False,
        "ambient": 1.0,
        "diffuse": 0.0,
        "specular": 0.0,
        "smooth_shading": False,
        "point_size": float(point_size),
    }

def _instanced_axis_aligned_ellipsoids(
    centers: np.ndarray,
    semiaxes_xyz: np.ndarray,
    *,
    phi_resolution: int = 14,
    theta_resolution: int = 14,
) -> Any:
    """Axis-aligned ellipsoids in scene units (mm): per-row X/Y/Z semiaxes.

    ``pyvista.PolyData.glyph(..., scale=<3-component array>, orient=False)`` does not
    apply per-axis scaling reliably on VTK 9.6 (glyphs stay near unit size), so we
    instance a unit sphere template with explicit point transforms.
    """
    require_pyvista()
    centers_u = np.asarray(centers, dtype=np.float64, order="C")
    semi_u = np.asarray(semiaxes_xyz, dtype=np.float64, order="C")
    n = int(centers_u.shape[0])
    if n == 0:
        return pv.PolyData(np.zeros((0, 3), dtype=np.float64))
    if centers_u.shape != (n, 3):
        raise ValueError("centers must have shape (n, 3)")
    if semi_u.shape != (n, 3):
        raise ValueError("semiaxes_xyz must have shape (n, 3)")

    tpl = _unit_sphere_glyph_template(phi_resolution, theta_resolution)
    tpl_pts = np.asarray(tpl.points, dtype=np.float64)
    m = int(tpl_pts.shape[0])
    tpl_faces = np.asarray(tpl.faces, dtype=np.int64).reshape(tpl.n_cells, 4)
    if not bool(np.all(tpl_faces[:, 0] == 3)):
        raise RuntimeError("internal: unit-sphere template must be all triangles")

    pts = (tpl_pts[np.newaxis, :, :] * semi_u[:, np.newaxis, :]).reshape(-1, 3)
    pts += np.repeat(centers_u, m, axis=0)

    tri = tpl_faces[:, 1:4]
    off = (np.arange(n, dtype=np.int64) * m)[:, np.newaxis, np.newaxis]
    inst_tris = (tri[np.newaxis, :, :] + off).reshape(-1, 3)
    face_arr = np.empty((inst_tris.shape[0], 4), dtype=np.int64)
    face_arr[:, 0] = 3
    face_arr[:, 1:4] = inst_tris
    return pv.PolyData(pts, face_arr.ravel())

def _plan_spot_visibility_rgba(visible_mask: np.ndarray) -> np.ndarray:
    """RGBA for plan spots; alpha=0 keeps geometry in bounds while hiding out-of-band spots."""
    vis = np.asarray(visible_mask, dtype=bool).reshape(-1)
    r, g, b = _hex_to_rgb_u8(_PLAN_COLOR_3D)
    n = int(vis.shape[0])
    rgba = np.empty((n, 4), dtype=np.uint8)
    rgba[:, 0] = r
    rgba[:, 1] = g
    rgba[:, 2] = b
    a_vis = int(round(0.45 * 255))
    rgba[:, 3] = np.where(vis, np.uint8(a_vis), np.uint8(0))
    return rgba

def _plan_spot_point_mesh(plan_pts: np.ndarray, *, visible_mask: np.ndarray | None = None) -> Any:
    """Plan spot point cloud with per-spot visibility via RGBA alpha."""
    require_pyvista()
    n = int(plan_pts.shape[0])
    m = pv.PolyData(plan_pts)
    if n == 0:
        return m
    if visible_mask is None:
        vis = np.ones(n, dtype=bool)
    else:
        vis = np.asarray(visible_mask, dtype=bool).reshape(-1)
        if int(vis.shape[0]) != n:
            raise ValueError("visible_mask length must match plan_pts row count")
    m["rgba"] = _plan_spot_visibility_rgba(vis)
    return m

def _plan_spot_cross_mesh(
    plan_pts: np.ndarray,
    *,
    spot_mask: np.ndarray,
    visible_mask: np.ndarray | None = None,
    color_hex: str = _PLAN_QA_FAIL_HEX,
) -> Any:
    """XY cross marker at selected plan spots (two orthogonal line segments each)."""
    require_pyvista()
    n = int(plan_pts.shape[0])
    sm = np.asarray(spot_mask, dtype=bool).reshape(-1)
    if int(sm.shape[0]) != n:
        raise ValueError("spot_mask length must match plan_pts row count")
    idx = np.flatnonzero(sm)
    if idx.size == 0:
        return pv.PolyData(np.zeros((0, 3), dtype=np.float64))
    centers = np.asarray(plan_pts[idx], dtype=np.float64)
    m = int(idx.size)
    arm = float(_PLAN_MISSING_CROSS_HALF_ARM_MM)
    pts = np.empty((m * 4, 3), dtype=np.float64)
    for j, c in enumerate(centers):
        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        base = j * 4
        pts[base + 0] = (cx - arm, cy, cz)
        pts[base + 1] = (cx + arm, cy, cz)
        pts[base + 2] = (cx, cy - arm, cz)
        pts[base + 3] = (cx, cy + arm, cz)
    line_arr: list[int] = []
    for j in range(m):
        base_pt = j * 4
        line_arr.extend([2, base_pt, base_pt + 1])
        line_arr.extend([2, base_pt + 2, base_pt + 3])
    poly = pv.PolyData(pts, lines=np.asarray(line_arr, dtype=np.int64))
    r, g, b = _hex_to_rgb_u8(color_hex)
    a_vis = int(round(0.85 * 255))
    rgba_pts = np.empty((m * 4, 4), dtype=np.uint8)
    rgba_pts[:, 0] = np.uint8(r)
    rgba_pts[:, 1] = np.uint8(g)
    rgba_pts[:, 2] = np.uint8(b)
    if visible_mask is None:
        rgba_pts[:, 3] = np.uint8(a_vis)
    else:
        vis = np.asarray(visible_mask, dtype=bool).reshape(-1)
        if int(vis.shape[0]) != n:
            raise ValueError("visible_mask length must match plan_pts row count")
        spot_vis = vis[idx]
        for j, v in enumerate(spot_vis):
            rgba_pts[j * 4 : (j + 1) * 4, 3] = np.uint8(a_vis if v else 0)
    poly["rgba"] = rgba_pts
    return poly

def _plan_spot_fwhm_glyph_mesh(
    plan_pts: np.ndarray,
    fwhm_xy_mm: np.ndarray,
    *,
    visible_mask: np.ndarray | None = None,
) -> Any:
    """At each plan point, an axis-aligned ellipsoid with X/Y semiaxis = FWHM/2 (mm) and thin Z."""
    require_pyvista()
    n = int(plan_pts.shape[0])
    if fwhm_xy_mm.shape != (n, 2):
        raise ValueError("plan_fwhm_xy_mm must have shape (n_plan, 2)")
    fx = fwhm_xy_mm[:, 0].astype(np.float64, copy=False)
    fy = fwhm_xy_mm[:, 1].astype(np.float64, copy=False)
    good = np.isfinite(fx) & np.isfinite(fy) & (fx > 0.0) & (fy > 0.0)
    med_x = float(np.nanmedian(fx[good])) if np.any(good) else _DEFAULT_PLAN_FWHM_MM_WHEN_MISSING
    med_y = float(np.nanmedian(fy[good])) if np.any(good) else _DEFAULT_PLAN_FWHM_MM_WHEN_MISSING
    if not math.isfinite(med_x) or med_x <= 0.0:
        med_x = _DEFAULT_PLAN_FWHM_MM_WHEN_MISSING
    if not math.isfinite(med_y) or med_y <= 0.0:
        med_y = _DEFAULT_PLAN_FWHM_MM_WHEN_MISSING
    sx = np.where(good, fx, med_x) * 0.5
    sy = np.where(good, fy, med_y) * 0.5
    zptp = float(np.ptp(plan_pts[:, 2])) if n else 1.0
    if not math.isfinite(zptp) or zptp <= 0.0:
        zptp = 1.0
    sz = max(zptp * _PLAN_FWHM_GLYPH_Z_SPAN_FRAC, 1e-9)
    scal = np.column_stack([sx, sy, np.full(n, sz, dtype=np.float64)])
    g = _instanced_axis_aligned_ellipsoids(plan_pts, scal)
    if visible_mask is not None:
        vis = np.asarray(visible_mask, dtype=bool).reshape(-1)
        if int(vis.shape[0]) != n:
            raise ValueError("visible_mask length must match plan_pts row count")
        tpl = _unit_sphere_glyph_template(14, 14)
        n_g = int(tpl.n_points)
        if int(g.n_points) == n_g * n:
            g["rgba"] = np.repeat(_plan_spot_visibility_rgba(vis), n_g, axis=0)
    return g

def _measured_spot_sigma_glyph_mesh(
    meas_pts: np.ndarray,
    sigma_xy_mm: np.ndarray,
    *,
    sigma_scale: float = MEASURED_SIGMA_GLYPH_SCALE_DEFAULT,
    rgba: np.ndarray | None = None,
) -> Any:
    """Per measured point, an axis-aligned ellipsoid: X/Y semiaxis = σ×scale (mm), diameter =
    2×scale×σ; thin Z."""
    require_pyvista()
    n = int(meas_pts.shape[0])
    if n == 0:
        return pv.PolyData(np.zeros((0, 3), dtype=np.float64))
    sig = np.asarray(sigma_xy_mm, dtype=np.float64).reshape(n, 2)
    if sig.shape[0] != n:
        raise ValueError("sigma_xy_mm row count must match meas_pts")
    sx_raw = sig[:, 0]
    sy_raw = sig[:, 1]
    good = np.isfinite(sx_raw) & np.isfinite(sy_raw) & (sx_raw > 0.0) & (sy_raw > 0.0)
    fb = float(MEASURED_SIGMA_GLYPH_FALLBACK_MM)
    med_x = float(np.nanmedian(sx_raw[good])) if np.any(good) else fb
    med_y = float(np.nanmedian(sy_raw[good])) if np.any(good) else fb
    if not math.isfinite(med_x) or med_x <= 0.0:
        med_x = fb
    if not math.isfinite(med_y) or med_y <= 0.0:
        med_y = fb
    scale = float(sigma_scale)
    sx = np.where(good, sx_raw, med_x).astype(np.float64, copy=False) * scale
    sy = np.where(good, sy_raw, med_y).astype(np.float64, copy=False) * scale
    sx = np.clip(sx, float(MEASURED_SIGMA_GLYPH_MIN_MM), float(MEASURED_SIGMA_GLYPH_MAX_MM))
    sy = np.clip(sy, float(MEASURED_SIGMA_GLYPH_MIN_MM), float(MEASURED_SIGMA_GLYPH_MAX_MM))
    zptp = float(np.ptp(meas_pts[:, 2]))
    if not math.isfinite(zptp) or zptp <= 0.0:
        zptp = 1.0
    sz = max(zptp * float(_PLAN_FWHM_GLYPH_Z_SPAN_FRAC), 1e-9)
    scal = np.column_stack([sx, sy, np.full(n, sz, dtype=np.float64)])
    tpl = _unit_sphere_glyph_template(14, 14)
    n_g = int(tpl.n_points)
    centers = np.asarray(meas_pts, dtype=np.float64, order="C")
    g = _instanced_axis_aligned_ellipsoids(centers, scal)
    if rgba is not None:
        rgba_u8 = np.asarray(rgba, dtype=np.uint8).reshape(-1, 4)
        if int(rgba_u8.shape[0]) == n:
            if int(g.n_points) == n_g * n:
                g["rgba"] = np.repeat(rgba_u8, n_g, axis=0)
            else:
                logger.warning(
                    "Measured σ glyph RGBA expansion skipped (point count mismatch: %s vs %s×%s)",
                    g.n_points,
                    n,
                    n_g,
                )
    return g
