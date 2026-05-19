"""Alignment."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.spatial import (
    _layer_nn_plan_targets,
    _layer_nn_rms_mm,
    _layer_xy_kdtrees_2d,
    nominal_layer_energies_mev,
)

_DETECTOR_ALIGN_ICP_MAX_ITER = 25
_DETECTOR_ALIGN_ICP_TOL_MM = 0.05
_DETECTOR_ALIGN_COARSE_ANGLES_DEG: tuple[int, ...] = tuple(range(0, 360, 15))

def _detector_align_coarse_angles_deg(n_samples: int) -> tuple[int, ...]:
    """Fewer rotation seeds for large acquisitions (ICP cost scales with sample count)."""
    n = int(n_samples)
    if n > 100_000:
        return (0, 90, 180, 270)
    if n > 50_000:
        return tuple(range(0, 360, 45))
    if n > 10_000:
        return tuple(range(0, 360, 30))
    return _DETECTOR_ALIGN_COARSE_ANGLES_DEG

def _rotation_matrix_2d(theta_deg: float) -> np.ndarray:
    th = math.radians(float(theta_deg))
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], dtype=np.float64)

def _measured_xy_for_align(
    row: tuple[float, ...],
    *,
    a_is_x: bool,
    swap_ab_axes: bool,
) -> tuple[float, float]:
    """Plan-frame XY used for alignment; ``swap_ab_axes`` maps Fit A→X and Fit B→Y."""
    a, b = float(row[0]), float(row[1])
    if swap_ab_axes:
        return a, b
    return measured_plan_xy_from_row(row, a_is_x=a_is_x)

def _build_align_samples(
    measured_rows: list[tuple[float, ...]],
    layer_xy: list[np.ndarray],
    hi: int,
    *,
    a_is_x: bool,
    swap_ab_axes: bool,
) -> tuple[np.ndarray, np.ndarray]:
    m_acc: list[list[float]] = []
    li_acc: list[int] = []
    for tup in measured_rows:
        li = int(round(float(tup[2])))
        if li < 0:
            li = 0
        elif li > hi:
            li = hi
        arr = np.asarray(layer_xy[li], dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] == 0:
            continue
        mx, my = _measured_xy_for_align(tup, a_is_x=a_is_x, swap_ab_axes=swap_ab_axes)
        m_acc.append([mx, my])
        li_acc.append(li)
    if not m_acc:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.intp)
    return np.asarray(m_acc, dtype=np.float64), np.asarray(li_acc, dtype=np.intp)

def _subsample_align_indices(n: int, max_n: int) -> np.ndarray:
    """Deterministic stride subsample for alignment fit (full transform applied to all rows)."""
    n = int(n)
    cap = int(max_n)
    if n <= 0:
        return np.zeros(0, dtype=np.intp)
    if n <= cap:
        return np.arange(n, dtype=np.intp)
    step = int(math.ceil(n / cap))
    return np.arange(0, n, step, dtype=np.intp)

def _icp_rigid_layer_nn(
    meas_xy: np.ndarray,
    layer_idx: np.ndarray,
    layer_xy: list[np.ndarray],
    *,
    layer_trees: list[Any | None] | None = None,
    r_init: np.ndarray | None = None,
    t_init: np.ndarray | None = None,
    max_iter: int = _DETECTOR_ALIGN_ICP_MAX_ITER,
    tol_mm: float = _DETECTOR_ALIGN_ICP_TOL_MM,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Iterative per-layer NN + Kabsch; returns cumulative ``R``, ``t``, RMS before/after."""
    trees = _layer_xy_kdtrees_2d(layer_xy) if layer_trees is None else layer_trees
    r_acc = (
        np.eye(2, dtype=np.float64)
        if r_init is None
        else np.asarray(r_init, dtype=np.float64).reshape(2, 2).copy()
    )
    t_acc = (
        np.zeros(2, dtype=np.float64)
        if t_init is None
        else np.asarray(t_init, dtype=np.float64).reshape(2).copy()
    )
    rms_nn = float("nan")
    rms_res = float("inf")
    prev = float("inf")
    n_iter = 0
    for n_iter in range(1, int(max_iter) + 1):
        trans = (r_acc @ meas_xy.T).T + t_acc
        m_pairs, p_pairs = _layer_nn_plan_targets(trans, layer_idx, layer_xy, layer_trees=trees)
        if int(m_pairs.shape[0]) < 2:
            break
        diff0 = m_pairs - p_pairs
        rms_nn = float(np.sqrt(np.mean(np.sum(diff0 * diff0, axis=1))))
        r_step, t_step, _, rms_res = _kabsch_rigid_2d(m_pairs, p_pairs)
        t_acc = r_step @ t_acc + t_step
        r_acc = r_step @ r_acc
        if math.isfinite(prev) and abs(prev - rms_res) < float(tol_mm):
            break
        prev = rms_res
    return r_acc, t_acc, rms_nn, rms_res, n_iter

def _detector_align_multistart_icp(
    meas_xy: np.ndarray,
    layer_idx: np.ndarray,
    layer_xy: list[np.ndarray],
    *,
    layer_trees: list[Any | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """Try coarse rotation seeds; return best cumulative ``R``, ``t``, RMS, ICP iters."""
    if int(meas_xy.shape[0]) < 2:
        raise ValueError("detector alignment needs at least 2 measured rows with plan spots")
    trees = _layer_xy_kdtrees_2d(layer_xy) if layer_trees is None else layer_trees
    best_rms = float("inf")
    best: tuple[np.ndarray, np.ndarray, float, float, int] | None = None
    centroid = np.mean(meas_xy, axis=0)
    angle_seeds = _detector_align_coarse_angles_deg(int(meas_xy.shape[0]))
    for init_deg in angle_seeds:
        r_seed = _rotation_matrix_2d(init_deg)
        t_seed = centroid - r_seed @ centroid
        r_acc, t_acc, rms_nn, rms_res, n_iter = _icp_rigid_layer_nn(
            meas_xy,
            layer_idx,
            layer_xy,
            layer_trees=trees,
            r_init=r_seed,
            t_init=t_seed,
        )
        if not math.isfinite(rms_res) or int(meas_xy.shape[0]) < 2:
            continue
        holdout = _layer_nn_rms_mm(meas_xy, layer_idx, layer_xy, r_acc, t_acc, layer_trees=trees)
        if holdout < best_rms:
            best_rms = holdout
            best = (r_acc, t_acc, rms_nn, holdout, n_iter)
    if best is None:
        raise ValueError("detector ICP alignment failed for all rotation seeds")
    r_acc, t_acc, rms_nn, rms_res, n_iter = best
    return r_acc, t_acc, rms_nn, rms_res, n_iter, int(meas_xy.shape[0])

def measured_plan_xy_from_row(row: tuple[float, ...], *, a_is_x: bool) -> tuple[float, float]:
    """Plan-frame (X, Y) mm from stored fit A/B row (same convention as the 3D plot)."""
    a, b = float(row[0]), float(row[1])
    return (a, b) if a_is_x else (b, a)

def measured_row_with_plan_xy(
    row: tuple[float, ...],
    x: float,
    y: float,
    *,
    a_is_x: bool,
) -> tuple[float, ...]:
    tail = row[2:]
    if a_is_x:
        return (float(x), float(y), *tail)
    return (float(y), float(x), *tail)

def _kabsch_rigid_2d(
    meas: np.ndarray,
    plan: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (R, t, rms_nn, rms_residual). Shapes (n,2), (n,2)."""
    n = int(meas.shape[0])
    if n == 0:
        raise ValueError("no point pairs for detector alignment")
    if n == 1:
        r_mat = np.eye(2, dtype=np.float64)
        tvec = (plan[0] - meas[0]).astype(np.float64)
        rms_nn = float(np.linalg.norm(meas[0] - plan[0]))
        rms_res = 0.0
        return r_mat, tvec, rms_nn, rms_res
    c_m = meas.mean(axis=0)
    c_p = plan.mean(axis=0)
    diff0 = meas - plan
    rms_nn = float(np.sqrt(np.mean(np.sum(diff0 * diff0, axis=1))))
    m_c = meas - c_m
    p_c = plan - c_p
    h = m_c.T @ p_c
    u, _, vt = np.linalg.svd(h)
    r_mat = vt.T @ u.T
    if float(np.linalg.det(r_mat)) < 0.0:
        vt2 = vt.copy()
        vt2[1, :] *= -1.0
        r_mat = vt2.T @ u.T
    tvec = c_p - r_mat @ c_m
    aligned = (r_mat @ meas.T).T + tvec
    diff1 = aligned - plan
    rms_res = float(np.sqrt(np.mean(np.sum(diff1 * diff1, axis=1))))
    return r_mat, tvec, rms_nn, rms_res

def _apply_rigid_xy_to_measured_rows(
    measured_rows: list[tuple[float, ...]],
    r_mat: np.ndarray,
    tvec: np.ndarray,
    *,
    a_is_x: bool,
    swap_ab_axes: bool,
) -> list[tuple[float, ...]]:
    """Apply ``r_mat @ m + tvec`` to every row (vectorized XY; tuple tails preserved)."""
    n = len(measured_rows)
    if n == 0:
        return []
    a_col = np.fromiter((float(r[0]) for r in measured_rows), dtype=np.float64, count=n)
    b_col = np.fromiter((float(r[1]) for r in measured_rows), dtype=np.float64, count=n)
    if swap_ab_axes or a_is_x:
        m = np.column_stack([a_col, b_col])
    else:
        m = np.column_stack([b_col, a_col])
    w = (np.asarray(r_mat, dtype=np.float64) @ m.T).T + np.asarray(tvec, dtype=np.float64).reshape(
        1, 2
    )
    out: list[tuple[float, ...]] = []
    if a_is_x:
        for i, row in enumerate(measured_rows):
            out.append((float(w[i, 0]), float(w[i, 1]), *row[2:]))
    else:
        for i, row in enumerate(measured_rows):
            out.append((float(w[i, 1]), float(w[i, 0]), *row[2:]))
    return out

def align_measured_to_plan_detector_xy(
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool,
    max_fit_samples: int = DETECTOR_ALIGN_MAX_FIT_SAMPLES,
) -> tuple[list[tuple[float, ...]], DetectorRigidAlign2D]:
    """Per-layer NN + multi-start ICP rigid XY, apply to every measured row.

    Handles arbitrary in-plane rotation (including 90°+ detector placement) and optional
    Fit A↔B axis swap by trying both axis conventions and coarse rotation seeds (every
    15°), then refining with iterative closest-point + Kabsch on each row's assigned
    nominal layer. For large acquisitions, the rigid fit uses a deterministic subsample
    (``max_fit_samples``); the returned transform is applied to **all** rows.
    """
    if not planned_xyz or not measured_rows:
        raise ValueError("plan and measured rows are required for detector alignment")
    layer_e = nominal_layer_energies_mev(planned_xyz)
    if not layer_e:
        raise ValueError("plan has no nominal energy layers")
    layer_xy = _plan_xy_by_energy_layer(planned_xyz, layer_e)
    hi = len(layer_e) - 1
    layer_trees = _layer_xy_kdtrees_2d(layer_xy)

    best: (
        tuple[
            np.ndarray,
            np.ndarray,
            float,
            float,
            bool,
            int,
            int,
            int,
        ]
        | None
    ) = None
    best_rms = float("inf")
    for swap_ab in (False, True):
        meas_xy, layer_idx = _build_align_samples(
            measured_rows,
            layer_xy,
            hi,
            a_is_x=a_is_x,
            swap_ab_axes=swap_ab,
        )
        n_all = int(meas_xy.shape[0])
        if n_all < 2:
            continue
        fit_idx = _subsample_align_indices(n_all, int(max_fit_samples))
        meas_fit = meas_xy[fit_idx]
        layer_fit = layer_idx[fit_idx]
        try:
            r_mat, tvec, rms_nn, rms_res, n_iter, _n_fit = _detector_align_multistart_icp(
                meas_fit,
                layer_fit,
                layer_xy,
                layer_trees=layer_trees,
            )
        except ValueError:
            continue
        if rms_res < best_rms:
            best_rms = rms_res
            best = (r_mat, tvec, rms_nn, rms_res, swap_ab, n_iter, n_all, int(fit_idx.size))

    if best is None:
        raise ValueError(
            "detector alignment needs at least 2 measured rows with plan spots on their layer "
            "(check layer assignment, plan, and detector orientation)"
        )

    r_mat, tvec, rms_nn, rms_res, swap_ab, n_iter, n_all, n_fit = best
    theta = float(math.degrees(math.atan2(float(r_mat[1, 0]), float(r_mat[0, 0]))))
    info = DetectorRigidAlign2D(
        theta_deg=theta,
        tx_mm=float(tvec[0]),
        ty_mm=float(tvec[1]),
        rms_nn_mm=rms_nn,
        rms_residual_mm=rms_res,
        n_pairs=n_all,
        ab_axes_swapped=bool(swap_ab),
        icp_iterations=int(n_iter),
        n_pairs_fit=int(n_fit),
    )
    out_rows = _apply_rigid_xy_to_measured_rows(
        measured_rows,
        r_mat,
        tvec,
        a_is_x=a_is_x,
        swap_ab_axes=swap_ab,
    )
    return out_rows, info

def format_detector_align_caption(info: DetectorRigidAlign2D) -> str:
    swap_note = "; Fit A↔B swapped for search" if info.ab_axes_swapped else ""
    fit_note = ""
    if info.n_pairs_fit > 0 and info.n_pairs_fit < info.n_pairs:
        fit_note = f", fit n={info.n_pairs_fit}/{info.n_pairs}"
    return (
        f"Detector align: θ={info.theta_deg:.5g}° CCW, t=({info.tx_mm:.5g}, {info.ty_mm:.5g}) mm, "
        f"RMS after={info.rms_residual_mm:.5g} mm (NN RMS before={info.rms_nn_mm:.5g} mm, "
        f"n={info.n_pairs}{fit_note}, ICP={info.icp_iterations}{swap_note})."
    )
