"""Spatial."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403


def _plan_xy_from_optional_ab(
    a_opt: float | None,
    b_opt: float | None,
    *,
    a_is_x: bool,
) -> tuple[float | None, float | None, int]:
    """Map optional raw fit A/B to plan (X,Y). Returns partial: 0=both, 1=raw A missing, 2=raw B
    missing, -1=both missing."""
    if a_opt is not None and b_opt is not None:
        mx, my = fit_ab_to_plan_xy(a_opt, b_opt, a_is_x=a_is_x)
        return mx, my, 0
    if a_opt is None and b_opt is None:
        return None, None, -1
    if a_is_x:
        if a_opt is None:
            return None, float(b_opt), 1
        return float(a_opt), None, 2
    if a_opt is None:
        return float(b_opt), None, 1
    return None, float(a_opt), 2

def _ab_from_plan_xy(mx: float, my: float, *, a_is_x: bool) -> tuple[float, float]:
    return (mx, my) if a_is_x else (my, mx)

def _nearest_layer_index_from_plan_energy(z: float, layer_e: list[float]) -> int:
    """Index of closest nominal layer energy to plan spot z (MeV)."""
    if not layer_e:
        return 0
    zf = float(z)
    best_i = 0
    best_d = float("inf")
    for i, e in enumerate(layer_e):
        d = abs(zf - float(e))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i

def _layer_xy_kdtrees_for_qa(
    layer_xyz: list[np.ndarray],
) -> list[Any | None]:
    """2D cKDTree per nominal layer (plan X/Y mm) for NN queries; None when scipy missing or
    empty."""
    trees: list[Any | None] = []
    for arr in layer_xyz:
        a2 = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
        if a2.shape[0] == 0 or _cKDTree is None:
            trees.append(None)
            continue
        xy = a2[:, 0:2]
        trees.append(_cKDTree(xy))
    return trees

def layer_nn_plan_xy_distances_and_expected_xyz(
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """XY distance (mm) to nearest plan spot on each row's nominal layer and that spot's plan
    (x,y,energy)."""
    if not planned_xyz:
        raise ValueError("plan is empty")
    layer_e = nominal_layer_energies_mev(planned_xyz)
    if not layer_e:
        raise ValueError("plan has no nominal energy layers")
    layer_xyz = _plan_xyz_by_energy_layer(planned_xyz, layer_e)
    hi = len(layer_e) - 1
    n = len(measured_rows)
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    li_raw = np.rint(np.asarray([float(t[2]) for t in measured_rows], dtype=np.float64)).astype(
        np.intp, copy=False
    )
    np.clip(li_raw, 0, hi, out=li_raw)

    if a_is_x:
        mx = np.fromiter((float(t[0]) for t in measured_rows), dtype=np.float64, count=n)
        my = np.fromiter((float(t[1]) for t in measured_rows), dtype=np.float64, count=n)
    else:
        mx = np.fromiter((float(t[1]) for t in measured_rows), dtype=np.float64, count=n)
        my = np.fromiter((float(t[0]) for t in measured_rows), dtype=np.float64, count=n)
    meas_xy = np.column_stack([mx, my])

    out_d = np.full(n, np.inf, dtype=np.float64)
    out_xyz = np.full((n, 3), np.nan, dtype=np.float64)
    trees = _layer_xy_kdtrees_for_qa(layer_xyz)

    for ell in range(len(layer_e)):
        mask = li_raw == ell
        if not np.any(mask):
            continue
        arr = np.asarray(layer_xyz[ell], dtype=np.float64).reshape(-1, 3)
        if arr.shape[0] == 0:
            continue
        q = meas_xy[mask]
        tree = trees[ell] if ell < len(trees) else None
        if tree is not None:
            dist, idx = _kdtree_query_k1(tree, q)
            dist = np.asarray(dist, dtype=np.float64).reshape(-1)
            idx = np.asarray(idx, dtype=np.intp).reshape(-1)
            out_d[mask] = dist
            out_xyz[mask] = arr[idx]
        else:
            xy_layer = arr[:, 0:2]
            d2 = np.sum((xy_layer[None, :, :] - q[:, None, :]) ** 2, axis=2)
            j = np.argmin(d2, axis=1)
            out_d[mask] = np.sqrt(d2[np.arange(q.shape[0], dtype=np.intp), j])
            out_xyz[mask] = arr[j]
    return out_d, out_xyz

def layer_nn_plan_match_for_measured(
    planned_xyz: list[tuple[float, float, float]],
    plan_mu: np.ndarray | None,
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest plan spot on each row's layer: distance (mm), expected XYZ, meterset weight (MU)."""
    dist, exp_xyz = layer_nn_plan_xy_distances_and_expected_xyz(
        planned_xyz, measured_rows, a_is_x=a_is_x
    )
    n = int(dist.shape[0])
    exp_mu = np.full(n, np.nan, dtype=np.float64)
    if plan_mu is None or len(plan_mu) != len(planned_xyz):
        return dist, exp_xyz, exp_mu

    layer_e = nominal_layer_energies_mev(planned_xyz)
    if not layer_e:
        return dist, exp_xyz, exp_mu

    mu_buckets: list[list[float]] = [[] for _ in layer_e]
    for i, (px, py, pe) in enumerate(planned_xyz):
        pf = float(pe)
        mu_v = float(plan_mu[i])
        for k, e in enumerate(layer_e):
            if abs(pf - float(e)) <= 1e-4:
                mu_buckets[k].append(mu_v)
                break

    li_raw = np.rint(np.asarray([float(t[2]) for t in measured_rows], dtype=np.float64)).astype(
        np.intp, copy=False
    )
    hi = len(layer_e) - 1
    np.clip(li_raw, 0, hi, out=li_raw)

    if a_is_x:
        mx = np.fromiter((float(t[0]) for t in measured_rows), dtype=np.float64, count=n)
        my = np.fromiter((float(t[1]) for t in measured_rows), dtype=np.float64, count=n)
    else:
        mx = np.fromiter((float(t[1]) for t in measured_rows), dtype=np.float64, count=n)
        my = np.fromiter((float(t[0]) for t in measured_rows), dtype=np.float64, count=n)
    meas_xy = np.column_stack([mx, my])

    layer_xyz = _plan_xyz_by_energy_layer(planned_xyz, layer_e)
    trees = _layer_xy_kdtrees_for_qa(layer_xyz)

    for ell in range(len(layer_e)):
        mask = li_raw == ell
        if not np.any(mask):
            continue
        arr = np.asarray(layer_xyz[ell], dtype=np.float64).reshape(-1, 3)
        mu_arr = np.asarray(mu_buckets[ell], dtype=np.float64).reshape(-1)
        if arr.shape[0] == 0 or mu_arr.shape[0] != arr.shape[0]:
            continue
        q = meas_xy[mask]
        tree = trees[ell] if ell < len(trees) else None
        if tree is not None:
            _dist, idx = _kdtree_query_k1(tree, q)
            idx = np.asarray(idx, dtype=np.intp).reshape(-1)
            exp_mu[mask] = mu_arr[idx]
        else:
            xy_layer = arr[:, 0:2]
            d2 = np.sum((xy_layer[None, :, :] - q[:, None, :]) ** 2, axis=2)
            j = np.argmin(d2, axis=1)
            exp_mu[mask] = mu_arr[j]
    return dist, exp_xyz, exp_mu

def distances_measured_xy_to_layer_nn_plan_mm(
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool,
) -> np.ndarray:
    """Plan XY distance (mm) from each measured row to the nearest plan spot on its nominal
    layer."""
    d, _ = layer_nn_plan_xy_distances_and_expected_xyz(planned_xyz, measured_rows, a_is_x=a_is_x)
    return d

def _layer_plan_mu_by_energy_layer(
    planned_xyz: list[tuple[float, float, float]],
    plan_mu: np.ndarray,
    layer_energies: list[float],
) -> list[np.ndarray]:
    buckets: list[list[float]] = [[] for _ in layer_energies]
    for i, (_px, _py, pe) in enumerate(planned_xyz):
        pf = float(pe)
        mu_v = float(plan_mu[i])
        for k, e in enumerate(layer_energies):
            if abs(pf - float(e)) <= 1e-4:
                buckets[k].append(mu_v)
                break
    return [np.asarray(b, dtype=np.float64) for b in buckets]

def layer_nn_local_spot_index_on_layer(
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool = False,
) -> np.ndarray:
    """Nearest plan spot index within each row's nominal energy layer."""
    n = len(measured_rows)
    out = np.full(n, -1, dtype=np.intp)
    if not planned_xyz or n == 0:
        return out
    layer_e = nominal_layer_energies_mev(planned_xyz)
    if not layer_e:
        return out
    li_raw = np.rint(np.asarray([float(t[2]) for t in measured_rows], dtype=np.float64)).astype(
        np.intp, copy=False
    )
    hi = len(layer_e) - 1
    np.clip(li_raw, 0, hi, out=li_raw)
    if a_is_x:
        mx = np.fromiter((float(t[0]) for t in measured_rows), dtype=np.float64, count=n)
        my = np.fromiter((float(t[1]) for t in measured_rows), dtype=np.float64, count=n)
    else:
        mx = np.fromiter((float(t[1]) for t in measured_rows), dtype=np.float64, count=n)
        my = np.fromiter((float(t[0]) for t in measured_rows), dtype=np.float64, count=n)
    meas_xy = np.column_stack([mx, my])
    layer_xyz = _plan_xyz_by_energy_layer(planned_xyz, layer_e)
    trees = _layer_xy_kdtrees_for_qa(layer_xyz)
    for ell in range(len(layer_e)):
        mask = li_raw == ell
        if not np.any(mask):
            continue
        arr = np.asarray(layer_xyz[ell], dtype=np.float64).reshape(-1, 3)
        if arr.shape[0] == 0:
            continue
        q = meas_xy[mask]
        tree = trees[ell] if ell < len(trees) else None
        if tree is not None:
            _dist, idx = _kdtree_query_k1(tree, q)
            out[mask] = np.asarray(idx, dtype=np.intp).reshape(-1)
        else:
            xy_layer = arr[:, 0:2]
            d2 = np.sum((xy_layer[None, :, :] - q[:, None, :]) ** 2, axis=2)
            out[mask] = np.argmin(d2, axis=1).astype(np.intp)
    return out

def nominal_layer_energies_mev(planned_xyz: list[tuple[float, float, float]]) -> list[float]:
    out: list[float] = []
    last: float | None = None
    for *_, e in planned_xyz:
        ef = float(e)
        if last is None or ef != last:
            out.append(ef)
            last = ef
    return out

def fit_ab_to_plan_xy(a: float, b: float, *, a_is_x: bool) -> tuple[float, float]:
    return (a, b) if a_is_x else (b, a)

def _min_xy_dist_to_nominal_energy(
    planned_xyz: list[tuple[float, float, float]],
    e_nom: float,
    mx: float,
    my: float,
) -> float:
    best = float("inf")
    for px, py, pe in planned_xyz:
        if abs(float(pe) - float(e_nom)) > 1e-4:
            continue
        d = float(np.hypot(mx - px, my - py))
        if d < best:
            best = d
    return best

def _kdtree_query_k1(tree: Any, q: np.ndarray) -> tuple[Any, Any]:
    """Nearest-neighbor query; parallel workers only for large batches (avoids Windows overhead)."""
    pts = np.asarray(q, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    nq = int(pts.shape[0])
    if nq == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.intp)
    if nq >= 256:
        try:
            return tree.query(pts, k=1, workers=-1)
        except TypeError:
            pass
    return tree.query(pts, k=1)

def _layer_xy_kdtrees_2d(layer_xy: list[np.ndarray]) -> list[Any | None]:
    """2D cKDTree per nominal layer for fast NN during detector alignment."""
    trees: list[Any | None] = []
    for arr in layer_xy:
        a2 = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
        if a2.shape[0] == 0 or _cKDTree is None:
            trees.append(None)
            continue
        trees.append(_cKDTree(a2))
    return trees

def _layer_nn_plan_targets(
    meas_xy: np.ndarray,
    layer_idx: np.ndarray,
    layer_xy: list[np.ndarray],
    *,
    layer_trees: list[Any | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest plan spot on each row's assigned layer for the given measured XY positions."""
    n = int(meas_xy.shape[0])
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    trees = _layer_xy_kdtrees_2d(layer_xy) if layer_trees is None else layer_trees
    m_out = np.empty((n, 2), dtype=np.float64)
    p_out = np.empty((n, 2), dtype=np.float64)
    keep = np.zeros(n, dtype=bool)
    li_arr = np.asarray(layer_idx, dtype=np.intp).reshape(-1)
    for ell in np.unique(li_arr):
        mask = li_arr == int(ell)
        if not np.any(mask):
            continue
        arr = np.asarray(layer_xy[int(ell)], dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] == 0:
            continue
        q = np.asarray(meas_xy[mask], dtype=np.float64).reshape(-1, 2)
        tree = trees[int(ell)] if int(ell) < len(trees) else None
        if tree is not None:
            _, idx = _kdtree_query_k1(tree, q)
            idx = np.asarray(idx, dtype=np.intp).reshape(-1)
            m_out[mask] = q
            p_out[mask] = arr[idx]
            keep[mask] = True
        else:
            d2 = np.sum((arr[None, :, :] - q[:, None, :]) ** 2, axis=2)
            j = np.argmin(d2, axis=1)
            m_out[mask] = q
            p_out[mask] = arr[j]
            keep[mask] = True
    if not np.any(keep):
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return m_out[keep], p_out[keep]

def _layer_nn_rms_mm(
    meas_xy: np.ndarray,
    layer_idx: np.ndarray,
    layer_xy: list[np.ndarray],
    r_mat: np.ndarray,
    tvec: np.ndarray,
    *,
    layer_trees: list[Any | None] | None = None,
) -> float:
    """RMS NN distance (mm) after applying ``r_mat @ m + tvec`` with per-layer matching."""
    trans = (np.asarray(r_mat, dtype=np.float64) @ meas_xy.T).T + np.asarray(
        tvec, dtype=np.float64
    ).reshape(1, 2)
    m_pairs, p_pairs = _layer_nn_plan_targets(trans, layer_idx, layer_xy, layer_trees=layer_trees)
    if int(m_pairs.shape[0]) == 0:
        return float("inf")
    diff = m_pairs - p_pairs
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

def _plan_xy_by_energy_layer(
    planned_xyz: list[tuple[float, float, float]],
    layer_energies: list[float],
) -> list[np.ndarray]:
    buckets: list[list[list[float]]] = [[] for _ in layer_energies]
    for px, py, pe in planned_xyz:
        pf = float(pe)
        for k, e in enumerate(layer_energies):
            if abs(pf - float(e)) <= 1e-4:
                buckets[k].append([float(px), float(py)])
                break
    return [
        np.asarray(b, dtype=np.float64) if b else np.zeros((0, 2), dtype=np.float64)
        for b in buckets
    ]

def _plan_xyz_by_energy_layer(
    planned_xyz: list[tuple[float, float, float]],
    layer_energies: list[float],
) -> list[np.ndarray]:
    """Same layer bucketing as :func:`_plan_xy_by_energy_layer`; each bucket is (n, 3) X, Y, energy
    (MeV)."""
    buckets: list[list[list[float]]] = [[] for _ in layer_energies]
    for px, py, pe in planned_xyz:
        pf = float(pe)
        for k, e in enumerate(layer_energies):
            if abs(pf - float(e)) <= 1e-4:
                buckets[k].append([float(px), float(py), float(pe)])
                break
    return [
        np.asarray(b, dtype=np.float64) if b else np.zeros((0, 3), dtype=np.float64)
        for b in buckets
    ]

def _build_layer_kdtrees(layer_xy: list[np.ndarray]) -> list[Any] | None:
    """Per-layer 2D nearest point; speeds time_gap refill vs scanning the full plan."""
    if _cKDTree is None:
        return None
    trees: list[Any] = []
    for arr in layer_xy:
        a = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
        if a.shape[0] == 0:
            trees.append(None)
        else:
            trees.append(_cKDTree(a))
    return trees

def _nearest_sqdist_sq_mm2_chunked(
    meas_xy: np.ndarray, plan_xy: np.ndarray, chunk: int = 2048
) -> np.ndarray:
    """min_j ||meas_i - plan_j||^2 for each row i (pure NumPy)."""
    if plan_xy.shape[0] == 0:
        return np.full(meas_xy.shape[0], np.inf, dtype=np.float64)
    out = np.empty(meas_xy.shape[0], dtype=np.float64)
    n = meas_xy.shape[0]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        diff = meas_xy[s:e, None, :] - plan_xy[None, :, :]
        out[s:e] = (diff * diff).sum(axis=2).min(axis=1)
    return out

def _nearest_sqdist_sq_mm2_to_points(meas_xy: np.ndarray, plan_xy: np.ndarray) -> np.ndarray:
    """Squared distance (mm^2) from each meas row to nearest plan row."""
    if plan_xy.shape[0] == 0:
        return np.full(meas_xy.shape[0], np.inf, dtype=np.float64)
    n, m = meas_xy.shape[0], plan_xy.shape[0]
    if _cKDTree is not None and m > 0 and (n * m > 40_000 or n > 25_000):
        tree = _cKDTree(plan_xy)
        d, _ = _kdtree_query_k1(tree, meas_xy)
        return np.asarray(d, dtype=np.float64) ** 2
    return _nearest_sqdist_sq_mm2_chunked(meas_xy, plan_xy)

def _emit_sqdist_to_layers_mm2(meas_xy: np.ndarray, layer_plan_xy: list[np.ndarray]) -> np.ndarray:
    """Squared XY distance (mm²) from each measured row to nearest plan spot on each layer."""
    n, L = meas_xy.shape[0], len(layer_plan_xy)
    cost = np.zeros((n, L), dtype=np.float64)
    trees: list[Any | None] = []
    for pts in layer_plan_xy:
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if _cKDTree is not None and a.shape[0] > 0:
            trees.append(_cKDTree(a))
        else:
            trees.append(None)
    for k in range(L):
        pts = layer_plan_xy[k]
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if a.shape[0] == 0:
            cost[:, k] = np.inf
            continue
        t = trees[k]
        if t is not None:
            d, _ = _kdtree_query_k1(t, meas_xy)
            cost[:, k] = np.asarray(d, dtype=np.float64) ** 2
        else:
            cost[:, k] = _nearest_sqdist_sq_mm2_chunked(meas_xy, a)
    return cost
