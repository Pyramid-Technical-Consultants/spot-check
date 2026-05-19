"""Colors."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403


def _hex_to_rgb_u8(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) == 6:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 214, 39, 40

def _measured_alpha_u8_from_channel_weights(
    weights: np.ndarray,
    *,
    gamma: float = 0.5,
    alpha_floor: float = 0.1,
    alpha_ceil: float = 0.95,
) -> np.ndarray:
    """Per-point alpha uint8 from IX512 channel-sum style weights (same mapping as 3D measured
    tint)."""
    w = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-18)
    if w.size == 0:
        return np.zeros(0, dtype=np.uint8)
    lo, hi = float(np.percentile(w, 4)), float(np.percentile(w, 96))
    if hi <= lo * 1.001:
        wn = np.ones_like(w, dtype=np.float64)
    else:
        wn = np.clip((w - lo) / (hi - lo), 0.0, 1.0)
    alpha = alpha_floor + (alpha_ceil - alpha_floor) * np.power(wn, gamma)
    return (255.0 * alpha).astype(np.uint8)

def measured_rgba_by_channel_weight(
    weights: np.ndarray,
    *,
    color_hex: str = _MEASURED_COLOR_3D,
    gold_mask: np.ndarray | None = None,
    gold_hex: str = _PARTIAL_AXIS_MEAS_COLOR_3D,
    gamma: float = 0.5,
    alpha_floor: float = 0.1,
    alpha_ceil: float = 0.95,
) -> np.ndarray:
    """Per-point RGBA uint8; low channel sum -> lower alpha (de-emphasized)."""
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-18)
    n = w.shape[0]
    gm = np.asarray(gold_mask, dtype=bool) if gold_mask is not None else np.zeros(n, dtype=bool)
    if gm.shape[0] != n:
        raise ValueError("gold_mask length must match weights length")
    r0, g0, b0 = _hex_to_rgb_u8(color_hex)
    r1, g1, b1 = _hex_to_rgb_u8(gold_hex)
    rgba = np.zeros((n, 4), dtype=np.uint8)
    rgba[:, 0] = np.where(gm, np.uint8(r1), np.uint8(r0))
    rgba[:, 1] = np.where(gm, np.uint8(g1), np.uint8(g0))
    rgba[:, 2] = np.where(gm, np.uint8(b1), np.uint8(b0))
    rgba[:, 3] = _measured_alpha_u8_from_channel_weights(
        w, gamma=gamma, alpha_floor=alpha_floor, alpha_ceil=alpha_ceil
    )
    return rgba
