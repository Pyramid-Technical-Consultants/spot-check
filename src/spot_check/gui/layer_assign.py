"""Map GUI layer-assignment radio values to CSV load parameters."""

from __future__ import annotations

VALID_LAYER_ASSIGN_MODES = frozenset({"gate_counter", "auto", "auto_layer_em"})


def normalize_layer_assign_mode(raw: str) -> str:
    """Persisted / UI mode: ``gate_counter``, ``auto``, or ``auto_layer_em``."""
    m = str(raw or "").strip().lower().replace("-", "_")
    if m in ("unified",):
        return "auto"
    if m in ("time_gap", "plan_viterbi"):
        return "gate_counter"
    if m in VALID_LAYER_ASSIGN_MODES:
        return m
    return "gate_counter"


def resolve_layer_assign_mode(
    layer_assign_mode: str,
) -> tuple[str, str, bool]:
    """Return ``(layer_mode, auto_assign_method, auto_infer_params)`` for CSV load."""
    m = normalize_layer_assign_mode(layer_assign_mode)
    if m == "auto_layer_em":
        return "auto", "layer_em", False
    if m == "auto":
        return "auto", "episodes", True
    return "gate_counter", "episodes", False
