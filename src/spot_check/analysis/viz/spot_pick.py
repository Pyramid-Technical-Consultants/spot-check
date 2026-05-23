"""Double-click spot picking on the PyVista comparison plotter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from spot_check.analysis.viz.spot_info import SpotKind

SpotPickCallback = Callable[["SpotPickEvent"], None]


@dataclass(frozen=True)
class SpotPickEvent:
    kind: SpotKind
    spot_index: int
    display_x: int
    display_y: int


def _actor_mesh(actor: Any) -> Any | None:
    mapper = getattr(actor, "mapper", None)
    if mapper is None:
        return None
    return getattr(mapper, "dataset", None) or getattr(mapper, "input", None)


def _spot_index_from_pick(actor: Any, point_id: int) -> int | None:
    mesh = _actor_mesh(actor)
    if mesh is None or point_id < 0:
        return None
    try:
        arr = mesh.point_data["spot_index"]
    except (KeyError, TypeError, IndexError):
        return None
    if point_id >= int(arr.shape[0]):
        return None
    return int(arr[point_id])


def disconnect_spot_pick(plotter: Any) -> None:
    """Remove spot-pick observers from a plotter (safe if never wired)."""
    ctx = getattr(plotter, "_spot_check_pick", None)
    if not ctx:
        return
    iren = getattr(plotter, "iren", None)
    obs_id = ctx.get("observer_id")
    if iren is not None and obs_id is not None:
        try:
            iren.remove_observer(obs_id)
        except Exception:
            pass
    plotter._spot_check_pick = None


def wire_spot_double_click_pick(
    plotter: Any,
    *,
    plan_actor: Any | None,
    meas_actor: Any | None,
    plan_missing_actor: Any | None,
    plan_visible_mask: np.ndarray | None,
    plan_has_data_mask: np.ndarray | None,
    on_picked: SpotPickCallback | None,
) -> None:
    """Attach a double-click observer that cell-picks plan or measured spot actors."""
    disconnect_spot_pick(plotter)
    if on_picked is None:
        return
    iren = getattr(plotter, "iren", None)
    if iren is None:
        return

    from vtkmodules.vtkRenderingCore import vtkCellPicker

    picker = vtkCellPicker()
    picker.SetTolerance(0.005)

    actor_kind: dict[int, SpotKind] = {}
    for actor in (plan_actor, plan_missing_actor):
        if actor is not None:
            actor_kind[id(actor)] = "plan"
    if meas_actor is not None:
        actor_kind[id(meas_actor)] = "measured"

    has_data = (
        np.asarray(plan_has_data_mask, dtype=bool).reshape(-1)
        if plan_has_data_mask is not None
        else None
    )
    vis_ref: dict[str, np.ndarray | None] = {
        "mask": (
            np.asarray(plan_visible_mask, dtype=bool).reshape(-1)
            if plan_visible_mask is not None
            else None
        )
    }

    def _plan_spot_visible(spot_i: int, *, from_missing: bool) -> bool:
        mask = vis_ref.get("mask")
        if mask is None or spot_i < 0 or spot_i >= int(mask.shape[0]):
            return True
        if not bool(mask[spot_i]):
            return False
        if has_data is None or spot_i >= int(has_data.shape[0]):
            return True
        if from_missing:
            return not bool(has_data[spot_i])
        return bool(has_data[spot_i])

    def _on_double_click(_obj: Any, _event: str) -> None:
        interactor = iren.interactor if hasattr(iren, "interactor") else iren
        x, y = interactor.GetEventPosition()
        if not picker.Pick(x, y, 0, plotter.renderer):
            return
        actor = picker.GetActor()
        if actor is None:
            return
        kind = actor_kind.get(id(actor))
        if kind is None:
            return
        point_id = int(picker.GetPointId())
        spot_i = _spot_index_from_pick(actor, point_id)
        if spot_i is None:
            return
        if kind == "plan" and not _plan_spot_visible(
            spot_i, from_missing=actor is plan_missing_actor
        ):
            return
        on_picked(
            SpotPickEvent(
                kind=kind,
                spot_index=spot_i,
                display_x=int(x),
                display_y=int(y),
            )
        )

    observer_id = iren.add_observer("LeftButtonDoubleClickEvent", _on_double_click)
    plotter._spot_check_pick = {
        "observer_id": observer_id,
        "picker": picker,
        "update_plan_visibility": lambda mask: vis_ref.update(
            {"mask": np.asarray(mask, dtype=bool).reshape(-1) if mask is not None else None}
        ),
    }


def update_spot_pick_plan_visibility(plotter: Any, plan_visible_mask: np.ndarray | None) -> None:
    ctx = getattr(plotter, "_spot_check_pick", None)
    if not ctx:
        return
    fn = ctx.get("update_plan_visibility")
    if callable(fn):
        fn(plan_visible_mask)
