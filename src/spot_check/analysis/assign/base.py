"""Base classes and shared post-assignment hooks for all layer assigners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

import numpy as np

from spot_check.analysis.assign.types import (
    LAYER_ASSIGN_MODES,
    AssignCsvParams,
    MeasuredAssignResult,
)
from spot_check.analysis.spatial import (
    _plan_xy_by_energy_layer,
    layer_nn_local_spot_index_on_layer,
    nominal_layer_energies_mev,
)


def normalize_layer_mode(mode: str) -> str:
    m = str(mode).strip().lower().replace("-", "_")
    if m not in LAYER_ASSIGN_MODES:
        raise ValueError(
            f"layer_mode must be one of {sorted(LAYER_ASSIGN_MODES)}, got {mode!r}"
        )
    return m


def plan_index_per_row_layer_nn(
    rows: list[tuple[float, ...]],
    planned_xyz: list[tuple[float, float, float]],
    *,
    a_is_x: bool,
) -> list[int]:
    """Map each row to a global plan slot via layer-local nearest-neighbor plan XY."""
    n = len(rows)
    if n == 0:
        return []
    layer_e = nominal_layer_energies_mev(planned_xyz)
    layer_xy = _plan_xy_by_energy_layer(planned_xyz, layer_e)
    spots_per = [int(np.asarray(arr, dtype=np.float64).reshape(-1, 2).shape[0]) for arr in layer_xy]
    cumul: list[int] = [0]
    for c in spots_per:
        cumul.append(cumul[-1] + int(c))
    hi = max(0, len(spots_per) - 1)
    local = layer_nn_local_spot_index_on_layer(planned_xyz, rows, a_is_x=a_is_x)
    out: list[int] = []
    for row, loc in zip(rows, local, strict=True):
        if int(loc) < 0:
            out.append(-1)
            continue
        lay = max(0, min(int(float(row[2])), hi))
        out.append(int(cumul[lay]) + int(loc))
    return out


class LayerAssignerBase(ABC):
    """Common hooks for CSV layer assignment modes."""

    layer_mode: str

    def validate(self, params: AssignCsvParams) -> None:
        """Raise ``ValueError`` when mode-specific params are invalid."""

    @abstractmethod
    def assign(self, params: AssignCsvParams) -> MeasuredAssignResult: ...

    def spot_ids_are_plan_slots(self, result: MeasuredAssignResult) -> bool:
        """True when ``spot_ids`` are delivery-order plan slot indices (0…N−1)."""
        return True

    def plan_index_per_row(
        self,
        result: MeasuredAssignResult,
        planned_xyz: list[tuple[float, float, float]],
    ) -> list[int]:
        if self.spot_ids_are_plan_slots(result) and len(result.spot_ids) == len(result.rows):
            return [int(s) for s in result.spot_ids]
        return plan_index_per_row_layer_nn(
            result.rows,
            planned_xyz,
            a_is_x=result.a_is_x,
        )

    def finalize(
        self,
        result: MeasuredAssignResult,
        *,
        planned_xyz: list[tuple[float, float, float]] | None,
    ) -> MeasuredAssignResult:
        """Map rows to plan slots and mark plan spots with no assigned data."""
        if not planned_xyz:
            return replace(result, plan_index_per_row=None, plan_spots_no_data=None)
        n_plan = len(planned_xyz)
        if not result.rows:
            mask = np.ones(n_plan, dtype=bool)
            return replace(
                result,
                n_plan_spots=n_plan,
                planned_xyz=list(planned_xyz),
                plan_index_per_row=[],
                plan_spots_no_data=mask,
            )
        plan_idx_row = self.plan_index_per_row(result, planned_xyz)
        have: set[int] = set()
        for pi in plan_idx_row:
            if int(pi) >= 0:
                have.add(int(pi))
        mask = np.ones(n_plan, dtype=bool)
        for i in have:
            if 0 <= i < n_plan:
                mask[i] = False
        return replace(
            result,
            n_plan_spots=n_plan,
            planned_xyz=list(planned_xyz),
            plan_index_per_row=plan_idx_row,
            plan_spots_no_data=mask,
        )


def finalize_measured_assign_coverage(
    result: MeasuredAssignResult,
    *,
    planned_xyz: list[tuple[float, float, float]] | None,
) -> MeasuredAssignResult:
    """Dispatch finalize to the assigner registered for ``result.layer_mode``."""
    from spot_check.analysis.assign import get_layer_assigner

    impl = get_layer_assigner(result.layer_mode)
    return impl.finalize(result, planned_xyz=planned_xyz)


def plan_spots_without_assignment_data(result: MeasuredAssignResult) -> np.ndarray | None:
    """Bool mask (len = n plan spots): True where assignment left no rows on a plan slot."""
    return result.plan_spots_no_data
