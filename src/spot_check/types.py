"""Shared type aliases for SpotCheck."""

from __future__ import annotations

from typing import Literal

LayerAssignMode = Literal["auto", "gate_counter", "time_gap", "plan_viterbi"]
SpotWeightMode = Literal["channel_sum", "fit_amplitude_a", "fit_amplitude_b"]
PlanQaMode = Literal["position", "dose"]
