"""
SpotCheck — RT Ion plan vs acquisition analysis.

Re-exports configuration and domain errors. Core algorithms live in
:mod:`spot_check.analysis`; the desktop GUI is :mod:`spot_check.gui.app`.

**Not a medical device.** Operational qualification is the responsibility of the
deploying organisation; this package provides engineering tooling only.
"""

from __future__ import annotations

from . import analysis, constants
from ._version import __version__
from .constants import project_root
from .exceptions import (
    AcquisitionDataError,
    GeometryConfigError,
    PlanDataError,
    SpotCheckError,
)

__all__ = [
    "AcquisitionDataError",
    "GeometryConfigError",
    "PlanDataError",
    "SpotCheckError",
    "analysis",
    "constants",
    "project_root",
    "__version__",
]
