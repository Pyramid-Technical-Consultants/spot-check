"""Domain-specific errors for SpotCheck plan vs acquisition tooling."""

from __future__ import annotations


class SpotCheckError(ValueError):
    """Base class for predictable failures in plan/CSV processing."""


class PlanDataError(SpotCheckError):
    """DICOM plan missing or not usable (e.g. no ion spots)."""


class AcquisitionDataError(SpotCheckError):
    """CSV acquisition missing required columns or parseable rows."""


class GeometryConfigError(SpotCheckError):
    """Invalid combination of geometry / layer / QA parameters."""
