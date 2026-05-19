"""Pyvista Backend."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403

_pyvista_import_error: ImportError | None = None

try:
    import pyvista as pv
except ImportError as _pv_exc:  # pragma: no cover
    pv = None  # type: ignore[assignment]
    _pyvista_import_error = _pv_exc

def require_pyvista() -> Any:
    """Return the PyVista module, re-importing once if the optional first import failed."""
    global pv, _pyvista_import_error
    if pv is not None:
        return pv
    try:
        import pyvista as _pv

        pv = _pv
        _pyvista_import_error = None
        return pv
    except ImportError as exc:
        _pyvista_import_error = exc
        detail = f": {exc}" if exc else ""
        raise RuntimeError(
            "PyVista is required for the 3D view"
            f"{detail}. Install with: pip install pyvista"
        ) from exc
