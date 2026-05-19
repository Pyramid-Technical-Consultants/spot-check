"""
SpotCheck — **plan (DICOM) vs acquisition (CSV)** review GUI with embedded PyVista 3D.

**Regulatory / quality posture**

This application is **engineering and quality-assurance tooling**, not a medical device.
It supports visual and numerical review of raster‑scan ion plan data against vendor or
in‑house measurements. **Any clinical or safety‑critical use** is the responsibility of
the deploying organization, including validation, IQ/OQ/PQ, and applicable regulatory
submission. Tunable heuristics and defaults live in :mod:`spot_check.constants`; change them
only under controlled configuration management.

**Intended use**

- Load one **plan** (RT Ion ``.dcm`` or Pyramid plan ``.csv``) and one **acquisition**
  export (``.csv``).
- Assign CSV rows to nominal energy layers per configurable rules, then compare
  **measured vs planned** spot handling in 3D and optional **plan QA** coloring.
- The 3D view **refreshes automatically** when inputs validate (numeric fields use a
  short debounce while typing).

**Persistence**

Window geometry and control values are stored in ``.spot_check_gui_state.json``
under the project root (or current working directory). The file contains no patient
identifiers when paths are anonymized.

**Requirements:** ``pip install pydicom pyvista PySide6`` (VTK is bundled with PyVista).
For **large acquisitions** (hundreds of thousands+ rows), also install **scipy** so plan QA
and Viterbi layer costs use **cKDTree** acceleration (see :mod:`spot_check.analysis`).

**Diagnostics:** Set environment variable ``SPOT_CHECK_LOG`` to ``DEBUG``, ``INFO``,
``WARNING``, or ``ERROR`` (see :func:`spot_check.logging_utils.configure_logging`).
"""

import logging
import sys

from spot_check.gui.controller import SpotCheckController
from spot_check.logging_utils import configure_logging


def run_gui() -> None:
    """Launch the SpotCheck plan vs acquisition GUI."""
    SpotCheckController().run()


def main() -> None:
    """Console entry point for ``spot-check``."""
    run_gui()


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        raise
    except Exception as exc:
        # stderr for operators without log config; full traceback when SPOT_CHECK_LOG allows
        configure_logging()
        logging.getLogger(__name__).exception("SpotCheck GUI startup failed")
        print("SpotCheck GUI startup failed:", exc, file=sys.stderr)
        raise
