"""
RT Ion plan vs acquisition — DICOM + CSV → PyVista 3D comparison.

**Regulatory:** This is engineering analysis software, not a medical device. Operational
qualification and clinical use are the responsibility of the deploying site. Tunable defaults
and heuristics live in :mod:`spot_check.constants`; do not change them without documented review.

Planned beams: spot (x, y, nominal energy) from the Ion Control Point maps; optional
**Scanning Spot Size** (300A,0398) FWHM in mm (gantry X, Y) can scale plan markers as thin
ellipsoids in 3D (see ``scale_plan_spots_by_dicom_fwhm`` in :func:`show_comparison_3d_pyvista`).

Layer assignment (nominal energy index per row):

Acquisition CSVs may include a ``Gate Signal`` column (e.g. IC256); it is **never read**.
Only ``Gate Counter`` is used, and only in **gate_counter** mode / optional aggregation.

- **time_gap** — ``TIME_LAYER_GAP_S_DEFAULT`` plus refill heuristics (constants below).
  ``Gate Counter`` is ignored unless ``aggregate_spots=True``.

- **auto** — **deadtime** segmentation from a scale-free delivery metric: geometric mean of
  IX512 channel sum and fit amplitude A versus rolling baselines (works across accelerators),
  plan spot-count alignment with optional **plan XY boundary refinement**, then
  delivery-order layers when spot count matches the plan; otherwise proportional spread
  by episode order (not XY Viterbi). Never reads ``Gate Counter``.
  With ``aggregate_spots=True``, each episode is one **weighted-mean** spot; when False, every
  on-spot CSV row is plotted and shares the episode layer.

- **plan_viterbi** — global decode: each row keeps measured A/B; layer index comes from
  a monotone path (stay or +1 layer) minimizing distance-to-plan, plus a penalty for
  advancing. No invented coordinates; works when the machine changes energy with no timing gap.

- **gate_counter** — uses CSV ``Gate Counter``: **odd** values mark a **spot** phase (many
  consecutive rows may share the same count — one spot); **even** marks **deadtime** (also
  many rows per value). Nominal layer follows DICOM spot order; **spot index advances only
  when the gate count changes to a new odd value**. Requires ``planned_xyz``.

**Refill (time_gap mode):** a large Δt only *suppresses* a nominal energy step if the fit returns
near the **same plan XY** as the last row before the gap (``REFILL_SAME_SPOT_XY_TOLERANCE_MM``),
otherwise a step may occur when the next slice explains XY better
(``_layer_advance_plausible_vs_refill``).

**Spot aggregation (optional):** when ``aggregate_spots=True``, each run of consecutive rows with
the same **odd** ``Gate Counter`` value is one delivered spot (**even** gate ends the spot), or
with ``aggregate_even_rows_after_odd`` > 0 in **gate_counter** mode, up to that many **even-phase**
rows with valid fits after each odd→even transition are merged into the same weighted mean.
Rows in that run collapse to one point: **weighted means** of fit mean positions
(after imputation), nominal layer index, and fit σ_A / σ_B when present. Weights come from
``spot_weight_mode``: IX512 channel sum and/or Fit Amplitude A/B columns (see
:func:`measured_spot_weight_from_row`). Each measured row is a 7-tuple
``(..., σ_A, σ_B)`` (σ may be NaN); aggregation replaces runs with one weighted-mean row.

**Detector XY alignment (optional):** :func:`align_measured_to_plan_detector_xy` matches each
measured row to the nearest plan spot on its assigned nominal layer, then fits a 2D rigid
transform (rotation + translation) minimizing RMSE to those plan targets and applies it to
all measured A/B positions in plan coordinates (see :class:`DetectorRigidAlign2D`). Large
detector rotations (90° and beyond) and A↔B axis swaps are handled via multi-start ICP with
coarse rotation seeds.

**Plan QA coloring (optional):** colors each measured point from Euclidean XY distance to the
nearest plan spot on its assigned layer: green ≤ pass mm (default 1), amber between pass and
warn mm (default 3), red > warn mm (see :func:`measured_rgba_by_plan_qa`). Measured markers
are drawn as flat circular discs; optional **spot-weight opacity** (see
``weight_measured_by_channel``) or
Optionally draw **error lines** from warn/fail points to that nearest plan spot (see
``plan_qa_draw_error_lines`` in :func:`show_comparison_3d_pyvista`).
When ``plan_qa_hide_pass_spots`` is true, pass-tier measured points are dropped from the cloud
so only warn and fail markers are drawn (counts in the caption still include all tiers).

In 3D, **nominal-layer band** filtering can be driven from the GUI when ``embed_qt`` / ``slice_qt``
are passed (PySide6 host + checkbox / slider bindings from :mod:`spot_check.gui`). Standalone
runs use PyVista widgets: **5-layer slice** vs **full stack**, with a **center layer** control in
plan order.
Plan QA error lines follow the same filter.

Implementation is split across submodules; this package re-exports the stable surface
used by the GUI and external scripts.
"""

from __future__ import annotations

from spot_check.analysis._core import *  # noqa: F403
from spot_check.plan import (
    find_dicom_for_csv as find_dicom_for_csv,
)
from spot_check.plan import (
    infer_csv_plan_tag as infer_csv_plan_tag,
)
from spot_check.plan import (
    is_supported_plan_file as is_supported_plan_file,
)
from spot_check.plan import (
    plan_label_from_path as plan_label_from_path,
)
from spot_check.plan import (
    planned_spot_position_counts_from_dicom as planned_spot_position_counts_from_dicom,
)
from spot_check.plan import (
    planned_spot_xyz_and_counts_from_dicom as planned_spot_xyz_and_counts_from_dicom,
)
from spot_check.plan import (
    planned_spot_xyz_and_counts_from_plan as planned_spot_xyz_and_counts_from_plan,
)
from spot_check.plan import (
    planned_spot_xyz_from_dicom as planned_spot_xyz_from_dicom,
)
from spot_check.plan import (
    rt_plan_label_from_csv_stem as rt_plan_label_from_csv_stem,
)
