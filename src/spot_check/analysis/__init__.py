"""
Plan vs acquisition analysis (public API).

Implementation is split across subpackages; this module re-exports the stable surface
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
    planned_spot_position_counts_from_dicom as planned_spot_position_counts_from_dicom,
)
from spot_check.plan import (
    planned_spot_xyz_and_counts_from_dicom as planned_spot_xyz_and_counts_from_dicom,
)
from spot_check.plan import (
    planned_spot_xyz_from_dicom as planned_spot_xyz_from_dicom,
)
from spot_check.plan import (
    rt_plan_label_from_csv_stem as rt_plan_label_from_csv_stem,
)
