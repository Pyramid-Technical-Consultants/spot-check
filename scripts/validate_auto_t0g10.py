#!/usr/bin/env python3
"""Optional local check: auto vs gate_counter agreement on T0G10-like files under test_data/.



Skips gracefully when matching .dcm/.csv pairs are absent (typical in CI).

"""



from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_SRC = _REPO / "src"

if _SRC.is_dir():

    sys.path.insert(0, str(_SRC))





def main() -> int:

    try:

        import numpy as np

        from spot_check import analysis
        from spot_check.plan import planned_spot_xyz_and_counts_from_dicom

    except ImportError:

        print("Install package in editable mode: pip install -e .")

        return 1



    test_dir = _REPO / "test_data"

    if not test_dir.is_dir():

        print(f"No {test_dir}; skip.")

        return 0



    dcms = sorted(test_dir.glob("*.dcm"))

    if not dcms:

        print("No .dcm under test_data; skip.")

        return 0



    for dcm in dcms[:1]:

        csvs = sorted(test_dir.glob("*.csv"))

        if not csvs:

            print("No CSV next to test_data; skip.")

            return 0

        csv_path = csvs[0]

        planned, _, _, n_kept, n_raw = planned_spot_xyz_and_counts_from_dicom(dcm)

        gate = analysis.measured_spot_abc_from_csv(

            csv_path,

            planned_xyz=list(planned),

            layer_mode="gate_counter",

            a_is_x=False,

            aggregate_spots=True,

        )

        auto = analysis.measured_spot_abc_from_csv(

            csv_path,

            planned_xyz=list(planned),

            layer_mode="auto",

            a_is_x=False,

            auto_infer_params=True,

        )

        diag = analysis.last_auto_episode_diagnostics()

        params = analysis.last_auto_layer_params()

        print(dcm.name, csv_path.name)

        print(f"  n_plan_kept={n_kept} n_plan_raw={n_raw} n_gate={len(gate)} n_auto={len(auto)}")

        if diag:

            print(

                f"  auto raw_episodes={diag.n_raw_episodes} "

                f"aligned={diag.n_after_align}/{diag.n_plan} ok={diag.count_align_ok}"

            )

        if params:

            print(
                f"  params dt={params.episode_gap_s:g}s xy_jump={params.spot_xy_jump_mm:g}mm"
            )

        if len(gate) == len(auto) and gate:

            lg = [int(r[2]) for r in gate]

            la = [int(r[2]) for r in auto]

            m = sum(1 for a, b in zip(lg, la) if a == b)

            ga = np.array([[r[0], r[1]] for r in gate])

            aa = np.array([[r[0], r[1]] for r in auto])

            d = np.sqrt(np.sum((ga - aa) ** 2, axis=1))

            print(f"  layer match {m}/{len(gate)} ({100*m/len(gate):.1f}%)")

            print(f"  XY mm median={float(np.median(d)):.3f} p90={float(np.percentile(d,90)):.3f}")

    return 0





if __name__ == "__main__":

    raise SystemExit(main())

