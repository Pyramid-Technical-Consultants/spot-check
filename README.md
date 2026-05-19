# SpotCheck

Compare **RT Ion therapy plans** (DICOM or Pyramid plan CSV) with **machine acquisition exports** (CSV) in an interactive 3D viewer. SpotCheck loads planned spot positions and nominal energies, aligns measured fit positions to the plan, and highlights pass/warn/fail against plan QA thresholds.

**Not a medical device.** Validate any clinical workflow independently before use.

## What it does

- **Plan** — Read spot (x, y, nominal energy) from Ion Control Point DICOM or Pyramid plan CSV (`X_POSITION`, `Y_POSITION`, `ENERGY`, `CHARGE_REQ`, optional `BEAM_SIZE`); optional FWHM for ellipsoid plan markers.
- **Acquisition** — Parse tabular exports (`.csv` or `.csv.gz`): fit positions, amplitudes, IX512 channel sum, optional Gate Counter and σ columns.
- **Layer assignment** — Group rows into delivered spots and assign a nominal energy layer index (see [Layer modes](#layer-modes)).
- **3D view** — PyVista plan vs measured clouds; proton water-depth Z axis; optional layer slice band, plan QA coloring, and error lines to nearest plan spots.
- **Detector alignment** — Optional 2D rigid fit of measured XY to plan (multi-start ICP; GUI on by default).

The PySide6 GUI persists paths and settings in `.spot_check_gui_state.json` (project root in dev; next to the executable when frozen).

## Layer modes

| Mode | Used in GUI | Summary |
|------|-------------|---------|
| **gate_counter** | Yes (default) | Odd **Gate Counter** phases = one spot; even = deadtime. Nominal layer follows DICOM delivery order. Requires Gate Counter on the CSV when aggregating spots. |
| **auto** | Yes | **Signal** episodes (timing, weight, XY); merge/split to plan spot count. Gate Counter is not used for segmentation. When alignment succeeds, nominal layers match gate_counter aggregation if that column is present; otherwise delivery order, else **Viterbi** on centroids. Thresholds from `infer_auto_layer_params`. |
| **time_gap** | API only | Layer steps from inter-row Δt and refill heuristics. |
| **plan_viterbi** | API only | Per-row monotone Viterbi decode to nearest plan layer (no episode segmentation). |

**Auto vs gate_counter:** Auto is for exports **without** a reliable Gate Counter. Clinic-scale fixtures under `test_data/` (e.g. T0G10) are used in tests to check that auto agrees with gate_counter when both apply; production auto mode does not consume that column.

Python details: `spot_check.analysis` package docstring and `measured_spot_abc_from_csv()`.

## Install (development)

Requires **Python 3.10+**. On Windows, use the project venv (avoids PySide6 path-length issues with Store Python):

```bat
scripts\setup.bat
run-spot-check.bat
```

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
spot-check
```

```bash
./scripts/setup.sh
source .venv/bin/activate
spot-check
```

Manual: `python -m venv .venv`, activate, then `pip install -e ".[fast,dev]"`.

- **`fast`** — SciPy (faster plan QA and Viterbi on large spot lists).
- **`dev`** — pytest, ruff, pre-commit, mypy.

## Run

```bash
spot-check
# or
python -m spot_check.gui
```

Logging: set `SPOT_CHECK_LOG` to `DEBUG`, `INFO`, `WARNING`, or `ERROR`.

## Development

```bash
ruff check src tests packaging
pytest -q
pytest -q -m slow          # large episode / CSV regressions
```

Optional clinic checks (need `test_data/` plan + CSV):  
`pytest -q tests/test_auto_t0g10_agreement.py`

Pre-commit (after setup): same Ruff rules on staged `src/`, `tests/`, `packaging/`.

## Versioning

Version: `src/spot_check/_version.py` (`__version__`). Bump:

```bash
./scripts/bump-version.sh patch   # or minor | major | 1.2.3
```

Tag `vX.Y.Z` must match `__version__` for the Windows release workflow.

## Windows executable

Git Bash on Windows:

```bash
./scripts/build-windows.sh
```

- `dist/SpotCheck/SpotCheck.exe` — run with the full folder
- `dist/SpotCheck-<version>-windows-x64.zip` — distribution zip

Frozen builds omit SciPy/matplotlib; `packaging/trim_windows_bundle.py` trims Qt/VTK bulk. Set `SPOT_CHECK_TRIM_AGGRESSIVE=0` before build if 3D fails on software OpenGL.

`test_data/` is not bundled (gitignored).

## CI

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **CI** | Push / PR to `main` / `master` | Ruff, byte-compile, pytest |
| **CI → Windows exe (beta)** | Push (non-tag) | Beta zip artifact |
| **Release Windows** | Tag `v*` or manual | Release zip |
| **Regression** | Weekly + manual | `pytest -m slow`; optional T0G10 tests if `test_data/` present |

## Layout

```
spot-check/
├── pyproject.toml
├── packaging/          # PyInstaller spec, trim script
├── scripts/            # setup, build-windows, bump-version, validate_auto_t0g10
├── src/spot_check/
│   ├── analysis/       # measured, episodes, auto_columns, viz, plan QA
│   ├── plan/           # DICOM
│   ├── geometry/       # Z axis, cube axes
│   └── gui/            # PySide6 app
└── tests/
```

## License

MIT — see [LICENSE](LICENSE).
