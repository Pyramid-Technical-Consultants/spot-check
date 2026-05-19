# SpotCheck

Engineering QA tooling to compare **RT Ion therapy plans** (DICOM) against **tabular acquisition exports** (CSV), with interactive 3D visualization.

**Not a medical device.** Validate any clinical workflow independently before use.

## Install (development)

On **Windows** (especially the Microsoft Store `python`), do **not** run bare `pip install -e .` — it installs into a user folder whose path is too long for PySide6 and fails with `OSError: [Errno 2] No such file or directory` under `...\PySide6\qml\...`.

Use the project virtualenv instead:

**Command Prompt (no PowerShell script policy issues):**

```bat
scripts\setup.bat
run-spot-check.bat
```

Or after setup, double-click `run-spot-check.bat` in the project root.

**PowerShell:**

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
spot-check
```

If `setup.ps1` is blocked (*running scripts is disabled*), either use `scripts\setup.bat` above or run once:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
```

That only relaxes policy for the current PowerShell window.

**Git Bash / macOS / Linux:**

```bash
./scripts/setup.sh
source .venv/bin/activate
spot-check
```

Manual equivalent:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -e ".[fast,dev]"
```

`fast` adds SciPy for accelerated plan QA and Viterbi layer assignment on large acquisitions.

If you must install outside a venv, enable [Windows long paths](https://pip.pypa.io/warnings/enable-long-paths) and prefer [python.org](https://www.python.org/downloads/) over the Store build.

## Run the GUI

```bash
spot-check
```

Or:

```bash
python -m spot_check.gui
```

Window layout and control values persist in `.spot_check_gui_state.json` next to the executable (frozen build) or the project root (development).

Set `SPOT_CHECK_LOG` to `DEBUG`, `INFO`, `WARNING`, or `ERROR` for stderr logging.

## Versioning

The release version lives in a single file:

- `src/spot_check/_version.py` → `__version__`

`pyproject.toml` reads that value automatically. The GUI window title shows the current version.

Bump the version:

```bash
./scripts/bump-version.sh patch   # or minor | major | 1.2.3
git add src/spot_check/_version.py
git commit -m "chore: release v1.0.1"
git tag v1.0.1
git push && git push --tags
```

The git tag `vX.Y.Z` must match `__version__` in `_version.py`. Pushing a tag triggers the Windows release workflow.

## Build Windows executable (local)

Requires **Git Bash** (or WSL) on Windows:

```bash
./scripts/build-windows.sh
```

Output:

- `dist/SpotCheck/SpotCheck.exe` — run this (keep the whole folder together)
- `dist/SpotCheck-<version>-windows-x64.zip` — zip for distribution

The frozen build omits **SciPy** and **matplotlib** (a runtime hook stubs PyVista’s optional
VTK matplotlib text backend; axis labels use FreeType). `packaging/trim_windows_bundle.py`
strips unused Qt/VTK/PIL payloads after PyInstaller (including `opengl32sw.dll` by default —
set `SPOT_CHECK_TRIM_AGGRESSIVE=0` before building if 3D fails on software-OpenGL systems).

Optional: pin the build version without editing the file:

```bash
SPOT_CHECK_VERSION=1.0.1 ./scripts/build-windows.sh
```

Test data under `test_data/` is not included in the bundle (and is gitignored).

## Lint (before push)

CI runs `ruff check src tests packaging`. After `scripts/setup.*`, **pre-commit** hooks run the same check (with `--fix`) on staged files under `src/`, `tests/`, and `packaging/`.

Manual check:

```bash
ruff check src tests packaging
# or auto-fix import order / other fixable issues:
ruff check --fix src tests packaging
```

## CI / releases

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **CI** | Push / PR to `main` / `master` | Ruff, compile, pytest |
| **CI → Windows exe (beta)** | Every **push** (not tags) | Beta folder artifact (one zip from Actions) |
| **Release Windows** | Tag `v*` or manual | Stable `.zip` on GitHub Release + folder artifact |

**Beta builds:** On each push, download **Actions → workflow run → Artifacts** → unzip once → run `SpotCheck/SpotCheck.exe` (keep the whole folder). CI uploads `dist/SpotCheck/`, not the release zip, so you are not double-zipped.

**Stable releases:** Tag `v1.0.0` (must match `src/spot_check/_version.py`) → GitHub Release attaches `SpotCheck-<version>-windows-x64.zip` (single zip for distribution).

Manual stable build without a tag: Actions → *Release Windows* → *Run workflow*.

## Layout

```
spot-check/
├── pyproject.toml
├── packaging/spot-check.spec
├── scripts/build-windows.sh
├── src/spot_check/
│   ├── analysis/     # plan vs acquisition (public API)
│   ├── plan/         # DICOM plan loading
│   ├── geometry/     # Z axis / cube-axes helpers
│   ├── gui/          # PySide6 app (state, pipeline, app)
│   ├── models.py
│   └── ...
└── tests/
```

## License

MIT — see [LICENSE](LICENSE).
