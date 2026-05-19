"""Split gui/app.py into app.py + controller.py."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
app_path = ROOT / "src/spot_check/gui/app.py"
lines = app_path.read_text(encoding="utf-8").splitlines(keepends=True)

body_lines = lines[124:1583]
indented = [("    " + ln) if ln.strip() else ln for ln in body_lines]

controller_parts = [
    '"""SpotCheck GUI controller — builds the main window and 3D refresh pipeline."""\n',
    "\nfrom __future__ import annotations\n\n",
    "".join(lines[34:121]),
    "\n\nclass SpotCheckController:\n",
    '    """Builds and runs the SpotCheck Qt main window."""\n\n',
    "    def run(self) -> None:\n",
    "".join(indented),
]
controller_path = ROOT / "src/spot_check/gui/controller.py"
controller_path.write_text("".join(controller_parts), encoding="utf-8")

thin_parts = [
    "".join(lines[:33]),
    "\nfrom spot_check.gui.controller import SpotCheckController\n",
    "from spot_check.logging_utils import configure_logging\n\n",
    "import logging\nimport sys\n\n",
    "from spot_check._version import __version__\n\n\n",
    "def run_gui() -> None:\n",
    '    """Launch the SpotCheck plan vs acquisition GUI."""\n',
    "    SpotCheckController().run()\n\n\n",
    "".join(lines[1585:]),
]
app_path.write_text("".join(thin_parts), encoding="utf-8")
print("wrote", controller_path, len(controller_parts[0].splitlines()) + len(indented))
print("wrote", app_path, len("".join(thin_parts).splitlines()))
