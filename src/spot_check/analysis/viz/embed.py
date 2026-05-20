"""Embed."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.viz.data import _nominal_layer_index_band_mev

_VTK_TK_PUMP: dict[str, Any] = {"after_id": None, "plotter": None}

def _vtk_rendering_tk_dll_present() -> bool:
    """Return True if the VTK–Tk bridge library is present (often absent in pip wheels on
    Windows)."""
    try:
        from pathlib import Path

        import vtkmodules

        libs = Path(vtkmodules.__file__).resolve().parent.parent / "vtk.libs"
        if not libs.is_dir():
            return False
        if any(libs.glob("vtkRenderingTk*.dll")):
            return True
        if any(libs.glob("libvtkRenderingTk*.so*")):
            return True
    except Exception:
        return False
    return False

def _stop_tk_vtk_event_pump(tk_master: Any) -> None:
    """Cancel VTK event pumping for a separate PyVista window coordinated with Tk."""
    aid = _VTK_TK_PUMP.get("after_id")
    if tk is not None and tk_master is not None and aid is not None:
        try:
            tk_master.after_cancel(aid)
        except (tk.TclError, ValueError, TypeError):
            pass
    _VTK_TK_PUMP["after_id"] = None
    prev = _VTK_TK_PUMP.get("plotter")
    _VTK_TK_PUMP["plotter"] = None
    if prev is not None:
        try:
            prev.close()
        except Exception:
            pass

def _ensure_pyvista_iren_initialized(plotter: Any) -> None:
    """``Plotter.show(interactive_update=True)`` skips ``iren.initialize()`` on VTK 9.2.3+, but
    :meth:`RenderWindowInteractor.process_events` requires an initialized interactor."""
    try:
        iren_wrap = getattr(plotter, "iren", None)
        if iren_wrap is None:
            return
        if not bool(getattr(iren_wrap, "initialized", False)):
            iren_wrap.initialize()
    except Exception:
        pass

def _start_tk_vtk_event_pump(tk_master: Any, plotter: Any) -> None:
    """Drive a non-embedded PyVista window while Tk's mainloop runs (``interactive_update``
    mode)."""
    if tk is None:
        return
    _stop_tk_vtk_event_pump(tk_master)
    _VTK_TK_PUMP["plotter"] = plotter

    def pump() -> None:
        plr = _VTK_TK_PUMP.get("plotter")
        if plr is None:
            _VTK_TK_PUMP["after_id"] = None
            return
        rw = getattr(plr, "render_window", None)
        if rw is None:
            _VTK_TK_PUMP["after_id"] = None
            _VTK_TK_PUMP["plotter"] = None
            return
        try:
            if plr.iren is not None:
                if not bool(getattr(plr.iren, "initialized", False)):
                    _ensure_pyvista_iren_initialized(plr)
                plr.iren.process_events()
        except Exception:
            pass
        try:
            if rw is not None:
                rw.Render()
        except Exception:
            pass
        try:
            _VTK_TK_PUMP["after_id"] = tk_master.after(33, pump)
        except (tk.TclError, RuntimeError):
            _VTK_TK_PUMP["after_id"] = None

    _VTK_TK_PUMP["after_id"] = tk_master.after(33, pump)

def _show_tk_vtk_fallback_panel(parent: Any) -> None:
    """Explain separate-window 3D when ``vtkRenderingTk`` is not in the VTK wheel."""
    if tk is None:
        return
    inner = tk.Frame(parent, bg="#0d1117")
    inner.pack(fill=tk.BOTH, expand=True)
    msg = (
        "3D view is open in a separate window.\n\n"
        "This Python environment’s VTK build does not ship vtkRenderingTk (typical for "
        "`pip install vtk` on Windows), so the renderer cannot be embedded in this pane.\n\n"
        "Options: use conda-forge VTK built with Tk, or keep using the separate window — "
        "slice controls in the drawer still apply after Show 3D."
    )
    tk.Label(
        inner,
        text=msg,
        bg="#0d1117",
        fg="#8b949e",
        font=("", 10),
        justify=tk.LEFT,
        wraplength=420,
    ).pack(anchor="n", padx=16, pady=20)

def idle_slice_band_controls(slice_tk: dict[str, Any] | None) -> None:
    """Disable 3D layer-band widgets until a plot exists (Tk GUI)."""
    if slice_tk is None or tk is None:
        return
    try:
        slice_tk["scale"].configure(state=tk.DISABLED)
        slice_tk["checkbtn"].state(["disabled"])
        slice_tk["status_var"].set("Run Show 3D to enable the layer band.")
        slice_tk["var_slice"].set(False)
    except (tk.TclError, KeyError):
        pass

def _wire_slice_band_controls(
    slice_tk: dict[str, Any],
    slice_cfg: dict[str, bool | int],
    layer_energies_plan: list[float],
    n_plan_layers: int,
    apply_slice: Any,
) -> None:
    if tk is None or n_plan_layers <= 0:
        return
    var_slice = slice_tk["var_slice"]
    scale = slice_tk["scale"]
    chk = slice_tk["checkbtn"]
    status_var = slice_tk["status_var"]

    def band_line() -> str:
        ci = int(slice_cfg["center_i"])
        emid = float(layer_energies_plan[ci])
        if not bool(slice_cfg["slice_on"]):
            return (
                f"Full stack: {n_plan_layers} nominal layer(s). "
                f"Slider ref — index {ci}, {emid:.2f} MeV (plan order)."
            )
        lo_m, hi_m = _nominal_layer_index_band_mev(layer_energies_plan, ci, half_width=2)
        return f"5-layer band: [{lo_m:.2f}, {hi_m:.2f}] MeV (center idx {ci}, {emid:.2f} MeV)."

    def refresh() -> None:
        status_var.set(band_line())

    def on_scale(val: str) -> None:
        slice_cfg["center_i"] = int(np.clip(int(round(float(val))), 0, n_plan_layers - 1))
        if bool(slice_cfg["slice_on"]):
            apply_slice()
        refresh()

    def on_chk() -> None:
        slice_cfg["slice_on"] = bool(var_slice.get())
        apply_slice()
        refresh()

    var_slice.set(bool(slice_cfg["slice_on"]))
    scale.configure(
        from_=0,
        to=max(0, n_plan_layers - 1),
        resolution=1,
        state=tk.NORMAL,
    )
    scale.set(int(slice_cfg["center_i"]))
    chk.state(["!disabled"])
    scale.configure(command=on_scale)
    chk.configure(command=on_chk)
    refresh()

def _embed_pyvista_plotter_in_tk(parent: Any, plotter: Any) -> Any:
    """Place an existing PyVista plotter's render window inside a Tk container (~left pane)."""
    from vtkmodules.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor

    if tk is None:
        raise RuntimeError("tkinter is required to embed the PyVista plotter")
    parent.update_idletasks()
    w = max(parent.winfo_width(), 320)
    h = max(parent.winfo_height(), 240)
    inner = tk.Frame(parent, bg="#0d1117")
    inner.pack(fill=tk.BOTH, expand=True)
    rw = plotter.render_window
    iren = vtkTkRenderWindowInteractor(inner, rw=rw, width=w, height=h)
    iren.pack(fill=tk.BOTH, expand=True)
    iren.Initialize()
    rw.SetInteractor(iren)
    if getattr(plotter, "iren", None) is not None:
        plotter.iren.interactor = iren
    plotter.render()

    def _on_resize(event: tk.Event) -> None:
        if event.widget is not parent:
            return
        nw = max(int(event.width), 2)
        nh = max(int(event.height), 2)
        try:
            rw.SetSize(nw, nh)
            plotter.render()
        except Exception:
            pass

    parent.bind("<Configure>", _on_resize)
    return iren

def _clear_qt_layout_items(parent: Any) -> None:
    """Remove all widgets from ``parent``'s ``QVBoxLayout`` (Qt embed pane)."""
    lay = parent.layout()
    if lay is None:
        return
    while lay.count():
        item = lay.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()

def _embed_pyvista_plotter_in_qt(parent: Any, plotter: Any) -> Any:
    """Place PyVista's render window in a Qt widget (works with pip VTK on Windows)."""
    from PySide6.QtWidgets import QVBoxLayout
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    _clear_qt_layout_items(parent)
    lay = parent.layout()
    if lay is None:
        lay = QVBoxLayout(parent)
        lay.setContentsMargins(0, 0, 0, 0)
        parent.setLayout(lay)
    rw = plotter.render_window
    vtk_widget = QVTKRenderWindowInteractor(parent, rw=rw)
    lay.addWidget(vtk_widget)
    if getattr(plotter, "iren", None) is not None:
        plotter.iren.interactor = vtk_widget._Iren
    vtk_widget.Initialize()
    # QVTK replaces the vtkRenderWindowInteractor. PyVista's theme style was applied to the
    # old interactor; without re-applying, VTK's default (often joystick camera) feels broken.
    try:
        plotter.enable_interactor_style()
    except Exception:
        try:
            plotter.enable_trackball_style()
        except Exception:
            pass
    # Without show(), Plotter.render() skips vtk Render() until this runs (see plotter._first_time).
    try:
        plotter._on_first_render_request()
    except Exception:
        pass
    plotter.render()
    return vtk_widget

def apply_comparison_3d_camera_view(
    plotter: Any,
    view: str,
    *,
    zoom: float = 1.05,
    render: bool = True,
) -> None:
    """Snap camera to a standard view: ``top`` (XY), ``left`` (−X), or ``right`` (+X)."""
    v = str(view).strip().lower()
    if v == "top":
        plotter.view_xy()
    elif v == "left":
        plotter.view_yz(negative=True)
    elif v == "right":
        plotter.view_yz(negative=False)
    else:
        raise ValueError(f"unknown 3D view {view!r} (expected top, left, or right)")
    try:
        plotter.camera.zoom(float(zoom))
    except Exception:
        pass
    if render:
        plotter.render()

def disconnect_slice_band_controls_qt(slice_qt: dict[str, Any] | None) -> None:
    """Stop Qt slice callbacks from a prior plot while a new 3D view is building."""
    if slice_qt is None:
        return
    chk: Any = slice_qt.get("check")
    sli: Any = slice_qt.get("slider")
    prev_chk = slice_qt.get("_slice_chk_handler")
    prev_sli = slice_qt.get("_slice_sli_handler")
    if chk is not None and prev_chk is not None:
        try:
            chk.toggled.disconnect(prev_chk)
        except (TypeError, RuntimeError):
            pass
    if sli is not None and prev_sli is not None:
        try:
            sli.valueChanged.disconnect(prev_sli)
        except (TypeError, RuntimeError):
            pass
    slice_qt.pop("_slice_chk_handler", None)
    slice_qt.pop("_slice_sli_handler", None)


def idle_slice_band_controls_qt(slice_qt: dict[str, Any] | None) -> None:
    if slice_qt is None:
        return
    disconnect_slice_band_controls_qt(slice_qt)
    try:
        slice_qt["slider"].setEnabled(False)
        slice_qt["check"].setEnabled(False)
        slice_qt["status"].setText("Layer band enables after a successful 3D plot.")
    except Exception:
        pass

def _wire_slice_band_controls_qt(
    slice_qt: dict[str, Any],
    slice_cfg: dict[str, bool | int],
    layer_energies_plan: list[float],
    n_plan_layers: int,
    apply_slice: Any,
) -> None:
    chk: Any = slice_qt["check"]
    sli: Any = slice_qt["slider"]
    status: Any = slice_qt["status"]
    if n_plan_layers <= 0:
        return

    def band_line() -> str:
        ci = int(slice_cfg["center_i"])
        emid = float(layer_energies_plan[ci])
        if not bool(slice_cfg["slice_on"]):
            return (
                f"Full stack: {n_plan_layers} nominal layer(s). "
                f"Slider ref — index {ci}, {emid:.2f} MeV (plan order)."
            )
        lo_m, hi_m = _nominal_layer_index_band_mev(layer_energies_plan, ci, half_width=2)
        return f"5-layer band: [{lo_m:.2f}, {hi_m:.2f}] MeV (center idx {ci}, {emid:.2f} MeV)."

    def refresh() -> None:
        status.setText(band_line())

    def on_sli(val: int) -> None:
        slice_cfg["center_i"] = int(np.clip(int(val), 0, n_plan_layers - 1))
        if bool(slice_cfg["slice_on"]):
            apply_slice()
        refresh()

    def on_chk(checked: bool) -> None:
        slice_cfg["slice_on"] = bool(checked)
        try:
            apply_slice(refit_camera=True)
        except TypeError:
            apply_slice()
        refresh()

    disconnect_slice_band_controls_qt(slice_qt)

    sli.setMinimum(0)
    sli.setMaximum(max(0, n_plan_layers - 1))
    sli.setSingleStep(1)
    try:
        sli.setTracking(True)
    except Exception:
        pass
    sli.blockSignals(True)
    sli.setValue(int(slice_cfg["center_i"]))
    sli.blockSignals(False)
    chk.blockSignals(True)
    chk.setChecked(bool(slice_cfg["slice_on"]))
    chk.blockSignals(False)
    chk.setEnabled(True)
    sli.setEnabled(True)
    slice_qt["_slice_sli_handler"] = on_sli
    slice_qt["_slice_chk_handler"] = on_chk
    sli.valueChanged.connect(on_sli)
    chk.toggled.connect(on_chk)
    refresh()
