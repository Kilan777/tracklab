"""Kinematics plot: one quantity at a time (single y-axis), synced to the video playhead."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QVBoxLayout, QWidget)

if TYPE_CHECKING:
    from .main import MainWindow

SURFACE = "#14171d"
INK = "#cfd6e4"
MUTED = "#7d8799"

# key, label, source kind, unit builder
QUANTITIES = [
    ("x", "Position X", "track", lambda u: u),
    ("y", "Position Y", "track", lambda u: u),
    ("speed", "Speed", "track", lambda u: f"{u}/s"),
    ("accel", "Acceleration", "track", lambda u: f"{u}/s²"),
    ("vx", "Velocity X", "track", lambda u: f"{u}/s"),
    ("vy", "Velocity Y", "track", lambda u: f"{u}/s"),
    ("ax", "Acceleration X", "track", lambda u: f"{u}/s²"),
    ("ay", "Acceleration Y", "track", lambda u: f"{u}/s²"),
    ("angle", "Angle", "angle", lambda u: "°"),
    ("angle_rate", "Angular velocity", "angle", lambda u: "°/s"),
    ("distance", "Distance", "distance", lambda u: u),
    ("distance_rate", "Distance rate", "distance", lambda u: f"{u}/s"),
]

pg.setConfigOptions(antialias=True, background=SURFACE, foreground=MUTED)


class PlotPanel(QWidget):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        self._hidden: set[int] = set()  # ids unticked by the user (survives rebuilds)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)

        side = QVBoxLayout()
        self.qty = QComboBox()
        for k, name, *_ in QUANTITIES:
            self.qty.addItem(name, k)
        self.qty.currentIndexChanged.connect(self.rebuild)
        side.addWidget(self.qty)
        self.items = QListWidget()
        self.items.setMaximumWidth(190)
        self.items.itemChanged.connect(self._item_changed)
        side.addWidget(self.items, 1)
        self.raw = QCheckBox("Show raw samples")
        self.raw.toggled.connect(self.redraw)
        side.addWidget(self.raw)
        self.cut = QLabel()
        self.cut.setStyleSheet(f"color:{MUTED}; font-size:11px")
        self.cut.setWordWrap(True)
        side.addWidget(self.cut)
        sw = QWidget()
        sw.setLayout(side)
        sw.setMaximumWidth(200)
        lay.addWidget(sw)

        self.pw = pg.PlotWidget()
        self.pw.showGrid(x=True, y=True, alpha=0.12)
        self.pw.setLabel("bottom", "time (s)")
        for ax in ("left", "bottom"):
            self.pw.getAxis(ax).enableAutoSIPrefix(False)
        self.pw.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.legend = self.pw.addLegend(offset=(10, 6), labelTextColor=INK, brush=pg.mkBrush(20, 23, 29, 200))
        self.playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen("#4da3ff", width=1.5))
        self.hover = pg.InfiniteLine(angle=90, pen=pg.mkPen(MUTED, width=1, style=Qt.DashLine))
        self.readout = pg.TextItem(color=INK, anchor=(0, 0), fill=pg.mkBrush(13, 15, 19, 220))
        self.readout.setZValue(10)
        lay.addWidget(self.pw, 1)
        self.pw.scene().sigMouseMoved.connect(self._mouse_moved)
        self.pw.scene().sigMouseClicked.connect(self._mouse_clicked)
        self._curves: list[tuple[str, str, np.ndarray]] = []
        self.rebuild()

    def show(self, key: str, names: list[str] | None = None, raw: bool | None = None) -> None:
        """Switch the graph to quantity `key`, showing only the named points/measurements (tours use this)."""
        idx = next((i for i, q in enumerate(QUANTITIES) if q[0] == key), None)
        if idx is None:
            return
        p = self.app.project
        if names is not None:
            keep = {o.id for o in (p.by_name(n) for n in names) if o is not None}
            every = {t.id for t in p.tracks} | {m.id for m in p.measures}
            self._hidden = every - keep
        if raw is not None:
            self.raw.blockSignals(True)
            self.raw.setChecked(raw)
            self.raw.blockSignals(False)
        if self.qty.currentIndex() == idx:
            self.rebuild()
        else:
            self.qty.setCurrentIndex(idx)

    def quantity(self):
        return QUANTITIES[self.qty.currentIndex()]

    def rebuild(self) -> None:
        """Re-list sources for the chosen quantity (tracks or measures), then redraw."""
        kind = self.quantity()[2]
        p = self.app.project
        self.items.blockSignals(True)
        self.items.clear()
        if kind == "track":
            srcs = [(t.id, t.name, t.color) for t in p.tracks if not t.static]
        else:
            kinds = ("angle", "segment") if kind == "angle" else ("distance",)
            srcs = [(m.id, m.name, m.color) for m in p.measures if m.kind in kinds]
        for sid, name, color in srcs:
            it = QListWidgetItem(name)
            it.setData(Qt.UserRole, sid)
            it.setForeground(QColor(color))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked if sid in self._hidden else Qt.Checked)
            self.items.addItem(it)
        if not srcs:
            it = QListWidgetItem({"track": "Add points to plot them", "angle": "Add an angle (A / G)",
                                  "distance": "Add a distance (D)"}[kind])
            it.setFlags(Qt.NoItemFlags)
            self.items.addItem(it)
        self.items.blockSignals(False)
        self.redraw()

    def _item_changed(self, it: QListWidgetItem) -> None:
        sid = it.data(Qt.UserRole)
        if it.checkState() == Qt.Checked:
            self._hidden.discard(sid)
        else:
            self._hidden.add(sid)
        self.redraw()

    def redraw(self) -> None:
        pw = self.pw
        pw.clear()
        self.legend.clear()
        self._curves = []
        kin = self.app.kin()
        key, name, kind, unitf = self.quantity()
        if kin is None:
            return
        unit = unitf(kin.unit)
        pw.setLabel("left", f"{name} ({unit})")
        t = kin.t
        p = self.app.project
        for i in range(self.items.count()):
            it = self.items.item(i)
            sid = it.data(Qt.UserRole)
            if sid is None or it.checkState() != Qt.Checked:
                continue
            if kind == "track":
                tr = p.track(sid)
                d = kin.tracks.get(sid)
                if d is None or tr is None:
                    continue
                y, color, label = d[key], tr.color, tr.name
                if self.raw.isChecked() and key in ("x", "y"):
                    raw = d[f"{key}_raw"]
                    pw.plot(t, raw, pen=None, symbol="o", symbolSize=4, symbolPen=None,
                            symbolBrush=pg.mkBrush(QColor(color).lighter(130)))
            else:
                me = next((m for m in p.measures if m.id == sid), None)
                d = kin.measures.get(sid)
                if d is None or me is None:
                    continue
                y = d["rate" if key.endswith("_rate") else "value"]
                color, label = me.color, me.name
            ok = np.flatnonzero(~np.isnan(y))
            if ok.size:  # plot only the tracked span so the time axis isn't mostly empty
                a, b = ok[0], ok[-1] + 1
                pw.plot(t[a:b], y[a:b], pen=pg.mkPen(color, width=2), name=label, connect="finite")
            self._curves.append((label, color, y))
        vb = pw.getPlotItem().getViewBox()
        vb.enableAutoRange()
        if key in ("speed", "accel", "distance") and self._curves:
            # magnitudes: start the axis at zero so small wobbles don't look dramatic
            top = max((float(np.nanmax(c[2])) for c in self._curves if np.isfinite(c[2]).any()), default=1.0)
            vb.setYRange(0, top * 1.1 if top > 0 else 1.0)
        pw.addItem(self.playhead, ignoreBounds=True)
        pw.addItem(self.hover, ignoreBounds=True)
        pw.addItem(self.readout, ignoreBounds=True)
        self.hover.hide()
        self.readout.hide()
        self.sync_playhead()
        cuts = {k: v for k, v in kin.cutoff_used.items() if v > 0}
        if p.filter_hz < 0:
            self.cut.setText("Filter: off (raw)")
        elif cuts:
            vals = sorted(cuts.values())
            self.cut.setText(f"Low-pass {vals[0]:.1f}–{vals[-1]:.1f} Hz" if p.filter_hz == 0 and len(vals) > 1
                             else f"Low-pass {vals[0]:.1f} Hz")
        else:
            self.cut.setText("")

    def sync_playhead(self) -> None:
        kin = self.app.kin()
        if kin is not None:
            self.playhead.setValue(self.app.frame * kin.dt)

    def _mouse_moved(self, pos) -> None:
        vb = self.pw.getPlotItem().getViewBox()
        kin = self.app.kin()
        if kin is None or not self.pw.sceneBoundingRect().contains(pos) or not self._curves:
            self.hover.hide()
            self.readout.hide()
            return
        mp = vb.mapSceneToView(pos)
        f = int(round(mp.x() / kin.dt))
        if not (0 <= f < kin.n):
            self.hover.hide()
            self.readout.hide()
            return
        self.hover.setValue(f * kin.dt)
        self.hover.show()
        unit = self.quantity()[3](kin.unit)
        rows = [f"<span style='color:{MUTED}'>t = {f * kin.dt:.3f} s · frame {f}</span>"]
        for label, color, y in self._curves:
            v = y[f]
            val = "—" if np.isnan(v) else f"{v:.4g} {unit}"
            rows.append(f"<span style='color:{color}'>●</span> {label}: <b>{val}</b>")
        self.readout.setHtml("<div style='padding:3px'>" + "<br>".join(rows) + "</div>")
        (x0, x1), (y0, y1) = vb.viewRange()
        right = mp.x() > (x0 + x1) / 2
        self.readout.setAnchor((1, 0) if right else (0, 0))
        self.readout.setPos(mp.x() + (-1 if right else 1) * (x1 - x0) * 0.01, y1)
        self.readout.show()

    def _mouse_clicked(self, ev) -> None:
        if ev.button() != Qt.LeftButton or ev.double():
            if ev.double():
                self.pw.getPlotItem().getViewBox().autoRange()
            return
        kin = self.app.kin()
        if kin is None:
            return
        mp = self.pw.getPlotItem().getViewBox().mapSceneToView(ev.scenePos())
        self.app.seek(int(round(mp.x() / kin.dt)))
