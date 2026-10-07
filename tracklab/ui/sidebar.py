"""Right-hand panels: points, measurements, calibration, tracking + analysis settings, live readout."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ..model import MANUAL
from ..render import fmt, measure_now
from ..tracker import auto_radii
from .guide import Guide
from .tour import TourPanel

if TYPE_CHECKING:
    from .main import MainWindow

UNITS = ["m", "cm", "mm", "in", "ft", "yd"]


def swatch(color: str) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(QColor(color))
    return QIcon(pm)


def _btn(text: str, tip: str, cb) -> QPushButton:
    b = QPushButton(text)
    b.setToolTip(tip)
    b.clicked.connect(cb)
    return b


class Sidebar(QScrollArea):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        self._busy = False
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(300)
        self.setMaximumWidth(420)
        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)
        self.tour = TourPanel(app)  # pinned above the scroll area by MainWindow, so it never scrolls away
        self.guide = Guide(app)
        lay.addWidget(self.guide)
        # kept as attributes so tours can point at them
        self.g_points = self._points_box()
        self.g_measures = self._measures_box()
        self.g_readout = self._readout_box()
        self.g_calib = self._calib_box()
        self.g_tracking = self._tracking_box()
        self.g_time = self._analysis_box()
        for g in (self.g_points, self.g_measures, self.g_readout, self.g_calib, self.g_tracking, self.g_time):
            lay.addWidget(g)
        lay.addStretch(1)
        self.setWidget(root)
        for sp in root.findChildren(QDoubleSpinBox) + root.findChildren(QSpinBox):
            sp.setMinimumWidth(70)  # big ranges otherwise make the panel too wide
        for cb in root.findChildren(QComboBox):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(6)

    # ---------------- points ----------------
    def _points_box(self) -> QGroupBox:
        g = QGroupBox("Points")
        v = QVBoxLayout(g)
        self.tracks = QListWidget()
        self.tracks.setMaximumHeight(150)
        self.tracks.currentItemChanged.connect(self._track_selected)
        self.tracks.itemChanged.connect(self._track_item_changed)
        v.addWidget(self.tracks)
        row = QHBoxLayout()
        row.addWidget(_btn("+ Add", "Point tool (P): click on the video to add a point", lambda: self.app.set_tool("point")))
        row.addWidget(_btn("Place here", "Click once on the video to put the selected point there on this frame (Q)",
                           self.app.arm_place))
        row.addWidget(_btn("Delete", "Delete the selected point's whole track (Shift+Del)", self.app.delete_selected_track))
        v.addLayout(row)
        self.props = QWidget()
        f = QFormLayout(self.props)
        f.setContentsMargins(0, 4, 0, 0)
        self.name = QLineEdit()
        self.name.editingFinished.connect(self._prop_changed)
        f.addRow("Name", self.name)
        self.color = QPushButton()
        self.color.clicked.connect(self._pick_color)
        f.addRow("Color", self.color)
        self.method = QComboBox()
        self.method.addItem("Template — precise, markers/features", "template")
        self.method.addItem("CSRT — robust to rotation/shape change", "csrt")
        self.method.addItem("Blob — follows a colour (balls, tumbling things)", "blob")
        self.method.currentIndexChanged.connect(self._prop_changed)
        f.addRow("Tracker", self.method)
        self.tsize = QSpinBox()
        self.tsize.setRange(0, 200)
        self.tsize.setSpecialValueText("auto")
        self.tsize.setSuffix(" px")
        self.tsize.setToolTip("Half-size of the patch that is matched (solid box). Make it hug the marker.")
        self.tsize.valueChanged.connect(self._prop_changed)
        f.addRow("Template", self.tsize)
        self.ssize = QSpinBox()
        self.ssize.setRange(0, 600)
        self.ssize.setSpecialValueText("auto")
        self.ssize.setSuffix(" px")
        self.ssize.setToolTip("How far it searches per frame (dotted box). Grows automatically with speed.")
        self.ssize.valueChanged.connect(self._prop_changed)
        f.addRow("Search", self.ssize)
        self.static = QCheckBox("Fixed point (same position every frame)")
        self.static.toggled.connect(self._prop_changed)
        f.addRow(self.static)
        self.info = QLabel()
        self.info.setStyleSheet("color:#7d8799")
        self.info.setWordWrap(True)
        f.addRow(self.info)
        v.addWidget(self.props)
        return g

    def _track_selected(self, cur, prev) -> None:
        if self._busy or cur is None:
            return
        self.app.select_track(cur.data(Qt.UserRole))

    def _track_item_changed(self, it: QListWidgetItem) -> None:
        if self._busy:
            return
        t = self.app.project.track(it.data(Qt.UserRole))
        if t:
            t.visible = it.checkState() == Qt.Checked
            self.app.changed(data=False)

    def _prop_changed(self) -> None:
        if self._busy:
            return
        t = self.app.project.track(self.app.selected)
        if not t:
            return
        self.app.push_undo()
        name = self.name.text().strip() or t.name
        renamed = name != t.name
        t.name = name
        t.method = self.method.currentData()
        t.template_r = self.tsize.value()
        t.search_r = self.ssize.value()
        if self.static.isChecked() != t.static:
            p = t.at(self.app.frame) or (t.at(t.frames()[0]) if t.pts else None)
            t.static = self.static.isChecked()
            t.pts = {0: (p[0], p[1], 1.0, MANUAL)} if p else {}
        if renamed:
            for m in self.app.project.measures:
                if t.id in m.refs:
                    m.name = m.name.split(" ")[0] + " " + "-".join(
                        self.app.project.track(r).name for r in m.refs if self.app.project.track(r))
        self.app.changed()

    def _pick_color(self) -> None:
        t = self.app.project.track(self.app.selected)
        if not t:
            return
        c = QColorDialog.getColor(QColor(t.color), self, "Point color")
        if c.isValid():
            self.app.push_undo()
            t.color = c.name()
            self.app.changed(data=False)

    # ---------------- measures ----------------
    def _measures_box(self) -> QGroupBox:
        g = QGroupBox("Measurements")
        v = QVBoxLayout(g)
        self.measures = QListWidget()
        self.measures.setMaximumHeight(110)
        self.measures.itemChanged.connect(self._measure_item_changed)
        self.measures.currentItemChanged.connect(lambda *_: self._sync_signed())
        v.addWidget(self.measures)
        row = QGridLayout()
        row.addWidget(_btn("∠ Angle", "A: click 3 points — end, vertex, end", lambda: self.app.set_tool("angle")), 0, 0)
        row.addWidget(_btn("⦣ Tilt", "G: click 2 points — angle of the segment vs the x-axis", lambda: self.app.set_tool("segment")), 0, 1)
        row.addWidget(_btn("↔ Distance", "D: click 2 points", lambda: self.app.set_tool("distance")), 0, 2)
        row.addWidget(_btn("Delete", "Delete the selected measurement", self._delete_measure), 1, 2)
        self.signed = QCheckBox("0–360°")
        self.signed.setToolTip("Angle measured one way round (0–360°) instead of the inside angle (0–180°)")
        self.signed.toggled.connect(self._signed_changed)
        row.addWidget(self.signed, 1, 0, 1, 2)
        v.addLayout(row)
        hint = QLabel("Click existing points to measure between them; clicking empty space adds a fixed point.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#7d8799; font-size:11px")
        v.addWidget(hint)
        return g

    def _measure_item_changed(self, it: QListWidgetItem) -> None:
        if self._busy:
            return
        m = next((m for m in self.app.project.measures if m.id == it.data(Qt.UserRole)), None)
        if m:
            m.visible = it.checkState() == Qt.Checked
            self.app.changed(data=False)

    def _current_measure(self):
        it = self.measures.currentItem()
        if not it:
            return None
        return next((m for m in self.app.project.measures if m.id == it.data(Qt.UserRole)), None)

    def _delete_measure(self) -> None:
        m = self._current_measure()
        if m:
            self.app.push_undo()
            self.app.project.measures.remove(m)
            self.app.changed()

    def _sync_signed(self) -> None:
        m = self._current_measure()
        self._busy = True
        self.signed.setEnabled(bool(m and m.kind == "angle"))
        self.signed.setChecked(bool(m and m.signed))
        self._busy = False

    def _signed_changed(self, on: bool) -> None:
        m = self._current_measure()
        if self._busy or not m:
            return
        self.app.push_undo()
        m.signed = on
        self.app.changed()

    # ---------------- readout ----------------
    def _readout_box(self) -> QGroupBox:
        g = QGroupBox("At this frame")
        v = QVBoxLayout(g)
        self.readout = QLabel()
        self.readout.setTextFormat(Qt.RichText)
        self.readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.readout.setWordWrap(True)
        v.addWidget(self.readout)
        return g

    def update_readout(self) -> None:
        app = self.app
        p = app.project
        f = app.frame
        kin = app.kin() if not app.tracking else None
        u = p.calib.unit_label()
        rows = []
        for t in p.tracks:
            pt = p.pos(t, f)
            if pt is None:
                continue
            w = p.world(np.array([pt]), f)[0]
            if np.isnan(w).any():
                continue
            line = f"<span style='color:{t.color}'>●</span> <b>{t.name}</b> &nbsp;x {w[0]:.4g} &nbsp;y {w[1]:.4g} {u}"
            if kin and not t.static:
                d = kin.tracks.get(t.id)
                if d is not None and not math.isnan(d["speed"][f]):
                    line += f"<br>&nbsp;&nbsp;&nbsp;v {d['speed'][f]:.4g} {u}/s &nbsp;a {d['accel'][f]:.4g} {u}/s²"
            rows.append(line)
        for m in p.measures:
            v = measure_now(p, m, f)
            if v is None:
                continue
            val = fmt(v, u) if m.kind == "distance" else f"{v:.1f}°"
            rows.append(f"<span style='color:{m.color}'>●</span> {m.name}: <b>{val}</b>")
        self.readout.setText("<br>".join(rows) if rows else "<span style='color:#7d8799'>Nothing on this frame yet.</span>")

    # ---------------- calibration ----------------
    def _calib_box(self) -> QGroupBox:
        g = QGroupBox("Calibration")
        v = QVBoxLayout(g)
        self.cal_state = QLabel()
        self.cal_state.setWordWrap(True)
        v.addWidget(self.cal_state)
        row = QGridLayout()
        row.addWidget(_btn("Scale line", "C: click both ends of something of known length", lambda: self.app.set_tool("line")), 0, 0)
        row.addWidget(_btn("Plane", "Click the 4 corners of a known rectangle on the floor/wall (TL, TR, BR, BL) — "
                           "corrects perspective", lambda: self.app.set_tool("plane")), 0, 1)
        row.addWidget(_btn("Wheel", "Click top, front, bottom, back of a wheel's tyre and give its diameter — corrects for "
                           "a bike/object filmed at an angle instead of exactly side-on", lambda: self.app.set_tool("circle")), 0, 2)
        row.addWidget(_btn("Origin", "O: click where (0,0) should be", lambda: self.app.set_tool("origin")), 1, 0)
        row.addWidget(_btn("Level to line", "Make the scale line the x-axis direction (e.g. calibrated along the floor)",
                           self._level), 1, 1)
        row.addWidget(_btn("Clear", "Remove calibration", self._clear_calib), 1, 2)
        row.addWidget(_btn("Scale from gravity", "No ruler in shot? Select a point that's flying freely (thrown ball, "
                           "jumper in the air), put the playhead in that flight (or set [ ] around it) and click: the "
                           "fall's curve is 9.81 m/s², which gives the scale and which way is down.",
                           self.app.calibrate_from_gravity), 2, 0, 1, 3)
        v.addLayout(row)
        self.stab = QCheckBox("Camera moves (handheld) — follow it")
        self.stab.setToolTip("Measures how the camera pans/zooms on every frame and cancels it out, so positions, speeds "
                             "and the calibration stay attached to the scene. Needs a calibration first.")
        self.stab.toggled.connect(lambda on: None if self._busy else self.app.set_stabilize(on))
        v.addWidget(self.stab)
        self.stab_state = QLabel()
        self.stab_state.setWordWrap(True)
        self.stab_state.setStyleSheet("color:#8fa0bb; font-size:11px")
        v.addWidget(self.stab_state)
        f = QFormLayout()
        self.cal_len = QDoubleSpinBox()
        self.cal_len.setRange(1e-6, 1e9)
        self.cal_len.setDecimals(4)
        self.cal_len.valueChanged.connect(self._calib_changed)
        f.addRow("Line length", self.cal_len)
        wh = QHBoxLayout()
        self.cal_w = QDoubleSpinBox()
        self.cal_h = QDoubleSpinBox()
        for s in (self.cal_w, self.cal_h):
            s.setRange(1e-6, 1e9)
            s.setDecimals(4)
            s.valueChanged.connect(self._calib_changed)
            wh.addWidget(s)
        f.addRow("Plane W × H", wh)
        self.cal_d = QDoubleSpinBox()
        self.cal_d.setRange(1e-6, 1e9)
        self.cal_d.setDecimals(4)
        self.cal_d.setToolTip("Outside diameter of the wheel incl. tyre (road 700c ≈ 0.67–0.68 m, MTB 29\" ≈ 0.74 m)")
        self.cal_d.valueChanged.connect(self._calib_changed)
        f.addRow("Wheel ⌀", self.cal_d)
        self.cal_unit = QComboBox()
        self.cal_unit.addItems(UNITS)
        self.cal_unit.currentTextChanged.connect(self._calib_changed)
        f.addRow("Unit", self.cal_unit)
        self.axis = QDoubleSpinBox()
        self.axis.setRange(-180, 180)
        self.axis.setDecimals(2)
        self.axis.setSuffix("°")
        self.axis.setToolTip("Rotate the x-axis (or drag the red arrow tip on the video)")
        self.axis.valueChanged.connect(self._calib_changed)
        f.addRow("Axis rotation", self.axis)
        self.yup = QCheckBox("Y points up")
        self.yup.toggled.connect(self._calib_changed)
        f.addRow(self.yup)
        v.addLayout(f)
        return g

    def _calib_changed(self, *_) -> None:
        if self._busy:
            return
        c = self.app.project.calib
        self.app.push_undo(coalesce="calib")
        c.length = self.cal_len.value()
        c.width, c.height = self.cal_w.value(), self.cal_h.value()
        c.diameter = self.cal_d.value()
        c.unit = self.cal_unit.currentText()
        c.axis_angle = -math.radians(self.axis.value())
        c.y_up = self.yup.isChecked()
        self.app.changed()

    def _level(self) -> None:
        c = self.app.project.calib
        if c.mode != "line" or len(c.line) != 2:
            self.app.status("Set a scale line first (C).")
            return
        self.app.push_undo()
        (x0, y0), (x1, y1) = c.line
        if x1 < x0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        c.axis_angle = math.atan2(y1 - y0, x1 - x0)
        if c.origin is None:
            c.origin = [x0, y0]
        self.app.changed()

    def _clear_calib(self) -> None:
        from ..model import Calibration

        self.app.push_undo()
        self.app.project.calib = Calibration(origin=self.app.project.calib.origin)
        self.app.changed()

    # ---------------- tracking ----------------
    def _tracking_box(self) -> QGroupBox:
        g = QGroupBox("Tracking")
        f = QFormLayout(g)
        o = self.app.opts
        self.thr = QDoubleSpinBox()
        self.thr.setRange(0.2, 0.98)
        self.thr.setSingleStep(0.05)
        self.thr.setValue(o["threshold"])
        self.thr.setToolTip("Minimum match score. Higher = stops sooner when unsure; lower = pushes through blur.")
        self.thr.valueChanged.connect(lambda v: self.app.set_opt("threshold", v))
        f.addRow("Min. match", self.thr)
        self.bridge = QSpinBox()
        self.bridge.setRange(0, 120)
        self.bridge.setSuffix(" frames")
        self.bridge.setValue(o["bridge"])
        self.bridge.setToolTip("Coast through short occlusions and pick the point back up. Hidden frames stay empty.")
        self.bridge.valueChanged.connect(lambda v: self.app.set_opt("bridge", v))
        f.addRow("Bridge gaps ≤", self.bridge)
        for key, text, tip in (
            ("stop_on_loss", "Stop everything when a point is lost", "Otherwise the other points keep going."),
            ("retrack", "Re-track after a correction", "Dragging a point (or placing a lost one) re-runs tracking from there."),
            ("snap", "Snap clicks onto markers", "Clicks lock onto the nearest blob/corner. Hold Shift to place exactly."),
        ):
            cb = QCheckBox(text)
            cb.setToolTip(tip)
            cb.setChecked(o[key])
            cb.toggled.connect(lambda v, k=key: self.app.set_opt(k, v))
            f.addRow(cb)
        return g

    # ---------------- analysis ----------------
    def _analysis_box(self) -> QGroupBox:
        g = QGroupBox("Time && filtering")
        f = QFormLayout(g)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0, 100000)
        self.fps.setDecimals(2)
        self.fps.setSpecialValueText("file")
        self.fps.setSuffix(" fps")
        self.fps.setToolTip("Real capture rate. Set this for slow-motion footage (e.g. 240) so time and speed are right.")
        self.fps.valueChanged.connect(self._analysis_changed)
        f.addRow("Capture rate", self.fps)
        fr = QHBoxLayout()
        self.filt = QComboBox()
        self.filt.addItems(["Auto", "Manual", "Off"])
        self.filt.setMinimumWidth(70)
        self.filt.setToolTip("Zero-lag Butterworth low-pass before differentiating. Auto picks the cutoff by residual analysis.")
        self.filt.currentIndexChanged.connect(self._analysis_changed)
        self.hz = QDoubleSpinBox()
        self.hz.setRange(0.1, 1000)
        self.hz.setSuffix(" Hz")
        self.hz.setValue(6.0)
        self.hz.valueChanged.connect(self._analysis_changed)
        fr.addWidget(self.filt)
        fr.addWidget(self.hz)
        f.addRow("Smoothing", fr)
        self.trail = QSpinBox()
        self.trail.setRange(0, 100000)
        self.trail.setSuffix(" frames")
        self.trail.valueChanged.connect(self._analysis_changed)
        f.addRow("Trail", self.trail)
        return g

    def _analysis_changed(self, *_) -> None:
        if self._busy:
            return
        p = self.app.project
        self.app.push_undo(coalesce="analysis")
        p.capture_fps = self.fps.value()
        mode = self.filt.currentText()
        self.hz.setEnabled(mode == "Manual")
        p.filter_hz = {"Auto": 0.0, "Off": -1.0}.get(mode, self.hz.value())
        p.trail = self.trail.value()
        self.app.changed()

    # ---------------- refresh ----------------
    def refresh(self) -> None:
        app, p = self.app, self.app.project
        self._busy = True
        try:
            self.tracks.clear()
            for t in p.tracks:
                n = len(t.pts)
                extra = "fixed" if t.static else f"{n} fr" + (f" · lost @{t.lost_at}" if t.lost_at is not None else "")
                it = QListWidgetItem(swatch(t.color), f"{t.name}   ({extra})")
                it.setData(Qt.UserRole, t.id)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if t.visible else Qt.Unchecked)
                it.setToolTip("Tick = visible")
                self.tracks.addItem(it)
                if t.id == app.selected:
                    self.tracks.setCurrentItem(it)
            t = p.track(app.selected)
            self.props.setEnabled(t is not None)
            if t:
                self.name.setText(t.name)
                self.color.setIcon(swatch(t.color))
                self.color.setText(t.color)
                self.method.setCurrentIndex(max(0, self.method.findData(t.method)))
                self.tsize.setValue(t.template_r)
                self.ssize.setValue(t.search_r)
                if app.video:
                    tr, sr = auto_radii(app.video.width, app.video.height)
                    self.tsize.setSpecialValueText(f"auto ({tr})")
                    self.ssize.setSpecialValueText(f"auto ({max(sr, 2 * (t.template_r or tr))})")
                self.static.setChecked(t.static)
                man = sum(1 for v in t.pts.values() if v[3] == MANUAL)
                low = sum(1 for v in t.pts.values() if v[3] != MANUAL and v[2] < 0.75)
                fr = t.frames()
                self.info.setText("" if t.static or not fr else
                                  f"Frames {fr[0]}–{fr[-1]} · {man} keyframe(s) · {low} low-confidence")
            else:
                self.name.clear()
                self.info.setText("Select a point to edit it.")
            cur = self.measures.currentItem().data(Qt.UserRole) if self.measures.currentItem() else None
            self.measures.clear()
            for m in p.measures:
                it = QListWidgetItem(swatch(m.color), m.name)
                it.setData(Qt.UserRole, m.id)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if m.visible else Qt.Unchecked)
                self.measures.addItem(it)
                if m.id == cur:
                    self.measures.setCurrentItem(it)
            c = p.calib
            if c.mode == "line" and len(c.line) == 2:
                px = math.dist(*c.line)
                self.cal_state.setText(f"<b>Scale line</b>: {px:.1f} px = {fmt(c.length, c.unit)}"
                                       + (f" <span style='color:#8fa0bb'>({c.note})</span>" if c.note else ""))
            elif c.mode == "plane" and len(c.quad) == 4:
                self.cal_state.setText(f"<b>Plane</b>: {fmt(c.width, c.unit)} × {fmt(c.height, c.unit)} "
                                       "(perspective-corrected; origin = bottom-left corner)")
            elif c.mode == "circle" and len(c.quad) == 4:
                self.cal_state.setText(f"<b>Wheel</b> ⌀ {fmt(c.diameter, c.unit)}: measures in the wheel's plane, "
                                       "corrected for the angle it's filmed at. Origin = where the wheel meets the ground; "
                                       "x along the bike, y up.")
            else:
                self.cal_state.setText("<span style='color:#ffae42'>Not calibrated — values are in pixels.</span>")
            self.cal_d.setValue(c.diameter)
            self.cal_d.setEnabled(c.mode == "circle")
            self.stab.setChecked(bool(p.stab))
            if p.stab and app.video:
                n = app.n_frames()
                miss = [f for f in range(n) if f not in p.stab]
                runs = []
                for f in miss:
                    if runs and f == runs[-1][1] + 1:
                        runs[-1][1] = f
                    else:
                        runs.append([f, f])
                txt = f"Following the camera on {n - len(miss)} of {n} frames."
                if runs:
                    txt += " View blocked (no measurements): " + ", ".join(
                        f"{a}–{b}" if a != b else str(a) for a, b in runs[:4]) + ("…" if len(runs) > 4 else "")
                self.stab_state.setText(txt)
            else:
                self.stab_state.setText("")
            self.cal_len.setValue(c.length)
            self.cal_w.setValue(c.width)
            self.cal_h.setValue(c.height)
            self.cal_unit.setCurrentText(c.unit)
            self.axis.setValue(-math.degrees(c.axis_angle))
            self.yup.setChecked(c.y_up)
            self.cal_len.setEnabled(c.mode == "line")
            self.cal_w.setEnabled(c.mode == "plane")
            self.cal_h.setEnabled(c.mode == "plane")
            self.axis.setEnabled(c.mode not in ("plane", "circle"))
            self.yup.setEnabled(c.mode not in ("plane", "circle"))
            self.fps.setSpecialValueText(f"file ({app.video.fps:.3g})" if app.video else "file")
            self.fps.setValue(p.capture_fps)
            self.filt.setCurrentText("Off" if p.filter_hz < 0 else "Auto" if p.filter_hz == 0 else "Manual")
            if p.filter_hz > 0:
                self.hz.setValue(p.filter_hz)
            self.hz.setEnabled(p.filter_hz > 0)
            self.trail.setValue(p.trail)
        finally:
            self._busy = False
        self._sync_signed()
        self.update_readout()
        self.guide.refresh()
