"""Main window: owns the project/video state and wires every panel together."""

from __future__ import annotations

import copy
import csv
import math
import os
import time

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QSettings, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QImage, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (QApplication, QStackedWidget, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox, QProgressDialog, QPushButton,
                               QSpinBox, QSplitter, QToolBar, QVBoxLayout, QWidget)

from ..kinematics import Kinematics
from ..model import AUTO, INTERP, MANUAL, Calibration, Project
from ..render import draw_overlays
from ..tracker import auto_radii, run_tracking, snap_to_feature
from ..video import VideoSource
from .canvas import Canvas, bgr_to_qimage
from .welcome import Welcome, examples
from .tour import Spotlight
from .plots import PlotPanel
from .sidebar import UNITS, Sidebar
from .timeline import Timeline

VIDEO_FILTER = "Videos (*.mp4 *.mov *.m4v *.avi *.mkv *.webm *.mts *.m2ts *.wmv *.mpg *.mpeg *.3gp);;All files (*)"

DEFAULT_OPTS = dict(threshold=0.65, bridge=10, stop_on_loss=True, retrack=True, snap=True)

TOOL_HINTS = {
    "select": "Drag points to correct them · right-click for options · wheel = zoom, right-drag = pan · T = track forward",
    "point": "Click on the video to add a point (snaps to markers, hold Shift to place exactly) · then press T to track",
    "angle": "Angle: click 3 points — end, VERTEX, end. Existing points are reused; empty space adds a fixed point",
    "segment": "Tilt: click 2 points — angle of that segment against the x-axis",
    "distance": "Distance: click 2 points",
    "line": "Scale: click both ends of something whose length you know",
    "plane": "Plane: click the 4 corners of a known rectangle in order: top-left, top-right, bottom-right, bottom-left",
    "circle": "Wheel: click the outer edge of the tyre at its top, front (right), bottom and back (left), then type its diameter",
    "origin": "Click where (0, 0) should be",
}


class TrackWorker(QObject):
    points = Signal(int, list)
    done = Signal(object)

    def __init__(self, path, tracks, start, direction, n, opts, stop_at, stab=None):
        super().__init__()
        self.args = (path, tracks, start, direction, n)
        self.stab = stab
        self.opts = opts
        self.stop_at = stop_at
        self.cancel = False

    def run(self) -> None:
        path, tracks, start, direction, n = self.args
        try:
            res = run_tracking(path, tracks, start, direction, n, self.points.emit, lambda: self.cancel,
                               threshold=self.opts["threshold"], stop_at=self.stop_at,
                               stop_on_loss=self.opts["stop_on_loss"], bridge=self.opts["bridge"], stab=self.stab)
        except Exception as e:  # surface, don't crash the UI
            res = e
        self.done.emit(res)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("tracklab", os.environ.get("TRACKLAB_SETTINGS", "tracklab"))
        self.opts = {k: type(v)(self.settings.value(f"opt/{k}", v, type=type(v))) for k, v in DEFAULT_OPTS.items()}
        self.project = Project()
        self.video: VideoSource | None = None
        self.frame = 0
        self.selected: int | None = None
        self.tool = "select"
        self.place_target: int | None = None
        self.pending_refs: list[int] = []
        self.calib_pending: list[list[float]] = []
        self.in_out: tuple[int, int] | None = None
        self.tracking = False
        self._worker = None
        self._thread = None
        self._kin: Kinematics | None = None
        self._kin_dirty = True
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._coalesce = (None, 0.0)
        self._last_ui = 0.0
        self._lost_msg = ""
        self.guess = None  # (track id, frame, x, y) suggestion shown when a point is lost

        self.setWindowTitle("TrackLab")
        self.resize(1500, 950)
        self.canvas = Canvas(self)
        self.timeline = Timeline(self)
        self.sidebar = Sidebar(self)
        self.plots = PlotPanel(self)

        self.hint = QLabel()
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        lv.addWidget(self.hint)
        self.welcome = Welcome(self)
        self.stage = QStackedWidget()
        self.stage.addWidget(self.welcome)
        self.stage.addWidget(self.canvas)
        lv.addWidget(self.stage, 1)
        self.transport_w = self._transport()
        lv.addWidget(self.transport_w)
        lv.addWidget(self.timeline)

        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(left)
        vsplit.addWidget(self.plots)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([680, 260])
        hsplit = QSplitter(Qt.Horizontal)
        hsplit.addWidget(vsplit)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        tour_wrap = QWidget()
        tw = QVBoxLayout(tour_wrap)
        tw.setContentsMargins(8, 8, 8, 0)
        tw.addWidget(self.sidebar.tour)
        rv.addWidget(tour_wrap)
        rv.addWidget(self.sidebar, 1)
        right.setMinimumWidth(self.sidebar.minimumWidth())
        right.setMaximumWidth(self.sidebar.maximumWidth())
        hsplit.addWidget(right)
        hsplit.setStretchFactor(0, 1)
        hsplit.setSizes([1180, 320])
        self.setCentralWidget(hsplit)
        self.spotlight = Spotlight(self)

        self._menus()
        self._toolbar()
        self._shortcuts()
        self.pos_label = QLabel()
        self.statusBar().addPermanentWidget(self.pos_label)

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(5)
        self._play_timer.timeout.connect(self._play_tick)
        self._autosave = QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(800)
        self._autosave.timeout.connect(self.save_sidecar)
        self.canvas.zoomChanged.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        self.set_tool("select")
        self.refresh_all()

    # ======================================================================= UI construction
    def _transport(self) -> QWidget:
        w = QWidget()
        w.setObjectName("transport")
        h = QHBoxLayout(w)
        h.setContentsMargins(8, 4, 8, 4)

        def b(text, tip, cb, width=34):
            x = QPushButton(text)
            x.setToolTip(tip)
            x.setFixedWidth(width)
            x.clicked.connect(cb)
            h.addWidget(x)
            return x

        b("⏮", "First frame (Home)", lambda: self.seek(self.in_out[0] if self.in_out else 0))
        b("⏪", "Back 10 frames (Shift+←)", lambda: self.step(-10))
        b("◀", "Previous frame (←)", lambda: self.step(-1))
        self.play_btn = b("▶", "Play / pause (Space)", self.toggle_play, 44)
        b("▶|", "Next frame (→)", lambda: self.step(1))
        b("⏩", "Forward 10 frames (Shift+→)", lambda: self.step(10))
        b("⏭", "Last frame (End)", lambda: self.seek(self.in_out[1] if self.in_out else self.n_frames() - 1))
        h.addSpacing(10)
        self.frame_spin = QSpinBox()
        self.frame_spin.setPrefix("frame ")
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.valueChanged.connect(lambda v: self.seek(v) if v != self.frame else None)
        h.addWidget(self.frame_spin)
        self.time_label = QLabel()
        self.time_label.setMinimumWidth(170)
        h.addWidget(self.time_label)
        self.speed = QComboBox()
        for s in ("0.05×", "0.1×", "0.25×", "0.5×", "1×", "2×"):
            self.speed.addItem(s, float(s[:-1]))
        self.speed.setCurrentIndex(4)
        self.speed.setToolTip("Playback speed")
        h.addWidget(self.speed)
        h.addStretch(1)
        b("[", "Set range start here ([)", lambda: self.set_in_out(0), 28)
        b("]", "Set range end here (])", lambda: self.set_in_out(1), 28)
        b("✕", "Clear range (\\)", lambda: self.set_in_out(None), 28)
        h.addSpacing(10)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(48)
        h.addWidget(self.zoom_label)
        b("Fit", "Fit video to window (F)", self.canvas.fit, 44)
        return w

    def _menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction("Open video…", QKeySequence.Open, self.open_dialog)
        self.recent_menu = m.addMenu("Open recent")
        self._fill_recent()
        m.addAction("Open project…", self.open_project_dialog)
        m.addAction("Save project as…", QKeySequence("Ctrl+Shift+S"), self.save_project_as)
        m.addSeparator()
        m.addAction("Export data (CSV)…", QKeySequence("Ctrl+E"), self.export_csv)
        m.addAction("Export video with overlays…", QKeySequence("Ctrl+Shift+E"), self.export_video)
        m.addAction("Export current frame (PNG)…", self.export_png)
        m.addSeparator()
        m.addAction("Quit", QKeySequence.Quit, self.close)
        e = mb.addMenu("&Edit")
        e.addAction("Undo", QKeySequence.Undo, self.undo)
        e.addAction("Redo", QKeySequence("Ctrl+Shift+Z"), self.redo)
        e.addSeparator()
        e.addAction("Delete point on this frame", QKeySequence.Delete, self.delete_point_here)
        e.addAction("Delete selected track", QKeySequence("Shift+Delete"), self.delete_selected_track)
        e.addAction("Fill gaps in selected track (interpolate)", self.fill_gaps)
        e.addAction("Clear auto points after this frame", self.clear_after)
        t = mb.addMenu("&Track")
        t.addAction("Track forward", QKeySequence("T"), lambda: self.track_run(1))
        t.addAction("Track backward", QKeySequence("Shift+T"), lambda: self.track_run(-1))
        t.addAction("Track selected point only", QKeySequence("Ctrl+T"),
                    lambda: self.track_run(1, ids=[self.selected] if self.selected else None))
        t.addAction("Stop", self.stop_tracking)
        h = mb.addMenu("&Help")
        ex = h.addMenu("Examples (already set up, with a tour)")
        for path, title, _ in examples():
            ex.addAction(title, lambda p=path: self.open_path(p))
        h.addAction("Restart this example's tour", self.restart_tour)
        h.addSeparator()
        h.addAction("Shortcuts & workflow", QKeySequence("F1"), self.show_help)

    def _toolbar(self) -> None:
        tb = QToolBar("Tools")
        self.toolbar = tb
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)
        tb.addAction("Open", self.open_dialog).setToolTip("Open a video (Ctrl+O) — or drop one on the window")
        tb.addSeparator()
        self.tool_actions = {}
        grp = QActionGroup(self)
        for key, text, sc in (("select", "Select", "V"), ("point", "+ Point", "P"), ("angle", "∠ Angle", "A"),
                              ("segment", "⦣ Tilt", "G"), ("distance", "↔ Distance", "D"),
                              ("line", "📏 Scale", "C"), ("plane", "▱ Plane", ""), ("origin", "⊕ Origin", "O")):
            a = QAction(text, self, checkable=True)
            a.setToolTip(f"{TOOL_HINTS[key]}" + (f"  [{sc}]" if sc else ""))
            a.triggered.connect(lambda _=False, k=key: self.set_tool(k))
            grp.addAction(a)
            tb.addAction(a)
            self.tool_actions[key] = a
        tb.addSeparator()
        self.act_back = tb.addAction("◀ Track", lambda: self.track_run(-1))
        self.act_back.setToolTip("Track all points backward from this frame (Shift+T)")
        self.act_fwd = tb.addAction("Track ▶", lambda: self.track_run(1))
        self.act_fwd.setToolTip("Track all points on this frame forward (T). Stops and asks you when unsure.")
        self.act_stop = tb.addAction("■ Stop", self.stop_tracking)
        self.act_stop.setToolTip("Stop tracking (Esc)")
        self.act_stop.setEnabled(False)
        tb.addSeparator()
        self.act_export_csv = tb.addAction("Export CSV", self.export_csv)
        self.act_export_csv.setToolTip("Positions, speeds, angles… (Ctrl+E)")
        tb.addAction("Export video", self.export_video).setToolTip("Render the video with overlays (Ctrl+Shift+E)")

    def _shortcuts(self) -> None:
        def sc(key, fn):
            s = QShortcut(QKeySequence(key), self)
            s.setContext(Qt.WindowShortcut)
            s.activated.connect(fn)

        sc("Space", self.toggle_play)
        sc("Right", lambda: self.step(1))
        sc("Left", lambda: self.step(-1))
        sc(".", lambda: self.step(1))
        sc(",", lambda: self.step(-1))
        sc("Shift+Right", lambda: self.step(10))
        sc("Shift+Left", lambda: self.step(-10))
        sc("Ctrl+Right", lambda: self.jump_key(1))
        sc("Ctrl+Left", lambda: self.jump_key(-1))
        sc("Home", lambda: self.seek(0))
        sc("End", lambda: self.seek(self.n_frames() - 1))
        sc("Escape", self.escape)
        sc("Return", self.accept_guess)
        sc("Enter", self.accept_guess)
        sc("F", self.canvas.fit)
        sc("Q", self.arm_place)
        sc("[", lambda: self.set_in_out(0))
        sc("]", lambda: self.set_in_out(1))
        sc("\\", lambda: self.set_in_out(None))
        sc("Ctrl+Y", self.redo)
        sc("Tab", lambda: self.cycle_selection(1))
        sc("Shift+Tab", lambda: self.cycle_selection(-1))
        for key, tool in (("V", "select"), ("P", "point"), ("A", "angle"), ("G", "segment"), ("D", "distance"),
                          ("C", "line"), ("O", "origin")):
            sc(key, lambda t=tool: self.set_tool(t))
        for i in range(1, 10):
            sc(str(i), lambda i=i: self.select_track(self.project.tracks[i - 1].id) if len(self.project.tracks) >= i else None)

    # ======================================================================= state helpers
    def n_frames(self) -> int:
        return self.video.frame_count if self.video else 0

    def fps(self) -> float:
        return self.project.capture_fps or (self.video.fps if self.video else 30.0)

    def kin(self) -> Kinematics | None:
        if self.video is None:
            return None
        if self._kin_dirty or self._kin is None:
            if self.tracking and self._kin is not None:
                return self._kin  # don't recompute mid-run
            self._kin = Kinematics(self.project, self.n_frames(), self.video.fps)
            self._kin_dirty = False
        return self._kin

    def track_radii(self, t) -> tuple[int, int] | None:
        if not self.video or t is None:
            return None
        atr, asr = auto_radii(self.video.width, self.video.height)
        tr = t.template_r or atr
        return tr, t.search_r or max(asr, 2 * tr)

    def set_opt(self, k: str, v) -> None:
        self.opts[k] = v
        self.settings.setValue(f"opt/{k}", v)

    def status(self, msg: str, ms: int = 6000) -> None:
        self.statusBar().showMessage(msg, ms)

    def show_cursor_pos(self, ip: QPointF) -> None:
        if not self.video:
            return
        w = self.project.world(np.array([[ip.x(), ip.y()]]), self.frame)[0]
        self.pos_label.setText(f"px {ip.x():.1f}, {ip.y():.1f}   →   {w[0]:.4g}, {w[1]:.4g} {self.project.calib.unit_label()}")

    # ======================================================================= change propagation
    def changed(self, data: bool = True) -> None:
        """Call after mutating the project. data=False for purely cosmetic changes."""
        if data:
            self._kin_dirty = True
        self.sidebar.refresh()
        self.timeline.refresh_height()
        self.canvas.update()
        self.plots.rebuild()
        self._update_hint()
        if self.video:
            self._autosave.start()

    def _show_stage(self) -> None:
        self.stage.setCurrentWidget(self.canvas if self.video else self.welcome)
        if not self.video:
            self.welcome.refresh()

    def refresh_all(self) -> None:
        self._show_stage()
        self._kin_dirty = True
        self.changed()
        self._update_transport()

    def _update_hint(self) -> None:
        if self.place_target and self.project.track(self.place_target):
            t = self.project.track(self.place_target)
            self.hint.setProperty("alert", True)
            msg = self._lost_msg or f"Click on the video to place “{t.name}” on frame {self.frame}."
            self.hint.setText(f"⚠  {msg}   (Esc: skip it and keep tracking the others)" if self._lost_msg
                              else f"⚠  {msg}   (Esc to cancel)")
        elif not self.video:
            self.hint.setProperty("alert", False)
            self.hint.setText("Open a video to start (Ctrl+O, or drag it onto the window).")
        else:
            self.hint.setProperty("alert", False)
            extra = ""
            if self.tool in ("angle", "segment", "distance") and self.pending_refs:
                need = {"angle": 3, "segment": 2, "distance": 2}[self.tool]
                extra = f"   —   {len(self.pending_refs)}/{need} picked"
            if self.tool in ("line", "plane") and self.calib_pending:
                extra = f"   —   {len(self.calib_pending)}/{2 if self.tool == 'line' else 4} clicked"
            self.hint.setText(TOOL_HINTS[self.tool] + extra)
        self.hint.style().unpolish(self.hint)
        self.hint.style().polish(self.hint)

    def _update_transport(self) -> None:
        n = self.n_frames()
        self.frame_spin.blockSignals(True)
        self.frame_spin.setRange(0, max(0, n - 1))
        self.frame_spin.setSuffix(f" / {max(0, n - 1)}")
        self.frame_spin.setValue(self.frame)
        self.frame_spin.blockSignals(False)
        fps = self.fps()
        t = self.frame / fps
        self.time_label.setText(f"  {t:8.3f} s   ({fps:g} fps{' capture' if self.project.capture_fps else ''})")

    # ======================================================================= undo
    def push_undo(self, coalesce: str | None = None) -> None:
        now = time.monotonic()
        if coalesce and self._coalesce[0] == coalesce and now - self._coalesce[1] < 1.5:
            self._coalesce = (coalesce, now)
            return
        self._coalesce = (coalesce, now)
        self._undo.append(self.project.snapshot())
        del self._undo[:-200]
        self._redo.clear()

    def _restore(self, snap: dict) -> None:
        vp = self.project.video_path
        self.project = Project.from_dict(snap)
        self.project.video_path = vp
        if not self.project.track(self.selected or -1):
            self.selected = None
        self.place_target = None
        self.refresh_all()

    def undo(self) -> None:
        if self.tracking or not self._undo:
            return
        self._redo.append(self.project.snapshot())
        self._restore(self._undo.pop())
        self.status("Undone")

    def redo(self) -> None:
        if self.tracking or not self._redo:
            return
        self._undo.append(self.project.snapshot())
        self._restore(self._redo.pop())
        self.status("Redone")

    # ======================================================================= files
    def open_dialog(self) -> None:
        d = self.settings.value("last_dir", os.path.expanduser("~/Videos"))
        path, _ = QFileDialog.getOpenFileName(self, "Open video", d, VIDEO_FILTER)
        if path:
            self.open_path(path)

    def open_project_dialog(self) -> None:
        d = self.settings.value("last_dir", os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(self, "Open project", d, "TrackLab project (*.json)")
        if path:
            self.open_path(path)

    def sidecar(self) -> str:
        return self.project.video_path + ".tracklab.json"

    def open_path(self, path: str) -> None:
        if self.tracking:
            return
        self._stop_play()
        proj = None
        if path.endswith(".json"):
            try:
                proj = Project.load(path)
            except Exception as e:
                QMessageBox.warning(self, "TrackLab", f"Couldn't read project:\n{e}")
                return
            vp = proj.video_path
            if not os.path.exists(vp):
                alt = os.path.join(os.path.dirname(path), os.path.basename(vp))
                vp = alt if os.path.exists(alt) else vp
            path = vp
        try:
            video = VideoSource(path)
        except Exception as e:
            QMessageBox.warning(self, "TrackLab", f"Couldn't open video:\n{path}\n\n{e}")
            return
        if self.video:
            self.save_sidecar()
            self.video.close()
        self.video = video
        side = path + ".tracklab.json"
        if proj is None and os.path.exists(side):
            try:
                proj = Project.load(side)
                self.status(f"Restored your previous work on this video ({len(proj.tracks)} points)", 8000)
            except Exception:
                proj = None
        self.project = proj or Project()
        self.project.video_path = os.path.abspath(path)
        self.selected = None
        self.place_target = None
        self.in_out = None
        self._undo.clear()
        self._redo.clear()
        self.settings.setValue("last_dir", os.path.dirname(path))
        self._add_recent(path)
        self.setWindowTitle(f"TrackLab — {self.project.title or os.path.basename(path)}")
        self.frame = 0
        self.canvas.auto_fit = True
        self.seek(0, force=True)
        self.refresh_all()
        self.status(f"{video.width}×{video.height} · {video.fps:g} fps · {video.frame_count} frames."
                    + ("" if self.project.capture_fps else " Slow-mo? Set the real capture rate in Time & filtering."),
                    10000)
        if self.project.tour:
            QTimer.singleShot(150, lambda: self.sidebar.tour.start(self.project.tour))
        else:
            self.sidebar.tour.stop()

    def save_sidecar(self) -> None:
        if not self.video:
            return
        try:
            self.project.save(self.sidecar())
            self.pos_label.setToolTip(f"Autosaved to {self.sidecar()}")
        except OSError as e:
            self.status(f"Autosave failed: {e}")

    def save_project_as(self) -> None:
        if not self.video:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save project", self.sidecar(), "TrackLab project (*.json)")
        if path:
            self.project.save(path)
            self.status(f"Saved {path}")

    def _add_recent(self, path: str) -> None:
        r = [p for p in (self.settings.value("recent", [], type=list) or []) if p != path]
        self.settings.setValue("recent", [path] + r[:9])
        self._fill_recent()

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        for p in self.settings.value("recent", [], type=list) or []:
            self.recent_menu.addAction(os.path.basename(p), lambda p=p: self.open_path(p)).setToolTip(p)

    def closeEvent(self, e) -> None:
        self.stop_tracking()
        if self._thread:
            self._thread.wait(2000)
        self.save_sidecar()
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(e)

    # ======================================================================= navigation / playback
    def seek(self, f: int, force: bool = False) -> None:
        if not self.video:
            return
        f = max(0, min(self.n_frames() - 1, int(f)))
        if f == self.frame and not force and self.canvas.img is not None:
            return
        if self.place_target and f != self.frame:
            self.place_target = None
            self.guess = None
            self._lost_msg = ""
            self._update_hint()
        self.frame = f
        img = self.video.frame(f)
        if img is not None:
            self.canvas.set_image(bgr_to_qimage(img))
        if f >= self.n_frames():  # frame count got trimmed
            self.frame = self.n_frames() - 1
        self.timeline.update()
        self.plots.sync_playhead()
        if not self.tracking:
            self.sidebar.update_readout()
        self._update_transport()

    def step(self, d: int) -> None:
        self._stop_play()
        self.seek(self.frame + d)

    def jump_key(self, d: int) -> None:
        """Ctrl+←/→: jump to the selected point's previous/next keyframe or gap edge."""
        t = self.project.track(self.selected or -1)
        if not t or t.static:
            return
        fr = t.frames()
        marks = sorted({f for f in fr if t.pts[f][3] == MANUAL} | {f for f in fr if f - 1 not in t.pts or f + 1 not in t.pts}
                       | ({t.lost_at} if t.lost_at is not None else set()))
        c = [m for m in marks if (m > self.frame if d > 0 else m < self.frame)]
        if c:
            self.seek(min(c) if d > 0 else max(c))

    def toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._stop_play()
        elif self.video and not self.tracking:
            lo, hi = self.in_out or (0, self.n_frames() - 1)
            if self.frame >= hi:
                self.seek(lo)
            self._play_t0 = time.monotonic()
            self._play_f0 = self.frame
            self._play_timer.start()
            self.play_btn.setText("⏸")

    def _stop_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_btn.setText("▶")
            self.sidebar.update_readout()

    def _play_tick(self) -> None:
        lo, hi = self.in_out or (0, self.n_frames() - 1)
        el = time.monotonic() - self._play_t0
        f = self._play_f0 + int(el * self.video.fps * self.speed.currentData())
        if f > hi:
            if self.in_out:  # loop the range
                self._play_t0, self._play_f0 = time.monotonic(), lo
                f = lo
            else:
                self.seek(hi)
                self._stop_play()
                return
        if f != self.frame:
            self.seek(f)

    def set_in_out(self, which) -> None:
        if not self.video:
            return
        if which is None:
            self.in_out = None
        else:
            lo, hi = self.in_out or (0, self.n_frames() - 1)
            if which == 0:
                lo = self.frame
                hi = max(hi, lo)
            else:
                hi = self.frame
                lo = min(lo, hi)
            self.in_out = (lo, hi)
            self.status(f"Range {lo}–{hi}: playback loops here and tracking stops at its ends")
        self.timeline.update()

    def cycle_selection(self, d: int) -> None:
        ts = self.project.tracks
        if not ts:
            return
        ids = [t.id for t in ts]
        i = ids.index(self.selected) if self.selected in ids else -1
        self.select_track(ids[(i + d) % len(ids)])

    # ======================================================================= tools & editing
    def set_tool(self, tool: str) -> None:
        if not self.video and tool != "select":
            self.status("Open a video first (Ctrl+O)")
            tool = "select"
        self.tool = tool
        self.pending_refs = []
        self.calib_pending = []
        for k, a in self.tool_actions.items():
            a.setChecked(k == tool)
        self._update_hint()
        self.canvas.update()

    def escape(self) -> None:
        if self.tracking:
            self.stop_tracking()
        elif self.place_target:
            was_lost = self._lost_msg != ""
            self.place_target = None
            self.guess = None
            self._lost_msg = ""
            self._update_hint()
            self.canvas.update()
            if was_lost and self.opts["retrack"]:
                # Skip the lost point; carry on with the ones that are still tracked.
                self.track_run(self._last_dir, after_fix=True)
        elif self.pending_refs or self.calib_pending:
            self.pending_refs, self.calib_pending = [], []
            self._update_hint()
            self.canvas.update()
        else:
            self.set_tool("select")

    def select_track(self, tid: int | None) -> None:
        self.selected = tid
        self.sidebar.refresh()
        self.timeline.update()
        self.canvas.update()

    def arm_place(self) -> None:
        if not self.project.track(self.selected or -1):
            self.status("Select a point first (click it in the list, or Tab)")
            return
        self.place_target = self.selected
        self._lost_msg = ""
        self._update_hint()

    def _ref(self, x: float, y: float) -> list[float] | None:
        """A click on the current frame -> calibration-frame pixels (same thing when the camera is still)."""
        q = self.project.to_ref([[x, y]], self.frame)[0]
        if np.isnan(q).any():
            self.status("The camera position on this frame is unknown (view blocked) — pick another frame")
            return None
        return [float(q[0]), float(q[1])]

    def _snap(self, x: float, y: float, mods) -> tuple[float, float]:
        if self.opts["snap"] and not (mods & Qt.ShiftModifier) and self.video:
            img = self.video.frame(self.frame)
            if img is not None:
                tr, _ = auto_radii(self.video.width, self.video.height)
                return snap_to_feature(img, x, y, max(6, int(tr * 1.2)))
        return x, y

    def _guard(self) -> bool:
        if self.tracking:
            self.status("Tracking is running — press Esc to stop it first")
            return False
        return self.video is not None

    def accept_guess(self) -> None:
        g = self.guess
        if g and self.place_target == g[0] and self.frame == g[1] and not self.tracking:
            self.canvas_click(g[2], g[3], Qt.ShiftModifier)  # exactly the guess, no snapping

    def canvas_click(self, x: float, y: float, mods) -> None:
        if not self._guard():
            return
        if self.place_target:
            t = self.project.track(self.place_target)
            self.place_target = None
            self.guess = None
            was_lost = self._lost_msg != ""
            self._lost_msg = ""
            if t:
                self.push_undo()
                x, y = self._snap(x, y, mods)
                t.set(self.frame, x, y, 1.0, MANUAL)
                if t.lost_at == self.frame:
                    t.lost_at = None
                self.changed()
                if was_lost and self.opts["retrack"]:
                    self.track_run(self._last_dir, after_fix=True)
            return
        tool = self.tool
        if tool == "select":
            self.select_track(None)
            return
        if tool == "point":
            self.push_undo()
            x, y = self._snap(x, y, mods)
            t = self.project.add_track()
            t.set(self.frame, x, y, 1.0, MANUAL)
            self.selected = t.id
            self.changed()
            self.status(f"Added {t.name}. Add more, or press T to track forward.")
            return
        if tool in ("angle", "segment", "distance"):
            h = self.canvas.hit(self.canvas.to_screen(x, y))
            if h and h[0] == "track":
                tid = h[1]
            else:
                self.push_undo()
                x, y = self._snap(x, y, mods)
                n = sum(1 for t in self.project.tracks if t.static) + 1
                r = self._ref(x, y)
                if r is None:
                    return
                t = self.project.add_track(name=f"F{n}", static=True)
                t.set(self.frame, *r)
                tid = t.id
            if tid in self.pending_refs:
                return
            self.pending_refs.append(tid)
            need = {"angle": 3, "segment": 2, "distance": 2}[tool]
            if len(self.pending_refs) >= need:
                self.push_undo()
                m = self.project.add_measure(tool, self.pending_refs)
                self.pending_refs = []
                self.changed()
                self.plots.show("angle" if tool != "distance" else "distance")
                self.status(f"Added {m.name}")
            else:
                self.changed(data=False)
            return
        if tool in ("line", "plane", "circle"):
            r = self._ref(x, y)
            if r is None:
                return
            self.calib_pending.append(r)
            need = 2 if tool == "line" else 4
            if len(self.calib_pending) >= need:
                pts = self.calib_pending
                self.calib_pending = []
                self._finish_calibration(tool, pts)
            self._update_hint()
            self.canvas.update()
            return
        if tool == "origin":
            self.push_undo()
            c = self.project.calib
            if c.mode in ("plane", "circle"):
                self.status("This calibration fixes the origin itself (rectangle's bottom-left / bottom of the wheel)")
            else:
                c.origin = self._ref(x, y) or c.origin
            self.set_tool("select")
            self.changed()

    def _finish_calibration(self, kind: str, pts: list) -> None:
        c = self.project.calib
        dlg = QDialog(self)
        dlg.setWindowTitle("Calibration")
        f = QFormLayout(dlg)
        unit = QComboBox()
        unit.addItems(UNITS)
        unit.setCurrentText(c.unit)
        spins = []
        labels = {"line": ["Real length"], "circle": ["Wheel diameter (outside of tyre)"]}.get(
            kind, ["Real width (TL→TR)", "Real height (TL→BL)"])
        defaults = {"line": [c.length], "circle": [c.diameter]}.get(kind, [c.width, c.height])
        for lab, dv in zip(labels, defaults):
            s = QDoubleSpinBox()
            s.setRange(1e-6, 1e9)
            s.setDecimals(4)
            s.setValue(dv)
            s.selectAll()
            f.addRow(lab, s)
            spins.append(s)
        f.addRow("Unit", unit)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        spins[0].setFocus()
        if dlg.exec() != QDialog.Accepted:
            self.set_tool("select")
            return
        self.push_undo()
        c.unit = unit.currentText()
        c.note = ""
        if kind == "line":
            c.mode, c.line, c.length = "line", pts, spins[0].value()
            if c.origin is None:
                c.origin = [pts[0][0], pts[0][1]] if pts[0][1] >= pts[1][1] else [pts[1][0], pts[1][1]]
        elif kind == "circle":
            c.mode, c.quad, c.diameter = "circle", pts, spins[0].value()
        else:
            c.mode, c.quad, c.width, c.height = "plane", pts, spins[0].value(), spins[1].value()
        if not self.project.stab:
            c.frame = self.frame
        self.set_tool("select")
        self.changed()
        self.status("Calibrated. Drag the yellow handles to fine-tune.")

    def drag_to(self, d, x: float, y: float, mods) -> None:
        if self.tracking:
            return
        c = self.project.calib
        if d[0] == "track":
            t = self.project.track(d[1])
            if t:
                if t.static and self.project.stab:
                    r = self._ref(x, y)
                    if r:
                        t.set(self.frame, *r)
                else:
                    t.set(self.frame, x, y, 1.0, MANUAL)
            self.sidebar.update_readout()
            return
        r = self._ref(x, y)
        if r is None:
            return
        if d[0] == "cal_line":
            c.line[d[1]] = r
        elif d[0] == "quad":
            c.quad[d[1]] = r
        elif d[0] == "origin":
            c.origin = r
        elif d[0] == "axis":
            o = c.origin
            a = math.atan2(r[1] - o[1], r[0] - o[0])
            if mods & Qt.ShiftModifier:
                a = round(a / (math.pi / 12)) * (math.pi / 12)
            c.axis_angle = a
        self.sidebar.update_readout()

    def drag_done(self, d, moved: bool) -> None:
        if not moved:
            if self._undo:
                self._undo.pop()  # it was just a click-select
            return
        self.changed()
        if d[0] == "track":
            t = self.project.track(d[1])
            if t and not t.static:
                if t.lost_at == self.frame:
                    t.lost_at = None
                if self.opts["retrack"] and self.frame < self.n_frames() - 1:
                    nxt = t.next_manual_after(self.frame)
                    has_after = any(f > self.frame for f in t.pts)
                    if has_after:
                        self.track_run(1, ids=[t.id], stop_at=(nxt - 1) if nxt else None, retrack=True)

    def context_menu(self, ip: QPointF, hit, gpos) -> None:
        if not self.video or self.tracking:
            return
        m = QMenu(self)
        if hit and hit[0] == "track":
            t = self.project.track(hit[1])
            self.select_track(t.id)
            m.addAction(f"{t.name}").setEnabled(False)
            m.addAction("Delete point on this frame (Del)", self.delete_point_here)
            m.addAction("Clear auto points after here", self.clear_after)
            m.addAction("Fill gaps (interpolate)", self.fill_gaps)
            m.addAction("Track only this point ▶ (Ctrl+T)", lambda: self.track_run(1, ids=[t.id]))
            m.addSeparator()
            m.addAction("Delete whole track (Shift+Del)", self.delete_selected_track)
        else:
            t = self.project.track(self.selected or -1)
            if t and t.at(self.frame) is None:
                m.addAction(f"Place “{t.name}” here", lambda: self._place_at(t.id, ip))
            m.addAction("Add new point here", lambda: self._add_at(ip))
        m.exec(gpos)

    def _place_at(self, tid: int, ip: QPointF) -> None:
        self.place_target = tid
        self.canvas_click(ip.x(), ip.y(), Qt.NoModifier)

    def _add_at(self, ip: QPointF) -> None:
        old = self.tool
        self.tool = "point"
        self.canvas_click(ip.x(), ip.y(), Qt.NoModifier)
        self.tool = old

    def delete_point_here(self) -> None:
        t = self.project.track(self.selected or -1)
        if not t or self.tracking:
            return
        if t.static:
            return self.delete_selected_track()
        if self.frame in t.pts:
            self.push_undo()
            del t.pts[self.frame]
            self.changed()

    def delete_selected_track(self) -> None:
        if self.selected and not self.tracking:
            self.push_undo()
            self.project.remove_track(self.selected)
            self.selected = None
            self.changed()

    def clear_after(self) -> None:
        t = self.project.track(self.selected or -1)
        if not t or self.tracking:
            return
        self.push_undo()
        for f in [f for f, v in t.pts.items() if f > self.frame and v[3] != MANUAL]:
            del t.pts[f]
        t.lost_at = None
        self.changed()

    def fill_gaps(self) -> None:
        """Interpolate short holes (cubic through neighbours) — marked as interpolated, not tracked."""
        t = self.project.track(self.selected or -1)
        if not t or t.static or len(t.pts) < 2:
            return
        from scipy.interpolate import CubicSpline

        fr = np.array(t.frames())
        xy = np.array([t.pts[f][:2] for f in fr])
        holes = [f for f in range(fr[0], fr[-1] + 1) if f not in t.pts]
        if not holes:
            self.status("No gaps to fill")
            return
        self.push_undo()
        cs = CubicSpline(fr, xy) if len(fr) >= 4 else None
        for f in holes:
            p = cs(f) if cs is not None else [np.interp(f, fr, xy[:, 0]), np.interp(f, fr, xy[:, 1])]
            t.pts[f] = (float(p[0]), float(p[1]), 0.5, INTERP)
        self.changed()
        self.status(f"Filled {len(holes)} frame(s) by interpolation")

    # ======================================================================= tracking
    _last_dir = 1

    def track_run(self, direction: int, ids=None, stop_at=None, retrack=False, after_fix=False) -> None:
        if self.tracking or not self.video:
            return
        self._stop_play()
        f0 = self.frame
        cand = [t for t in self.project.tracks if not t.static and t.visible and (ids is None or t.id in ids)]
        ready = [t for t in cand if t.at(f0) is not None]
        if not ready:
            missing = [t for t in cand if t.at(f0) is None]
            if not cand:
                self.status("Add a point first: press P and click on the thing to track")
            else:
                near = [t.anchor_before(f0) if direction > 0 else t.anchor_after(f0) for t in missing]
                near = [n for n in near if n is not None]
                self.status("No points on this frame. " + (f"Jump to frame {near[0]} (where one is) or place it here (Q)."
                                                           if near else "Place one here first."), 9000)
            return
        if stop_at is None and self.in_out:
            lo, hi = self.in_out
            if direction > 0 and f0 < hi:
                stop_at = hi
            if direction < 0 and f0 > lo:
                stop_at = lo
        if not after_fix and not retrack:
            self.push_undo()
        # Old auto points ahead (up to the next keyframe) are stale once we re-run from here.
        for t in ready:
            nxt = t.next_manual_after(f0) if direction > 0 else t.prev_manual_before(f0)
            for f in [f for f, v in t.pts.items() if v[3] != MANUAL and
                      ((f > f0 and (nxt is None or f < nxt)) if direction > 0 else (f < f0 and (nxt is None or f > nxt)))]:
                del t.pts[f]
            t.lost_at = None
        self._last_dir = direction
        self.tracking = True
        self._run_count = 0
        self.act_fwd.setEnabled(False)
        self.act_back.setEnabled(False)
        self.act_stop.setEnabled(True)
        self.hint.setText(f"Tracking {len(ready)} point(s) {'forward' if direction > 0 else 'backward'}…   Esc to stop")
        self._worker = TrackWorker(self.video.path, copy.deepcopy(ready), f0, direction, self.n_frames(), dict(self.opts),
                                   stop_at, self.project.stab or None)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.points.connect(self._on_points)
        self._worker.done.connect(self._on_done)
        self._thread.start()

    def _on_points(self, f: int, pts: list) -> None:
        for tid, fr, x, y, conf in pts:
            t = self.project.track(tid)
            if t is not None:
                existing = t.pts.get(fr)
                if existing is None or existing[3] != MANUAL:
                    t.pts[fr] = (x, y, conf, AUTO)
        self._run_count += 1
        now = time.monotonic()
        if now - self._last_ui > 1 / 30:
            self._last_ui = now
            self.seek(f)

    def _on_done(self, res) -> None:
        self.tracking = False
        self._thread.quit()
        self._thread.wait()
        self._thread = self._worker = None
        self.act_fwd.setEnabled(True)
        self.act_back.setEnabled(True)
        self.act_stop.setEnabled(False)
        if isinstance(res, Exception):
            QMessageBox.warning(self, "Tracking failed", str(res))
            self.changed()
            return
        self._kin_dirty = True
        if res.lost:
            for tid, f in res.lost:
                t = self.project.track(tid)
                if t:
                    t.lost_at = f
            tid, f = min(res.lost, key=lambda l: l[1] * self._last_dir)
            t = self.project.track(tid)
            self.seek(f)
            self.selected = tid
            self.place_target = tid
            others = len(res.lost) - 1
            g = (res.guesses or {}).get(tid)
            self.guess = (tid, f, g[0], g[1]) if g else None
            self._lost_msg = (f"Not sure where “{t.name}” is on frame {f}" + (f" (+{others} more)" if others else "") +
                              " — click it" + (", or press Enter to accept the dashed guess." if g else "."))
            prev = g or t.at(f - self._last_dir)
            if prev:
                self.canvas.center_on(*prev)
        else:
            self.seek(res.last_frame)
            msg = "Stopped" if res.cancelled else "Done"
            self.status(f"{msg}: tracked {res.frames_done} frame(s)")
        self.changed()

    def stop_tracking(self) -> None:
        if self._worker:
            self._worker.cancel = True

    # ======================================================================= export
    def export_csv(self) -> None:
        if not self.video or not self.project.tracks:
            self.status("Nothing to export yet")
            return
        base = os.path.splitext(self.project.video_path)[0]
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", base + "_tracking.csv", "CSV (*.csv)")
        if not path:
            return
        kin = self.kin()
        p = self.project
        u = kin.unit
        cols, data = ["frame", "time_s"], [np.arange(kin.n), kin.t]
        for t in p.tracks:
            d = kin.tracks[t.id]
            n = t.name
            cols += [f"{n}_x_px", f"{n}_y_px", f"{n}_x_{u}", f"{n}_y_{u}"]
            data += [d["px"][:, 0], d["px"][:, 1], d["x"], d["y"]]
            if not t.static:
                cols += [f"{n}_vx_{u}/s", f"{n}_vy_{u}/s", f"{n}_speed_{u}/s", f"{n}_accel_{u}/s2", f"{n}_match"]
                conf = np.full(kin.n, np.nan)
                for f, v in t.pts.items():
                    if 0 <= f < kin.n:
                        conf[f] = v[2] if v[3] != MANUAL else 1.0
                data += [d["vx"], d["vy"], d["speed"], d["accel"], conf]
        for m in p.measures:
            d = kin.measures[m.id]
            unit = u if m.kind == "distance" else "deg"
            cols += [f"{m.name} ({unit})", f"{m.name} rate ({unit}/s)"]
            data += [d["value"], d["rate"]]
        rows = np.column_stack(data)
        span = [i for i in range(kin.n) if not np.all(np.isnan(rows[i, 2:]))]
        lo, hi = (span[0], span[-1]) if span else (0, -1)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([f"# {os.path.basename(p.video_path)}  fps={self.fps():g}  units={u}  "
                        f"filter={'off' if p.filter_hz < 0 else 'auto' if p.filter_hz == 0 else f'{p.filter_hz}Hz'}"])
            w.writerow(cols)
            for i in range(lo, hi + 1):
                w.writerow([int(rows[i, 0])] + ["" if np.isnan(v) else f"{v:.6g}" for v in rows[i, 1:]])
        self.status(f"Exported {hi - lo + 1} rows → {path}", 10000)
        self.sidebar.guide.exported = True
        self.sidebar.guide.refresh()

    def _render_frame(self, idx: int, scale: float) -> QImage | None:
        img = self.video.frame(idx)
        if img is None:
            return None
        q = bgr_to_qimage(img).convertToFormat(QImage.Format_RGB888)
        p = QPainter(q)
        draw_overlays(p, self.project, idx, lambda x, y: QPointF(x, y), scale)
        p.end()
        return q

    def export_png(self) -> None:
        if not self.video:
            return
        base = os.path.splitext(self.project.video_path)[0]
        path, _ = QFileDialog.getSaveFileName(self, "Export frame", f"{base}_f{self.frame}.png", "PNG (*.png)")
        if path:
            self._render_frame(self.frame, max(1.0, self.video.height / 720)).save(path)
            self.status(f"Saved {path}")

    def export_video(self) -> None:
        if not self.video or self.tracking:
            return
        base = os.path.splitext(self.project.video_path)[0]
        path, _ = QFileDialog.getSaveFileName(self, "Export video", base + "_tracked.mp4", "MP4 (*.mp4)")
        if not path:
            return
        lo, hi = self.in_out or (0, self.n_frames() - 1)
        v = self.video
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), v.fps, (v.width, v.height))
        if not vw.isOpened():
            QMessageBox.warning(self, "Export", "Couldn't create the output video")
            return
        dlg = QProgressDialog("Rendering video…", "Cancel", lo, hi, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        scale = max(1.0, v.height / 720)
        for i in range(lo, hi + 1):
            if dlg.wasCanceled():
                break
            q = self._render_frame(i, scale)
            if q is None:
                break
            arr = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine())[:, :q.width() * 3]
            vw.write(cv2.cvtColor(arr.reshape(q.height(), q.width(), 3), cv2.COLOR_RGB2BGR))
            dlg.setValue(i)
        vw.release()
        dlg.close()
        self.seek(self.frame, force=True)
        self.status(f"Exported {path}", 10000)
        self.sidebar.guide.exported = True
        self.sidebar.guide.refresh()

    # ======================================================================= scale from gravity
    def calibrate_from_gravity(self) -> None:
        """Free fall gives the scale: fit a parabola to the selected point's flight, its curvature is 9.81 m/s².

        Uses the selected point's continuous run of tracked frames around the current frame (or the [ ] range),
        so put the playhead inside a flight phase. Also levels the axes: y points against gravity."""
        p = self.project
        t = p.track(self.selected or -1)
        if not t or t.static:
            self.status("Select a tracked point that is flying freely (a thrown ball, a jumper in the air)")
            return
        if self.in_out:
            frames = [f for f in range(self.in_out[0], self.in_out[1] + 1) if f in t.pts]
        else:
            f0 = self.frame
            if f0 not in t.pts:
                self.status("Move to a frame where the selected point is tracked and in the air")
                return
            a = b = f0
            while a - 1 in t.pts:
                a -= 1
            while b + 1 in t.pts:
                b += 1
            frames = list(range(a, b + 1))
        frames = [f for f in frames if t.pts[f][3] != INTERP]
        if len(frames) < 8:
            self.status("Need at least 8 tracked frames of free flight — set a range with [ and ] around one flight")
            return
        fr = np.array(frames)
        ref = p.to_ref(np.array([t.pts[f][:2] for f in fr]), fr)
        ok = ~np.isnan(ref).any(axis=1)
        fr, ref = fr[ok], ref[ok]
        tt = (fr - fr[0]) / self.fps()
        cx, cy = np.polyfit(tt, ref[:, 0], 2), np.polyfit(tt, ref[:, 1], 2)
        acc = np.array([2 * cx[0], 2 * cy[0]])  # px/s², image coordinates (y down)
        g = float(np.hypot(*acc))
        if g < 1e-6 or acc[1] <= 0:
            self.status("That doesn't look like free fall (no downward curve) — pick a flight phase")
            return
        fit = np.c_[np.polyval(cx, tt), np.polyval(cy, tt)]
        rms = float(np.sqrt(((fit - ref) ** 2).sum(axis=1).mean()))
        ppm = g / 9.81
        gd = acc / g  # gravity direction in the image
        o = ref[0]
        self.push_undo()
        c = p.calib
        c.mode, c.unit, c.length = "line", "m", 1.0
        c.line = [[float(o[0]), float(o[1])], [float(o[0] - gd[0] * ppm), float(o[1] - gd[1] * ppm)]]
        c.origin = [float(o[0]), float(o[1])]
        c.axis_angle = float(math.atan2(-gd[0], gd[1]))
        c.y_up = True
        c.note = f"from gravity: {t.name}, frames {fr[0]}–{fr[-1]}"
        if not p.stab:
            c.frame = self.frame
        self.changed()
        tilt = math.degrees(math.atan2(gd[0], gd[1]))
        self.status(f"Scale from gravity: {ppm:.1f} px = 1 m ({len(fr)} frames, fit within {rms:.1f} px); "
                    f"'down' is {abs(tilt):.1f}° {'right' if tilt > 0 else 'left'} of the picture's vertical.", 15000)

    # ======================================================================= moving camera
    def set_stabilize(self, on: bool) -> None:
        p = self.project
        if not on:
            if p.stab:
                self.push_undo()
                p.stab = {}
                self.changed()
            return
        if not self.video or not p.calib.is_set():
            self.status("Calibrate first (Scale line / Plane / Wheel) — the camera is followed relative to that frame")
            self.sidebar.refresh()
            return
        from ..stabilize import compute, default_roi

        c = p.calib
        pts = c.quad if c.mode in ("plane", "circle") else c.line
        roi = default_roi(np.array(pts), self.video.width, self.video.height)
        dlg = QProgressDialog("Measuring camera motion on every frame…", "Cancel", 0, self.n_frames(), self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)

        def prog(f, n):
            dlg.setValue(f)
            QApplication.processEvents()

        st = compute(self.video.path, c.frame, self.n_frames(), prog, dlg.wasCanceled, roi=roi)
        dlg.close()
        if dlg.wasCanceled() or not st:
            self.sidebar.refresh()
            return
        self.push_undo()
        p.stab = st
        self.changed()
        self.seek(self.frame, force=True)
        self.status(f"Following the camera: {len(st)} of {self.n_frames()} frames lined up with frame {c.frame}")

    # ======================================================================= tours
    def restart_tour(self) -> None:
        if self.project.tour:
            self.sidebar.tour.start(self.project.tour)
        else:
            self.status("This video has no tour — open one from Help → Examples")

    def spot_target(self, key: str):
        sb = self.sidebar
        fixed = {"canvas": self.canvas, "timeline": self.timeline, "plots": self.plots,
                 "transport": self.transport_w, "points": sb.g_points, "measures": sb.g_measures,
                 "readout": sb.g_readout, "calib": sb.g_calib, "tracking": sb.g_tracking, "time": sb.g_time}
        if key in fixed:
            return fixed[key]
        if key.startswith("tool:"):
            a = self.tool_actions.get(key[5:])
            return self.toolbar.widgetForAction(a) if a else None
        acts = {"act:track": self.act_fwd, "act:export": self.act_export_csv, "act:stop": self.act_stop}
        if key in acts:
            return self.toolbar.widgetForAction(acts[key])
        return None

    def do_tour_step(self, s: dict) -> None:
        if self.tracking:
            return
        self._stop_play()
        if "seek" in s:
            self.seek(int(s["seek"]), force=True)
        if s.get("fit"):
            self.canvas.fit()
        if "zoom" in s:
            x, y, z = s["zoom"]
            self.canvas.focus(x, y, z)
        if "range" in s:
            self.in_out = tuple(s["range"]) if s["range"] else None
            self.timeline.update()
        if "select" in s:
            t = self.project.by_name(s["select"])
            if t is not None and hasattr(t, "pts"):
                self.select_track(t.id)
        self.set_tool(s.get("tool", "select"))
        if "plot" in s:
            q = s["plot"]
            self.plots.show(q["q"], q.get("show"), q.get("raw", False))
        targets = [self.spot_target(k) for k in s.get("spot", [])]
        side = [w for w in targets if w is not None and self.sidebar.widget().isAncestorOf(w)]
        if side:
            self.sidebar.ensureWidgetVisible(side[0], 0, 0)
        QTimer.singleShot(60, lambda: self.spotlight.show_on(targets))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self.spotlight.isVisible():
            self.spotlight.setGeometry(self.rect())

    def show_help(self) -> None:
        QMessageBox.information(self, "TrackLab — workflow", HELP)


HELP = """<h3>Workflow</h3>
<ol>
<li><b>Open</b> a video (Ctrl+O or drop it). Your work autosaves next to the video and comes back when you reopen it.</li>
<li>Slow-mo? Set the <b>capture rate</b> (Time &amp; filtering) so seconds and speeds are real.</li>
<li><b>Calibrate</b>: <i>Scale</i> (C) — click both ends of something you know the length of; or <i>Plane</i> — 4 corners
of a known rectangle for perspective-correct measurements. <i>Origin</i> (O) sets (0,0); drag the red arrow to rotate axes.</li>
<li><b>Add points</b> (P) — clicks snap onto markers (Shift = exact). The solid box is what gets matched; adjust it per point.</li>
<li><b>Track</b> (T forward, Shift+T backward). If it isn't sure, it <b>stops on that frame and asks</b> — press
<b>Enter</b> to accept its dashed guess, or click where the point is, and it carries on. Esc skips that point.
Brief occlusions are bridged automatically, and every correction teaches it what the point looks like.</li>
<li><b>Fix</b> anything by dragging it — it re-tracks from there to your next keyframe. A loupe shows the pixels.</li>
<li><b>Measure</b>: angles (A), segment tilt (G), distances (D) between any points.</li>
<li><b>Plot &amp; export</b>: the plot panel shows position/speed/acceleration/angles, smoothed with a zero-lag low-pass.
Click a plot to jump there. Ctrl+E exports CSV, Ctrl+Shift+E a video with overlays.</li>
</ol>
<h3>Keys</h3>
<table>
<tr><td>Space</td><td>play / pause</td><td>&nbsp;&nbsp;←/→ or ,/.</td><td>step 1 frame (Shift: 10)</td></tr>
<tr><td>Ctrl+←/→</td><td>prev/next keyframe or gap</td><td>&nbsp;&nbsp;Home/End</td><td>first/last frame</td></tr>
<tr><td>[ / ] / \\</td><td>range start / end / clear</td><td>&nbsp;&nbsp;F</td><td>fit to window</td></tr>
<tr><td>V P A G D C O</td><td>tools</td><td>&nbsp;&nbsp;Q</td><td>place selected point here</td></tr>
<tr><td>T / Shift+T</td><td>track fwd / back</td><td>&nbsp;&nbsp;Ctrl+T</td><td>track selected only</td></tr>
<tr><td>Tab, 1–9</td><td>select points</td><td>&nbsp;&nbsp;Del / Shift+Del</td><td>delete point here / track</td></tr>
<tr><td>Ctrl+Z / Ctrl+Shift+Z</td><td>undo / redo</td><td>&nbsp;&nbsp;Esc</td><td>stop / cancel / skip lost point</td></tr>
<tr><td>Enter</td><td>accept the tracker's guess</td><td></td><td></td></tr>
<tr><td>Wheel</td><td>zoom at cursor</td><td>&nbsp;&nbsp;Right/middle drag</td><td>pan</td></tr>
</table>"""
