"""Step-by-step checklist that ticks itself off from the project state, with a button per step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget

from ..model import AUTO

if TYPE_CHECKING:
    from .main import MainWindow


class Guide(QGroupBox):
    def __init__(self, app: "MainWindow"):
        super().__init__("Guide")
        self.app = app
        self.exported = False
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 4, 6, 6)
        self.toggle = QToolButton()
        self.toggle.setText("hide")
        self.toggle.setAutoRaise(True)
        self.toggle.clicked.connect(self._toggle)
        top = QGridLayout()
        self.headline = QLabel()
        self.headline.setWordWrap(True)
        self.headline.setObjectName("guidehead")
        top.addWidget(self.headline, 0, 0)
        top.addWidget(self.toggle, 0, 1, Qt.AlignTop)
        v.addLayout(top)
        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setVerticalSpacing(3)
        v.addWidget(self.body)
        # key, title, optional, button text, action, explanation shown when it's the current step
        self.steps = [
            ("open", "Open a video", False, "Open…", app.open_dialog,
             "Pick a clip. Your work saves automatically next to it."),
            ("fps", "Slow-mo? Set real frame rate", True, "Set…", self._focus_fps,
             "Only if it was filmed in slow motion: enter the real capture rate (e.g. 240) so speeds are right."),
            ("calib", "Set the scale", True, "Scale", lambda: app.set_tool("line"),
             "Click both ends of something you know the length of (a wheel, a frame tube, a ruler). "
             "Skip it and everything is in pixels."),
            ("points", "Mark points to track", False, "Add", lambda: app.set_tool("point"),
             "Click each thing to follow — e.g. hip, knee, ankle, pedal. Zoom with the wheel for precision."),
            ("track", "Track them", False, "Track ▶", lambda: app.track_run(1),
             "Follows every point through the video. If it gets unsure it stops — click the right spot and it continues."),
            ("measure", "Measure an angle or distance", True, "Angle", lambda: app.set_tool("angle"),
             "Angle: click end, vertex, end (e.g. hip → knee → ankle). Values appear on the video and in the graph."),
            ("export", "Export the results", False, "CSV", app.export_csv,
             "Saves every frame's positions, speeds and angles as a spreadsheet. Ctrl+Shift+E for a video."),
        ]
        self.rows = {}
        for i, (key, title, opt, btn, fn, _) in enumerate(self.steps):
            mark = QLabel()
            mark.setFixedWidth(18)
            lab = QLabel(title + (" <span style='color:#6b7486'>(optional)</span>" if opt else ""))
            lab.setWordWrap(True)
            b = QPushButton(btn)
            b.setFixedWidth(64)
            b.clicked.connect(fn)
            self.grid.addWidget(mark, i, 0)
            self.grid.addWidget(lab, i, 1)
            self.grid.addWidget(b, i, 2)
            self.rows[key] = (mark, lab, b)
        self.grid.setColumnStretch(1, 1)
        if app.settings.value("guide_hidden", False, type=bool):
            self._toggle()

    def _toggle(self) -> None:
        hidden = self.body.isVisibleTo(self)
        self.body.setVisible(not hidden)
        self.toggle.setText("show" if hidden else "hide")
        self.app.settings.setValue("guide_hidden", hidden)

    def _focus_fps(self) -> None:
        sb = self.app.sidebar
        sb.ensureWidgetVisible(sb.fps)
        sb.fps.setFocus()
        sb.fps.selectAll()

    def done(self) -> dict[str, bool]:
        a, p = self.app, self.app.project
        has_video = a.video is not None
        return {
            "open": has_video,
            "fps": p.capture_fps > 0,
            "calib": p.calib.is_set(),
            "points": any(not t.static for t in p.tracks),
            "track": any(sum(1 for v in t.pts.values() if v[3] == AUTO) > 1 for t in p.tracks),
            "measure": bool(p.measures),
            "export": self.exported,
        }

    def refresh(self) -> None:
        d = self.done()
        current = next((k for k, _, opt, *_ in self.steps if not opt and not d[k]), None)
        has_video = d["open"]
        for key, title, opt, _, _, expl in self.steps:
            mark, lab, b = self.rows[key]
            if d[key]:
                mark.setText("<span style='color:#3ddc97; font-weight:700'>✓</span>")
            elif key == current:
                mark.setText("<span style='color:#4da3ff; font-weight:700'>➜</span>")
            else:
                mark.setText("<span style='color:#4b5363'>○</span>")
            lab.setEnabled(key == "open" or has_video)
            b.setEnabled(key == "open" or has_video)
            b.setObjectName("guidego" if key == current else "")
            b.style().unpolish(b)
            b.style().polish(b)
        if current is None:
            self.headline.setText("<span style='color:#e8edf5'><b style='color:#3ddc97'>All done.</b> "
                                  "Keep adding points or measurements any time.</span>")
        else:
            expl = next(s[5] for s in self.steps if s[0] == current)
            # Point out the useful optional steps right before tracking.
            if current == "points" and not d["calib"]:
                expl += "<br><span style='color:#ffae42'>Tip: set the scale first if you want real units.</span>"
            self.headline.setText(f"<span style='color:#e8edf5'><b style='color:#4da3ff'>Next:</b> {expl}</span>")
