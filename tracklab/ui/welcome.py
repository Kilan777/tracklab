"""Start screen: open a finished example (with a guided tour), open your own video, or reopen something recent."""

from __future__ import annotations

import glob
import json
import os
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout,
                               QWidget)

if TYPE_CHECKING:
    from .main import MainWindow

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".ogv", ".mts", ".wmv")
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
ORDER = ["cycling", "crash_test", "paper_bag", "physics_lab", "floor_plane"]

FEATURES = {
    "cycling": "REAL · handheld camera following · wheel calibration (bike at an angle) · knee angle · cadence · gaps",
    "crash_test": "REAL · checked vs. the official 56 km/h · slow-mo clock · g-forces · crush · smoothing",
    "paper_bag": "REAL · 200 fps · colour (blob) tracking · grid scale · air drag · gravity check",
    "physics_lab": "simulated, exact answers · scale & origin · gravity · occlusion · lost point · fixed point",
    "floor_plane": "simulated · 4-corner plane calibration · real speed under perspective · distances",
}


def examples() -> list[tuple[str, str, str]]:
    """(project path, title, blurb) for every bundled example whose video exists."""
    out = []
    for path in glob.glob(os.path.join(EXAMPLES_DIR, "*.tracklab.json")):
        if os.path.basename(path)[: -len(".tracklab.json")].lower().endswith(VIDEO_EXT):
            continue  # an autosave next to an example video, not an example
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if not d.get("title"):
            continue
        vp = d.get("video_path", "")
        if not os.path.exists(vp) and not os.path.exists(os.path.join(EXAMPLES_DIR, os.path.basename(vp))):
            continue
        out.append((path, d["title"], d.get("blurb", "")))
    key = lambda e: next((i for i, k in enumerate(ORDER) if os.path.basename(e[0]).startswith(k)), 99)  # noqa: E731
    return sorted(out, key=key)


class Welcome(QScrollArea):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        inner.setObjectName("welcome")
        inner.setAttribute(Qt.WA_StyledBackground, True)
        self.setWidget(inner)
        self.setAcceptDrops(True)
        outer = QVBoxLayout(inner)
        outer.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        box = QWidget()
        box.setMaximumWidth(820)
        v = QVBoxLayout(box)
        v.setSpacing(12)
        v.setContentsMargins(24, 28, 24, 24)
        title = QLabel("TrackLab")
        title.setObjectName("wtitle")
        v.addWidget(title)
        sub = QLabel("Track points in a video, then measure angles, distances and speeds.")
        sub.setObjectName("wsub")
        v.addWidget(sub)

        head = QLabel("New here? Open an example — it's already done, and a tour shows you around")
        head.setObjectName("whead")
        v.addWidget(head)
        self.cards = QGridLayout()
        self.cards.setHorizontalSpacing(12)
        self.cards.setVerticalSpacing(12)
        v.addLayout(self.cards)

        head2 = QLabel("Your own video")
        head2.setObjectName("whead")
        v.addWidget(head2)
        row = QHBoxLayout()
        open_btn = QPushButton("Open a video…")
        open_btn.setObjectName("primary")
        open_btn.clicked.connect(app.open_dialog)
        row.addWidget(open_btn)
        how = QLabel("…or drag a video onto this window. Then: click the things to track → press <b>Track ▶</b> "
                     "→ add angles/distances → export. The <b>Guide</b> on the right walks you through it.")
        how.setWordWrap(True)
        how.setObjectName("wsub")
        row.addWidget(how, 1)
        v.addLayout(row)

        self.recent_head = QLabel("Recent")
        self.recent_head.setObjectName("whead")
        v.addWidget(self.recent_head)
        self.recent_box = QVBoxLayout()
        v.addLayout(self.recent_box)
        v.addStretch(1)
        outer.addWidget(box)
        self.refresh()

    def _clear(self, lay) -> None:
        while lay.count():
            w = lay.takeAt(0).widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def refresh(self) -> None:
        self._clear(self.cards)
        for i, (path, title, blurb) in enumerate(examples()):
            key = next((k for k in FEATURES if os.path.basename(path).startswith(k)), "")
            card = QPushButton()
            card.setObjectName("excard")
            card.setCursor(Qt.PointingHandCursor)
            card.clicked.connect(lambda _=False, p=path: self.app.open_path(p))
            lay = QVBoxLayout(card)
            lay.setContentsMargins(14, 12, 14, 12)
            t = QLabel(f"<span style='font-size:15px; font-weight:700; color:#e8edf5'>{title}</span>")
            t.setWordWrap(True)
            b = QLabel(f"<span style='color:#aab4c5'>{blurb}</span>")
            b.setWordWrap(True)
            f = QLabel(f"<span style='color:#7d8799; font-size:11px'>Shows: {FEATURES.get(key, '')}</span>")
            f.setWordWrap(True)
            go = QLabel("<span style='color:#4da3ff; font-weight:700'>Open with tour ▶</span>")
            for w in (t, b, f, go):
                w.setAttribute(Qt.WA_TransparentForMouseEvents)
                lay.addWidget(w)
            card.setMinimumHeight(235)
            self.cards.addWidget(card, i // 3, i % 3)
        recent = []
        seen = set()
        for p in self.app.settings.value("recent", [], type=list) or []:
            rp = os.path.realpath(p) if p else ""
            if p and os.path.isfile(p) and rp not in seen and not rp.startswith(EXAMPLES_DIR):
                seen.add(rp)
                recent.append(p)
        self._clear(self.recent_box)
        for p in recent[:5]:
            b = QPushButton(f"▶  {os.path.basename(p)}")
            b.setObjectName("link")
            b.setToolTip(p)
            b.clicked.connect(lambda _=False, p=p: self.app.open_path(p))
            self.recent_box.addWidget(b)
        self.recent_head.setVisible(bool(recent))

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        urls = e.mimeData().urls()
        if urls:
            self.app.open_path(urls[0].toLocalFile())
