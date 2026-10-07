"""Guided tours for example projects, and the spotlight that points at the UI being talked about.

A tour step is a dict stored in the project JSON:
    title, text (rich text)               what the panel shows
    seek: frame                           jump the video
    fit: true | zoom: [x, y, factor]      view (factor is relative to "fit")
    select: "<point name>"                select a point
    tool: "point" | "angle" | ...         switch tool
    plot: {"q": key, "show": [names], "raw": bool}   what the graph shows
    spot: ["plots", "canvas", "tool:point", "act:export", ...]   what to highlight
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from .main import MainWindow


class Spotlight(QWidget):
    """Transparent overlay over the whole window that draws pulsing outlines around widgets."""

    def __init__(self, window: QWidget):
        super().__init__(window)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.targets: list[QWidget] = []
        self.t = 0
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)
        self.hide()

    def show_on(self, widgets: list[QWidget], ms: int = 4500) -> None:
        self.targets = [w for w in widgets if w is not None and w.isVisible()]
        if not self.targets:
            self.hide()
            return
        self.t = 0
        self.frames = ms // 40
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        self.timer.start()

    def _tick(self) -> None:
        self.t += 1
        if self.t > self.frames:
            self.timer.stop()
            self.hide()
            return
        self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        import math

        pulse = 0.5 + 0.5 * math.sin(self.t / 4.0)
        fade = min(1.0, (self.frames - self.t) / 12)
        for w in self.targets:
            if not w.isVisible():
                continue
            tl = w.mapTo(self.parentWidget(), w.rect().topLeft())
            r = QRectF(QRect(tl, w.size())).adjusted(-3, -3, 3, 3)
            c = QColor("#ffd23f")
            c.setAlphaF(fade * (0.55 + 0.45 * pulse))
            p.setPen(QPen(c, 3 + 2 * pulse))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r, 7, 7)


class TourPanel(QGroupBox):
    def __init__(self, app: "MainWindow"):
        super().__init__("Example tour")
        self.app = app
        self.steps: list[dict] = []
        self.i = 0
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 8)
        v.setSpacing(6)
        self.progress = QLabel()
        self.progress.setStyleSheet("color:#8fa0bb")
        v.addWidget(self.progress)
        self.head = QLabel()
        self.head.setWordWrap(True)
        v.addWidget(self.head)
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.RichText)
        v.addWidget(self.body)
        row = QHBoxLayout()
        self.back = QPushButton("◀ Back")
        self.back.clicked.connect(lambda: self.go(self.i - 1))
        self.again = QPushButton("Show me")
        self.again.setToolTip("Do this step again: jump there and highlight it")
        self.again.clicked.connect(lambda: self.go(self.i))
        self.next = QPushButton("Next ▶")
        self.next.setObjectName("guidego")
        self.next.clicked.connect(lambda: self.go(self.i + 1))
        for b in (self.back, self.again, self.next):
            row.addWidget(b)
        v.addLayout(row)
        self.end = QPushButton("End tour")
        self.end.setObjectName("link")
        self.end.clicked.connect(self.stop)
        v.addWidget(self.end, 0, Qt.AlignRight)
        self.hide()

    def start(self, steps: list[dict]) -> None:
        self.steps = steps or []
        if not self.steps:
            self.stop()
            return
        self.show()
        self.app.sidebar.guide.hide()
        self.go(0)

    def stop(self) -> None:
        self.hide()
        self.app.sidebar.guide.show()
        self.app.spotlight.hide()

    def go(self, i: int) -> None:
        if not self.steps:
            return
        if i >= len(self.steps):
            self.stop()
            return
        self.i = max(0, i)
        s = self.steps[self.i]
        self.progress.setText(f"Step {self.i + 1} of {len(self.steps)}")
        self.head.setText(f"<span style='font-size:15px; font-weight:700; color:#e8edf5'>{s.get('title', '')}</span>")
        self.body.setText(f"<span style='color:#d7dce6'>{s.get('text', '')}</span>")
        self.back.setEnabled(self.i > 0)
        self.next.setText("Finish ✓" if self.i == len(self.steps) - 1 else "Next ▶")
        self.app.do_tour_step(s)
