"""Scrubbable timeline with one coverage lane per point: see at a glance what's tracked."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from ..model import MANUAL

if TYPE_CHECKING:
    from .main import MainWindow

RULER = 18
LANE = 9
GAP = 2
MAX_LANES = 10


class Timeline(QWidget):
    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        self.setMouseTracking(True)
        self._drag = False
        self.setMinimumHeight(RULER + 14)

    def sizeHint(self):
        from PySide6.QtCore import QSize

        n = min(MAX_LANES, len(self._lanes()))
        return QSize(400, RULER + 10 + n * (LANE + GAP))

    def _lanes(self):
        return [t for t in self.app.project.tracks if not t.static]

    def refresh_height(self) -> None:
        n = min(MAX_LANES, len(self._lanes()))
        self.setFixedHeight(RULER + 10 + max(1, n) * (LANE + GAP))
        self.update()

    def _x(self, f: float) -> float:
        n = max(1, self.app.n_frames() - 1)
        return 8 + (self.width() - 16) * f / n

    def _f(self, x: float) -> int:
        n = max(1, self.app.n_frames() - 1)
        return int(round((x - 8) / max(1, self.width() - 16) * n))

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#14171d"))
        app = self.app
        n = app.n_frames()
        if n <= 1:
            return
        W = self.width()
        # in/out range
        if app.in_out:
            a, b = app.in_out
            p.fillRect(QRectF(self._x(a), 0, self._x(b) - self._x(a), self.height()), QColor(77, 163, 255, 28))
        # ruler
        p.setPen(QColor("#5c667a"))
        fps = app.fps()
        step = _nice_step(n, W)
        f = p.font()
        f.setPointSizeF(7.5)
        p.setFont(f)
        for k in range(0, n, step):
            x = self._x(k)
            p.drawLine(QPointF(x, RULER - 5), QPointF(x, RULER))
            p.drawText(QPointF(x + 2, RULER - 6), f"{k / fps:.2f}s" if step >= 10 else str(k))
        # lanes
        lanes = self._lanes()[:MAX_LANES]
        px_per_f = (W - 16) / max(1, n - 1)
        for i, t in enumerate(lanes):
            y = RULER + 4 + i * (LANE + GAP)
            p.fillRect(QRectF(8, y, W - 16, LANE), QColor("#1d2129"))
            col = QColor(t.color)
            if not t.visible:
                col.setAlpha(80)
            lo = QColor("#ffae42")
            # Draw runs, merging consecutive frames (fast even for long clips).
            run = None
            for fr in sorted(t.pts):
                c = lo if t.pts[fr][2] < 0.75 and t.pts[fr][3] != MANUAL else col
                if run and fr == run[1] + 1 and run[2] == c.name():
                    run[1] = fr
                    continue
                if run:
                    self._run(p, run, y, px_per_f)
                run = [fr, fr, c.name()]
            if run:
                self._run(p, run, y, px_per_f)
            p.setPen(QPen(QColor("#ffffff"), 1.5))
            for fr, v in t.pts.items():
                if v[3] == MANUAL:
                    x = self._x(fr)
                    p.drawLine(QPointF(x, y - 1), QPointF(x, y + LANE + 1))
            if t.lost_at is not None:
                x = self._x(t.lost_at)
                p.fillRect(QRectF(x - 2, y - 1, 4, LANE + 2), QColor("#ff3b3b"))
            if t.id == app.selected:
                p.setPen(QPen(QColor("#ffffff"), 1))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(7, y - 1, W - 14, LANE + 2))
        # playhead
        x = self._x(app.frame)
        p.setPen(QPen(QColor("#4da3ff"), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.setBrush(QColor("#4da3ff"))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(x, 4), 4, 4)

    def _run(self, p: QPainter, run, y: float, px_per_f: float) -> None:
        x0 = self._x(run[0]) - px_per_f / 2
        x1 = self._x(run[1]) + px_per_f / 2
        p.fillRect(QRectF(x0, y, max(1.5, x1 - x0), LANE), QColor(run[2]))

    def _lane_at(self, y: float):
        if y < RULER + 4:
            return None
        i = int((y - RULER - 4) // (LANE + GAP))
        lanes = self._lanes()[:MAX_LANES]
        return lanes[i] if 0 <= i < len(lanes) else None

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._drag = True
            t = self._lane_at(e.position().y())
            if t is not None:
                self.app.select_track(t.id)
            self.app.seek(self._f(e.position().x()))

    def mouseMoveEvent(self, e) -> None:
        f = max(0, min(self.app.n_frames() - 1, self._f(e.position().x())))
        if self._drag:
            self.app.seek(f)
        t = self._lane_at(e.position().y())
        tip = f"frame {f}  ·  {f / self.app.fps():.3f}s"
        if t is not None:
            rec = t.pts.get(f)
            state = "—" if rec is None else ("keyframe" if rec[3] == MANUAL else f"auto, match {rec[2]:.2f}")
            tip += f"\n{t.name}: {state}"
        QToolTip.showText(e.globalPosition().toPoint(), tip, self)

    def mouseReleaseEvent(self, e) -> None:
        self._drag = False


def _nice_step(n: int, w: int) -> int:
    target = max(1, n * 70 // max(1, w))
    for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000):
        if s >= target:
            return s
    return 20000
