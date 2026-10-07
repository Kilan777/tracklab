"""Video canvas: zoom/pan, overlays, direct manipulation, precision loupe."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..render import draw_overlays, label

if TYPE_CHECKING:
    from .main import MainWindow

HIT_R = 12  # screen px


def bgr_to_qimage(img: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    q = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    q._keep = rgb  # keep buffer alive
    return q


class Canvas(QWidget):
    zoomChanged = Signal(float)

    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        self.img: QImage | None = None
        self.zoom = 1.0
        self.off = QPointF(0, 0)
        self.auto_fit = True
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 200)
        self.setAcceptDrops(True)
        self._pan_start: tuple[QPointF, QPointF] | None = None
        self._pan_moved = False
        self._drag = None  # ("track", tid) | ("cal_line", i) | ("quad", i) | ("origin",) | ("axis",)
        self._drag_moved = False
        self.hover_track: int | None = None
        self.mouse_img: QPointF | None = None
        self.mouse_screen = QPointF(0, 0)  # last pointer pos (QCursor.pos() is unreliable on Wayland)
        self.loupe = False
        self.space_down = False

    # ---------- coordinates ----------
    def to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(self.off.x() + x * self.zoom, self.off.y() + y * self.zoom)

    def to_img(self, p: QPointF) -> QPointF:
        return QPointF((p.x() - self.off.x()) / self.zoom, (p.y() - self.off.y()) / self.zoom)

    def set_image(self, img: QImage | None) -> None:
        first = self.img is None or (img is not None and img.size() != self.img.size())
        self.img = img
        if first or self.auto_fit:
            self.fit()
        self.update()

    def fit(self) -> None:
        if self.img is None:
            return
        w, h = self.img.width(), self.img.height()
        z = min(self.width() / w, self.height() / h)
        self.zoom = z
        self.off = QPointF((self.width() - w * z) / 2, (self.height() - h * z) / 2)
        self.auto_fit = True
        self.zoomChanged.emit(self.zoom)
        self.update()

    def zoom_at(self, factor: float, at: QPointF) -> None:
        if self.img is None:
            return
        z = float(np.clip(self.zoom * factor, 0.05, 64))
        ip = self.to_img(at)
        self.zoom = z
        self.off = QPointF(at.x() - ip.x() * z, at.y() - ip.y() * z)
        self.auto_fit = False
        self.zoomChanged.emit(self.zoom)
        self.update()

    def focus(self, x: float, y: float, factor: float) -> None:
        """Zoom to `factor` × the fit zoom, centred on an image point."""
        if self.img is None:
            return
        fz = min(self.width() / self.img.width(), self.height() / self.img.height())
        self.zoom = fz * factor
        self.off = QPointF(self.width() / 2 - x * self.zoom, self.height() / 2 - y * self.zoom)
        self.auto_fit = False
        self.zoomChanged.emit(self.zoom)
        self.update()

    def center_on(self, x: float, y: float) -> None:
        """Pan (not zoom) so an image point is visible — used when jumping to a lost point."""
        q = self.to_screen(x, y)
        m = 60
        if m < q.x() < self.width() - m and m < q.y() < self.height() - m:
            return
        self.off = QPointF(self.width() / 2 - x * self.zoom, self.height() / 2 - y * self.zoom)
        self.auto_fit = False
        self.update()

    def resizeEvent(self, e) -> None:
        if self.auto_fit:
            self.fit()

    # ---------- painting ----------
    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0d0f13"))
        app = self.app
        if self.img is None:
            p.setPen(QColor("#8a93a6"))
            f = p.font()
            f.setPointSize(14)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, "Drop a video here  ·  or press Ctrl+O")
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 3)
        p.drawImage(QRectF(self.off, QPointF(self.off.x() + self.img.width() * self.zoom,
                                             self.off.y() + self.img.height() * self.zoom)), self.img)
        sel = app.project.track(app.selected) if app.selected else None
        boxes = app.track_radii(sel) if sel else None
        draw_overlays(p, app.project, app.frame, self.to_screen, 1.0,
                      selected=app.selected, hover=self.hover_track, show_boxes=boxes, zoom=self.zoom,
                      pending=app.pending_refs)
        self._paint_guess(p)
        self._paint_tool_preview(p)
        if self.loupe and self.mouse_img is not None:
            self._paint_loupe(p)

    def _paint_guess(self, p: QPainter) -> None:
        g = self.app.guess
        if not g or self.app.place_target != g[0] or self.app.frame != g[1]:
            return
        q = self.to_screen(g[2], g[3])
        pen = QPen(QColor("#ffffff"), 2, Qt.DashLine)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(q, 11, 11)
        label(p, q + QPointF(15, 14), "guess — Enter to accept", "#ffffff", 0.9)

    def _paint_tool_preview(self, p: QPainter) -> None:
        app = self.app
        pts = app.calib_pending
        if not pts or self.mouse_img is None:
            return
        pen = QPen(QColor("#ffd23f"), 1.5, Qt.DashLine)
        pen.setCosmetic(True)
        p.setPen(pen)
        shown = app.project.from_ref(pts, app.frame)  # pending clicks are kept in calibration-frame pixels
        if shown is None:
            return
        q = [self.to_screen(*pt) for pt in shown] + [self.to_screen(self.mouse_img.x(), self.mouse_img.y())]
        for a, b in zip(q, q[1:]):
            p.drawLine(a, b)
        names = {"plane": ["TL", "TR", "BR", "BL"], "circle": ["top", "front", "bottom", "back"]}.get(app.tool, [])
        for i, a in enumerate(q[:-1]):
            p.drawEllipse(a, 4, 4)
            if names:
                label(p, a + QPointF(8, -10), names[i], "#ffd23f")

    def _paint_loupe(self, p: QPainter) -> None:
        S = 170
        LZ = max(6.0, self.zoom * 3)
        mp = self.mouse_screen
        x = mp.x() + 28 if mp.x() + 28 + S < self.width() else mp.x() - 28 - S
        y = mp.y() + 28 if mp.y() + 28 + S < self.height() else mp.y() - 28 - S
        box = QRectF(x, y, S, S)
        c = self.mouse_img
        half = S / 2 / LZ
        src = QRectF(c.x() - half, c.y() - half, 2 * half, 2 * half)
        p.save()
        p.setClipRect(box)
        p.fillRect(box, QColor("#000"))
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        p.drawImage(box, self.img, src)
        # overlays inside the loupe
        def m(ix: float, iy: float) -> QPointF:
            return QPointF(box.x() + (ix - src.x()) * LZ, box.y() + (iy - src.y()) * LZ)
        app = self.app
        draw_overlays(p, app.project, app.frame, m, 0.8, selected=app.selected)
        pen = QPen(QColor(255, 255, 255, 200), 1)
        p.setPen(pen)
        cx, cy = box.center().x(), box.center().y()
        p.drawLine(QPointF(cx - 12, cy), QPointF(cx - 3, cy))
        p.drawLine(QPointF(cx + 3, cy), QPointF(cx + 12, cy))
        p.drawLine(QPointF(cx, cy - 12), QPointF(cx, cy - 3))
        p.drawLine(QPointF(cx, cy + 3), QPointF(cx, cy + 12))
        p.restore()
        p.setPen(QPen(QColor("#4da3ff"), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(box, 6, 6)
        label(p, QPointF(box.x() + 4, box.bottom() - 10), f"{c.x():.1f}, {c.y():.1f}", "#cfd6e4", 0.85)

    # ---------- hit testing ----------
    def hit(self, sp: QPointF):
        app = self.app
        best, bd = None, HIT_R
        for t in app.project.tracks:
            if not t.visible:
                continue
            pt = app.project.pos(t, app.frame)
            if pt is None:
                continue
            d = _d(self.to_screen(*pt), sp)
            if d < bd:
                best, bd = ("track", t.id), d
        cal = app.project.calib
        proj = app.project

        def here(pt):  # calibration handles live on the calibration frame; show them where they are now
            q = proj.from_ref([pt], app.frame)
            return None if q is None else self.to_screen(*q[0])

        handles = []
        if cal.mode == "line":
            handles += [(("cal_line", i), pt) for i, pt in enumerate(cal.line)]
        if cal.mode in ("plane", "circle"):
            handles += [(("quad", i), pt) for i, pt in enumerate(cal.quad)]
        if cal.mode not in ("plane", "circle") and cal.origin is not None:
            handles.append((("origin",), cal.origin))
        for key, pt in handles:
            q = here(pt)
            if q is not None and _d(q, sp) < bd:
                best, bd = key, _d(q, sp)
        if cal.mode not in ("plane", "circle") and cal.origin is not None:
            o = here(cal.origin)
            if o is not None:
                ax = o + QPointF(math.cos(cal.axis_angle) * 60, math.sin(cal.axis_angle) * 60)
                if _d(ax, sp) < bd:
                    best, bd = ("axis",), _d(ax, sp)
        return best

    # ---------- mouse ----------
    def mousePressEvent(self, e) -> None:
        self.setFocus()
        sp = e.position()
        if e.button() in (Qt.MiddleButton, Qt.RightButton) or (e.button() == Qt.LeftButton and self.space_down):
            self._pan_start = (sp, QPointF(self.off))
            self._pan_moved = False
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() != Qt.LeftButton or self.img is None:
            return
        ip = self.to_img(sp)
        self.mouse_screen = sp
        app = self.app
        h = self.hit(sp)
        if app.tool in ("select", "point") and h is not None and not app.place_target:
            if h[0] == "track":
                app.select_track(h[1])
            app.push_undo()
            self._drag = h
            self._drag_moved = False
            self.loupe = True
            self.setCursor(Qt.BlankCursor)
            self.mouse_img = ip
            self.update()
            return
        app.canvas_click(ip.x(), ip.y(), e.modifiers())

    def mouseMoveEvent(self, e) -> None:
        sp = e.position()
        ip = self.to_img(sp)
        self.mouse_img = ip
        self.mouse_screen = sp
        app = self.app
        if self._pan_start:
            d = sp - self._pan_start[0]
            if abs(d.x()) + abs(d.y()) > 3:
                self._pan_moved = True
            self.off = self._pan_start[1] + d
            self.auto_fit = False
            self.update()
            return
        if self._drag:
            self._drag_moved = True
            app.drag_to(self._drag, ip.x(), ip.y(), e.modifiers())
            self.update()
            return
        h = self.hit(sp) if self.img is not None else None
        ht = h[1] if h and h[0] == "track" else None
        if ht != self.hover_track:
            self.hover_track = ht
        placing = app.tool in ("point", "angle", "segment", "distance", "line", "plane", "circle", "origin") or app.place_target
        self.loupe = bool(placing) and self.img is not None
        if h is not None and app.tool in ("select", "point") and not app.place_target:
            self.setCursor(Qt.OpenHandCursor)
        elif placing:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        app.show_cursor_pos(ip)
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        if self._pan_start:
            moved = self._pan_moved
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
            if not moved and e.button() == Qt.RightButton:
                self.app.context_menu(self.to_img(e.position()), self.hit(e.position()), e.globalPosition().toPoint())
            return
        if self._drag:
            d, moved = self._drag, self._drag_moved
            self._drag = None
            self.loupe = False
            self.setCursor(Qt.OpenHandCursor)
            self.app.drag_done(d, moved)
            self.update()

    def mouseDoubleClickEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.hit(e.position()) is None and self.app.tool == "select":
            self.fit()

    def wheelEvent(self, e) -> None:
        dy = e.angleDelta().y()
        if dy:
            self.zoom_at(1.0015 ** dy, e.position())

    def leaveEvent(self, e) -> None:
        self.mouse_img = None
        self.loupe = False
        self.update()

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            # Space is play/pause; holding it while dragging pans.
            self.space_down = True
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self.space_down = False
        super().keyReleaseEvent(e)

    # ---------- drag & drop ----------
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        urls = e.mimeData().urls()
        if urls:
            self.app.open_path(urls[0].toLocalFile())


def _d(a: QPointF, b: QPointF) -> float:
    return math.hypot(a.x() - b.x(), a.y() - b.y())
