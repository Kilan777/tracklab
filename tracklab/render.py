"""Overlay drawing shared by the live canvas and the video exporter.

Everything is drawn in screen space through `to_screen` so line widths and
labels stay crisp regardless of zoom; `s` scales sizes for exports.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF

from .model import MANUAL, Measure, Project

Mapper = Callable[[float, float], QPointF]


def _pen(c: QColor | str, w: float, style=Qt.SolidLine) -> QPen:
    p = QPen(QColor(c), w, style)
    p.setCapStyle(Qt.RoundCap)
    p.setJoinStyle(Qt.RoundJoin)
    p.setCosmetic(True)
    return p


def label(p: QPainter, pos: QPointF, text: str, color: str, s: float = 1.0, align_left: bool = True) -> None:
    f = QFont(p.font())
    f.setPointSizeF(9 * s)
    f.setBold(True)
    p.setFont(f)
    fm = p.fontMetrics()
    w, h = fm.horizontalAdvance(text) + 8 * s, fm.height() + 2 * s
    x = pos.x() if align_left else pos.x() - w
    r = QRectF(x, pos.y() - h / 2, w, h)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(15, 17, 22, 200))
    p.drawRoundedRect(r, 4 * s, 4 * s)
    p.setPen(QColor(color))
    p.drawText(r, Qt.AlignCenter, text)


def fmt(v: float, unit: str) -> str:
    a = abs(v)
    d = 0 if a >= 1000 else 1 if a >= 100 else 2 if a >= 1 else 3
    return f"{v:.{d}f} {unit}"


def draw_overlays(
    p: QPainter,
    proj: Project,
    frame: int,
    m: Mapper,
    s: float = 1.0,
    selected: int | None = None,
    hover: int | None = None,
    show_boxes: tuple[int, int] | None = None,  # (template_r, search_r) for the selected track
    zoom: float = 1.0,
    pending: list[int] | None = None,  # track ids picked so far for a measure being built
) -> None:
    p.setRenderHint(QPainter.Antialiasing)
    if proj.stab:
        def mc(x: float, y: float, _m=m) -> QPointF:
            q = proj.from_ref([[x, y]], frame)
            return _m(*q[0]) if q is not None else _m(float("nan"), float("nan"))
        if proj.from_ref([[0, 0]], frame) is not None:
            _draw_calibration(p, proj, mc, s)
    else:
        _draw_calibration(p, proj, m, s)
    _draw_trails(p, proj, frame, m, s, selected)
    _draw_measures(p, proj, frame, m, s)
    for t in proj.tracks:
        if not t.visible:
            continue
        pt = proj.pos(t, frame)
        sel = t.id == selected
        c = QColor(t.color)
        if pt is None:
            # Show where it was lost so the user knows what to click on.
            if t.lost_at == frame:
                prev = proj.pos(t, frame - 1) or proj.pos(t, frame + 1)
                if prev:
                    q = m(*prev)
                    p.setPen(_pen("#ff3b3b", 2 * s, Qt.DashLine))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(q, 16 * s, 16 * s)
                    label(p, q + QPointF(20 * s, -14 * s), f"{t.name}: last seen here", "#ff6b6b", s)
            continue
        q = m(*pt)
        if sel and show_boxes and not t.static:
            tr, sr = show_boxes
            for r, style in ((tr, Qt.SolidLine), (sr, Qt.DotLine)):
                a, b = m(pt[0] - r, pt[1] - r), m(pt[0] + r, pt[1] + r)
                p.setPen(_pen(QColor(255, 255, 255, 150), 1, style))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(a, b))
        rec = t.pts.get(frame) if not t.static else None
        conf = rec[2] if rec else 1.0
        manual = (rec[3] == MANUAL) if rec else True
        r = (8 if sel or t.id == hover else 6) * s
        p.setPen(_pen(QColor(0, 0, 0, 180), 4 * s))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(q, r, r)
        ring = c if conf >= 0.75 else QColor("#ffae42")
        p.setPen(_pen(ring, 2 * s))
        if t.static:
            p.drawRect(QRectF(q.x() - r, q.y() - r, 2 * r, 2 * r))
        else:
            p.drawEllipse(q, r, r)
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        if manual:
            p.drawPolygon(QPolygonF([q + QPointF(0, -3 * s), q + QPointF(3 * s, 0), q + QPointF(0, 3 * s), q + QPointF(-3 * s, 0)]))
        else:
            p.drawEllipse(q, 1.6 * s, 1.6 * s)
        if sel or t.id == hover or (pending and t.id in pending):
            label(p, q + QPointF(r + 5 * s, -r - 4 * s), t.name, t.color, s)
        if pending and t.id in pending:
            p.setPen(_pen("#ffffff", 2 * s, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(q, r + 5 * s, r + 5 * s)


def _on_frame(proj: Project, t, frames: list[int], frame: int) -> dict[int, tuple[float, float]]:
    """Positions of `t` on `frames`, expressed in frame `frame`'s pixels (trails stay put when the camera moves)."""
    if not frames:
        return {}
    xy = np.array([t.pts[f][:2] for f in frames], float)
    if not proj.stab:
        return {f: (q[0], q[1]) for f, q in zip(frames, xy)}
    cur = proj.from_ref(proj.to_ref(xy, np.array(frames)), frame)
    if cur is None:
        return {}
    return {f: (q[0], q[1]) for f, q in zip(frames, cur) if not np.isnan(q).any()}


def _draw_trails(p: QPainter, proj: Project, frame: int, m: Mapper, s: float, selected: int | None) -> None:
    for t in proj.tracks:
        if not t.visible or t.static or len(t.pts) < 2:
            continue
        c = QColor(t.color)
        fr = sorted(t.pts)
        pos = _on_frame(proj, t, fr, frame)
        # Whole trajectory faintly (gaps break the line), recent trail bright.
        faint = QColor(c)
        faint.setAlpha(70 if t.id == selected else 40)
        _poly(p, pos, fr, m, _pen(faint, 1.2 * s))
        if proj.trail > 0:
            recent = [f for f in fr if frame - proj.trail <= f <= frame and f in pos]
            if len(recent) >= 2:
                n = len(recent)
                prev = None
                for i, f in enumerate(recent):
                    q = m(*pos[f])
                    if prev is not None and f - prev[0] == 1:
                        cc = QColor(c)
                        cc.setAlpha(int(60 + 195 * i / n))
                        p.setPen(_pen(cc, 2.2 * s))
                        p.drawLine(prev[1], q)
                    prev = (f, q)


def _poly(p: QPainter, pos: dict, frames: list[int], m: Mapper, pen: QPen) -> None:
    path = QPainterPath()
    last = None
    for f in frames:
        if f not in pos:
            last = None
            continue
        q = m(*pos[f])
        if last is not None and f - last == 1:
            path.lineTo(q)
        else:
            path.moveTo(q)
        last = f
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)


def measure_now(proj: Project, me: Measure, frame: int) -> float | None:
    """Unfiltered value of a measure on one frame, in world units / degrees."""
    pts = []
    for r in me.refs:
        t = proj.track(r)
        pt = proj.pos(t, frame) if t else None
        if pt is None:
            return None
        pts.append(pt)
    w = proj.world(np.array(pts, float), frame)
    if np.isnan(w).any():
        return None
    if me.kind == "distance":
        return float(np.hypot(*(w[1] - w[0])))
    if me.kind == "segment":
        d = w[1] - w[0]
        return float(math.degrees(math.atan2(d[1], d[0])))
    a, b = w[0] - w[1], w[2] - w[1]
    ang = math.degrees(math.atan2(b[1], b[0]) - math.atan2(a[1], a[0]))
    return ang % 360.0 if me.signed else abs((ang + 180.0) % 360.0 - 180.0)


def _draw_measures(p: QPainter, proj: Project, frame: int, m: Mapper, s: float) -> None:
    unit = proj.calib.unit_label()
    for me in proj.measures:
        if not me.visible:
            continue
        pts = []
        for r in me.refs:
            t = proj.track(r)
            pt = proj.pos(t, frame) if t else None
            if pt is None:
                break
            pts.append(pt)
        if len(pts) != len(me.refs):
            continue
        v = measure_now(proj, me, frame)
        Q = [m(*pt) for pt in pts]
        c = QColor(me.color)
        if me.kind == "distance":
            p.setPen(_pen(c, 2 * s))
            p.drawLine(Q[0], Q[1])
            if v is not None:
                label(p, (Q[0] + Q[1]) / 2 + QPointF(6 * s, -10 * s), fmt(v, unit), me.color, s)
            continue
        if me.kind == "segment":
            v0, a = Q[0], Q[1]
            ref_len = max(30 * s, min(80 * s, math.dist((v0.x(), v0.y()), (a.x(), a.y()))))
            p.setPen(_pen(QColor(255, 255, 255, 120), 1, Qt.DashLine))
            p.drawLine(v0, v0 + QPointF(ref_len, 0))
            arms = (v0 + QPointF(ref_len, 0), a)
        else:
            v0, arms = Q[1], (Q[0], Q[2])
        p.setPen(_pen(c, 2 * s))
        if me.kind == "angle":
            p.drawLine(v0, arms[0])
        p.drawLine(v0, arms[1])
        if v is None:
            continue
        a1 = math.degrees(math.atan2(-(arms[0].y() - v0.y()), arms[0].x() - v0.x()))
        a2 = math.degrees(math.atan2(-(arms[1].y() - v0.y()), arms[1].x() - v0.x()))
        span = (a2 - a1) % 360
        if me.kind == "angle" and not me.signed and span > 180:
            span -= 360
        if me.kind == "segment" and span > 180:
            span -= 360
        rad = 28 * s
        rect = QRectF(v0.x() - rad, v0.y() - rad, 2 * rad, 2 * rad)
        fill = QColor(c)
        fill.setAlpha(60)
        p.setBrush(fill)
        p.setPen(_pen(c, 1.5 * s))
        p.drawPie(rect, int(a1 * 16), int(span * 16))
        mid = math.radians(a1 + span / 2)
        lp = v0 + QPointF(math.cos(mid) * (rad + 14 * s), -math.sin(mid) * (rad + 14 * s))
        label(p, lp, f"{v:.1f}°", me.color, s, align_left=math.cos(mid) >= 0)


def _draw_calibration(p: QPainter, proj: Project, m: Mapper, s: float) -> None:
    cal = proj.calib
    if cal.mode == "line" and len(cal.line) == 2:
        a, b = m(*cal.line[0]), m(*cal.line[1])
        p.setPen(_pen("#ffd23f", 2 * s))
        p.drawLine(a, b)
        d = b - a
        L = math.hypot(d.x(), d.y()) or 1
        n = QPointF(-d.y() / L, d.x() / L) * 7 * s
        for q in (a, b):
            p.drawLine(q - n, q + n)
        label(p, (a + b) / 2 + n * 2, fmt(cal.length, cal.unit), "#ffd23f", s)
    elif cal.mode == "plane" and len(cal.quad) >= 2:
        q = [m(*pt) for pt in cal.quad]
        p.setPen(_pen("#ffd23f", 2 * s))
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(QPolygonF(q + ([q[0]] if len(q) == 4 else [])))
        H = cal.homography()
        if H is not None:
            # Perspective grid so you can eyeball that the plane is right.
            Hi = np.linalg.inv(H)
            p.setPen(_pen(QColor(255, 210, 63, 90), 1))
            N = 8
            for i in range(1, N):
                for line in (
                    [(cal.width * i / N, 0), (cal.width * i / N, cal.height)],
                    [(0, cal.height * i / N), (cal.width, cal.height * i / N)],
                ):
                    w = np.array([[*line[0], 1], [*line[1], 1]]) @ Hi.T
                    w = w[:, :2] / w[:, 2:3]
                    p.drawLine(m(*w[0]), m(*w[1]))
            label(p, q[3] + QPointF(8 * s, 14 * s), f"{fmt(cal.width, cal.unit)} × {fmt(cal.height, cal.unit)}", "#ffd23f", s)
    elif cal.mode == "circle" and len(cal.quad) >= 2:
        q = [m(*pt) for pt in cal.quad]
        p.setPen(_pen("#ffd23f", 2 * s))
        p.setBrush(QColor("#ffd23f"))
        for pt in q:
            p.drawEllipse(pt, 3.5 * s, 3.5 * s)
        H = cal.homography()
        if H is not None:
            # The wheel as the app understands it: a true circle in the bike's plane, drawn back in perspective.
            Hi = np.linalg.inv(H)
            r = cal.diameter / 2
            a = np.linspace(0, 2 * math.pi, 73)
            w = np.c_[r * np.cos(a), r + r * np.sin(a), np.ones_like(a)] @ Hi.T
            w = w[:, :2] / w[:, 2:3]
            p.setBrush(Qt.NoBrush)
            p.setPen(_pen("#ffd23f", 2 * s))
            p.drawPolyline(QPolygonF([m(*pt) for pt in w]))
            p.setPen(_pen(QColor(255, 210, 63, 120), 1, Qt.DashLine))
            p.drawLine(q[0], q[2])
            p.drawLine(q[1], q[3])
            label(p, q[1] + QPointF(8 * s, 0), f"⌀ {fmt(cal.diameter, cal.unit)}", "#ffd23f", s)
    _draw_axes(p, proj, m, s)


def _draw_axes(p: QPainter, proj: Project, m: Mapper, s: float) -> None:
    cal = proj.calib
    if cal.mode in ("plane", "circle") and len(cal.quad) == 4:
        H = cal.homography()
        if H is None:
            return
        Hi = np.linalg.inv(H)
        ax, ay = (cal.width * 0.25, cal.height * 0.25) if cal.mode == "plane" else (cal.diameter * 0.35,) * 2
        w = np.array([[0, 0, 1], [ax, 0, 1], [0, ay, 1]]) @ Hi.T
        w = w[:, :2] / w[:, 2:3]
        O, X, Y = m(*w[0]), m(*w[1]), m(*w[2])
    elif cal.origin is not None:
        o = cal.origin
        L = 60
        ca, sa = math.cos(cal.axis_angle), math.sin(cal.axis_angle)
        O = m(*o)
        X = m(o[0] + L * ca / max(1e-9, _zoom_of(m)) , o[1] + L * sa / max(1e-9, _zoom_of(m)))
        yd = -1 if cal.y_up else 1
        Y = m(o[0] - yd * L * sa / _zoom_of(m), o[1] + yd * L * ca / _zoom_of(m))
    else:
        return
    for end, col, name in ((X, "#ff5c5c", "x"), (Y, "#3ddc97", "y")):
        p.setPen(_pen(col, 2 * s))
        p.drawLine(O, end)
        d = end - O
        L = math.hypot(d.x(), d.y()) or 1
        u = d / L
        nrm = QPointF(-u.y(), u.x())
        p.setBrush(QColor(col))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([end, end - u * 9 * s + nrm * 4 * s, end - u * 9 * s - nrm * 4 * s]))
        label(p, end + u * 6 * s + QPointF(0, 0), name, col, s, align_left=u.x() >= -0.2)
    p.setBrush(QBrush(QColor("#ffffff")))
    p.setPen(_pen("#000000", 1))
    p.drawEllipse(O, 3.5 * s, 3.5 * s)


def _zoom_of(m: Mapper) -> float:
    a, b = m(0, 0), m(100, 0)
    return max(1e-6, (b.x() - a.x()) / 100 if abs(b.x() - a.x()) > 1e-9 else math.hypot(b.x() - a.x(), b.y() - a.y()) / 100)
