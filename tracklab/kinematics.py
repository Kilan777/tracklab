"""Filtering + derivatives for tracked series (gap-aware, zero-phase)."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, filtfilt

from .model import Measure, Project


def segments(valid: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, stop) runs where valid is True."""
    out, start = [], None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(valid)))
    return out


def lowpass(y: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """2nd-order Butterworth applied forward+backward (=4th order, zero lag) per gap-free segment."""
    out = y.copy()
    if fc <= 0 or fc >= fs / 2:
        return out
    b, a = butter(2, fc / (fs / 2))
    for s, e in segments(~np.isnan(y)):
        n = e - s
        if n >= 3 * max(len(a), len(b)) + 1:
            out[s:e] = filtfilt(b, a, y[s:e], padtype="odd", padlen=min(n - 1, 3 * max(len(a), len(b)) * 3))
    return out


def auto_cutoff(y: np.ndarray, fs: float) -> float:
    """Residual analysis (Winter): pick the cutoff where filtered residuals hit the noise floor."""
    seg = max(segments(~np.isnan(y)), key=lambda s: s[1] - s[0], default=None)
    if seg is None or seg[1] - seg[0] < 20:
        return 0.0
    x = y[seg[0]:seg[1]]
    fcs = np.linspace(max(0.5, fs * 0.005), fs * 0.4, 60)
    res = np.array([np.sqrt(np.mean((x - lowpass(x, fs, fc)) ** 2)) for fc in fcs])
    # Fit a line to the high-frequency (noise) tail and extrapolate to fc=0.
    tail = slice(len(fcs) * 2 // 3, None)
    k, c = np.polyfit(fcs[tail], res[tail], 1)
    target = max(c, 1e-12)
    idx = np.argmax(res <= target)
    if res[idx] > target:
        return float(fcs[-1])
    return float(fcs[idx])


def derivative(y: np.ndarray, dt: float) -> np.ndarray:
    d = np.full_like(y, np.nan)
    for s, e in segments(~np.isnan(y)):
        if e - s >= 2:
            d[s:e] = np.gradient(y[s:e], dt)
    return d


def unwrap_deg(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for s, e in segments(~np.isnan(a)):
        out[s:e] = np.degrees(np.unwrap(np.radians(a[s:e])))
    return out


class Kinematics:
    """Everything the plots/CSV need, computed once per project change."""

    def __init__(self, project: Project, n_frames: int, fps: float):
        self.p = project
        self.n = n_frames
        self.fps = project.capture_fps or fps
        self.dt = 1.0 / self.fps
        self.t = np.arange(n_frames) * self.dt
        self.unit = project.calib.unit_label()
        self.cutoff_used: dict[str, float] = {}
        self.tracks: dict[int, dict[str, np.ndarray]] = {}
        self.measures: dict[int, dict[str, np.ndarray]] = {}
        for t in project.tracks:
            self.tracks[t.id] = self._track(t)
        for m in project.measures:
            self.measures[m.id] = self._measure(m)

    def _filter(self, y: np.ndarray, key: str) -> np.ndarray:
        fc = self.p.filter_hz
        if fc < 0:  # filtering disabled
            return y
        if fc == 0:
            fc = auto_cutoff(y, self.fps)
        self.cutoff_used[key] = fc
        return lowpass(y, self.fps, fc) if fc > 0 else y

    def _track(self, t) -> dict[str, np.ndarray]:
        px = t.series(self.n)
        valid = ~np.isnan(px[:, 0])
        w = np.full_like(px, np.nan)
        if valid.any():
            if t.static:  # fixed points are stored in calibration-frame pixels
                w[valid] = self.p.calib.to_world(px[valid])
            else:  # through the camera motion, if any (frames with unknown camera pose stay empty)
                w[valid] = self.p.world(px[valid], np.flatnonzero(valid))
        if t.static:
            x, y = w[:, 0], w[:, 1]
            z = np.zeros(self.n)
            return dict(px=px, x_raw=x, y_raw=y, x=x, y=y, vx=z, vy=z, speed=z, ax=z, ay=z, accel=z)
        x = self._filter(w[:, 0], f"{t.name} x")
        y = self._filter(w[:, 1], f"{t.name} y")
        vx, vy = derivative(x, self.dt), derivative(y, self.dt)
        ax, ay = derivative(vx, self.dt), derivative(vy, self.dt)
        return dict(px=px, x_raw=w[:, 0], y_raw=w[:, 1], x=x, y=y, vx=vx, vy=vy,
                    speed=np.hypot(vx, vy), ax=ax, ay=ay, accel=np.hypot(ax, ay))

    def _pos(self, tid: int) -> np.ndarray:
        k = self.tracks.get(tid)
        if k is None:
            return np.full((self.n, 2), np.nan)
        return np.c_[k["x"], k["y"]]

    def _measure(self, m: Measure) -> dict[str, np.ndarray]:
        P = [self._pos(r) for r in m.refs]
        if m.kind == "distance":
            v = np.hypot(*(P[1] - P[0]).T)
            return dict(value=v, rate=derivative(v, self.dt))
        if m.kind == "segment":
            d = P[1] - P[0]
            v = unwrap_deg(np.degrees(np.arctan2(d[:, 1], d[:, 0])))
        else:
            a, b = P[0] - P[1], P[2] - P[1]
            ang = np.degrees(np.arctan2(b[:, 1], b[:, 0]) - np.arctan2(a[:, 1], a[:, 0]))
            if m.signed:
                v = unwrap_deg(np.mod(ang, 360.0))
            else:
                v = np.abs((ang + 180.0) % 360.0 - 180.0)
        return dict(value=v, rate=derivative(v, self.dt))

    def measure_value_px(self, m: Measure, frame: int) -> float | None:
        v = self.measures.get(m.id, {}).get("value")
        if v is None or not (0 <= frame < self.n) or math.isnan(v[frame]):
            return None
        return float(v[frame])
