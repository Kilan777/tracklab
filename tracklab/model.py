"""Project data: tracks, measurements, calibration. Pure data + JSON (de)serialisation."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field

import numpy as np

# Categorical order validated for CVD separation on a dark surface (dataviz reference palette, dark steps).
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

# Point status flags
AUTO, MANUAL, INTERP = 0, 1, 2


@dataclass
class Track:
    id: int
    name: str
    color: str
    # frame -> (x, y, confidence, kind)   kind: AUTO | MANUAL
    pts: dict[int, tuple[float, float, float, int]] = field(default_factory=dict)
    static: bool = False  # a fixed point: one position for every frame
    method: str = "template"  # template | csrt
    template_r: int = 0  # half-size of template in px; 0 = auto from video size
    search_r: int = 0  # half-size of search window; 0 = auto
    visible: bool = True
    lost_at: int | None = None  # frame where the tracker last gave up

    def at(self, f: int) -> tuple[float, float] | None:
        if self.static:
            if not self.pts:
                return None
            p = next(iter(self.pts.values()))
            return p[0], p[1]
        p = self.pts.get(f)
        return (p[0], p[1]) if p else None

    def set(self, f: int, x: float, y: float, conf: float = 1.0, kind: int = MANUAL) -> None:
        if self.static:
            self.pts = {0: (x, y, 1.0, MANUAL)}
        else:
            self.pts[f] = (float(x), float(y), float(conf), kind)

    def frames(self) -> list[int]:
        return sorted(self.pts)

    def next_manual_after(self, f: int) -> int | None:
        c = [k for k, v in self.pts.items() if k > f and v[3] == MANUAL]
        return min(c) if c else None

    def prev_manual_before(self, f: int) -> int | None:
        c = [k for k, v in self.pts.items() if k < f and v[3] == MANUAL]
        return max(c) if c else None

    def anchor_before(self, f: int) -> int | None:
        """Nearest frame <= f that has a point (start for forward tracking)."""
        c = [k for k in self.pts if k <= f]
        return max(c) if c else None

    def anchor_after(self, f: int) -> int | None:
        c = [k for k in self.pts if k >= f]
        return min(c) if c else None

    def series(self, n_frames: int) -> np.ndarray:
        """(n_frames, 2) array of pixel positions, NaN where missing."""
        out = np.full((n_frames, 2), np.nan)
        if self.static:
            p = self.at(0)
            if p:
                out[:] = p
            return out
        for f, p in self.pts.items():
            if 0 <= f < n_frames:
                out[f] = p[0], p[1]
        return out


@dataclass
class Measure:
    """Angle (3 refs: a, vertex, b), segment angle (2 refs vs horizontal) or distance (2 refs)."""

    id: int
    kind: str  # "angle" | "segment" | "distance"
    refs: list[int]  # track ids
    name: str = ""
    color: str = "#ffffff"
    visible: bool = True
    # angle: signed (0..360 going counter-clockwise on screen) or unsigned (0..180)
    signed: bool = False


@dataclass
class Calibration:
    mode: str = "none"  # none | line | plane | circle
    # line: two image points and the real length between them
    line: list[list[float]] = field(default_factory=list)
    length: float = 1.0
    # plane: 4 image points (TL, TR, BR, BL of a real rectangle) + its real width/height
    quad: list[list[float]] = field(default_factory=list)
    width: float = 1.0
    height: float = 1.0
    # circle: quad holds 4 points on a circle seen at an angle (a wheel): top, right, bottom, left of the rim.
    # Corrects for the object being turned away from the camera; origin = bottom (where the wheel meets the ground).
    diameter: float = 0.68
    frame: int = 0  # video frame the calibration was drawn on (matters when the camera moves)
    note: str = ""  # how it was made, e.g. "from gravity, frames 120–145"
    unit: str = "m"
    # coordinate system (image px). Line mode: origin + x-axis angle. Plane mode uses quad[3] (BL) as origin.
    origin: list[float] | None = None
    axis_angle: float = 0.0  # radians, image x-axis rotation (line mode)
    y_up: bool = True

    def scale(self) -> float:
        """Real units per pixel (line mode)."""
        if self.mode == "line" and len(self.line) == 2:
            d = math.dist(self.line[0], self.line[1])
            return self.length / d if d > 1e-9 else 1.0
        return 1.0

    def homography(self) -> np.ndarray | None:
        if self.mode not in ("plane", "circle") or len(self.quad) != 4:
            return None
        import cv2

        src = np.array(self.quad, np.float32)
        if self.mode == "plane":
            w, h = self.width, self.height
            # TL, TR, BR, BL in a y-up world with origin at BL
            dst = np.array([[0, h], [w, h], [w, 0], [0, 0]], np.float32)
        else:
            r = self.diameter / 2
            # top, right, bottom, left of the wheel; origin at the bottom (ground contact), y up
            dst = np.array([[0, 2 * r], [r, r], [0, 0], [-r, r]], np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    def is_set(self) -> bool:
        return (self.mode == "line" and len(self.line) == 2) or (self.mode in ("plane", "circle") and len(self.quad) == 4)

    def to_world(self, xy: np.ndarray) -> np.ndarray:
        """Map (N,2) pixel coords (of the calibration frame) to world coords. Uncalibrated => pixels, y-up."""
        xy = np.asarray(xy, float)
        if self.mode in ("plane", "circle"):
            H = self.homography()
            if H is not None:
                pts = np.c_[xy, np.ones(len(xy))] @ H.T
                with np.errstate(invalid="ignore", divide="ignore"):
                    return pts[:, :2] / pts[:, 2:3]
        s = self.scale() if self.mode == "line" else 1.0
        o = np.array(self.origin if self.origin else [0.0, 0.0])
        d = xy - o
        c, sn = math.cos(-self.axis_angle), math.sin(-self.axis_angle)
        rx = d[:, 0] * c - d[:, 1] * sn
        ry = d[:, 0] * sn + d[:, 1] * c
        if self.y_up:
            ry = -ry
        return np.c_[rx, ry] * s

    def unit_label(self) -> str:
        return self.unit if self.is_set() else "px"


@dataclass
class Project:
    video_path: str = ""
    capture_fps: float = 0.0  # 0 => use file fps. Set for slow-mo footage.
    tracks: list[Track] = field(default_factory=list)
    measures: list[Measure] = field(default_factory=list)
    calib: Calibration = field(default_factory=Calibration)
    filter_hz: float = 0.0  # 0 => auto
    trail: int = 30
    # Example projects carry a title, a one-line blurb and a guided tour (list of steps, see ui/tour.py).
    title: str = ""
    blurb: str = ""
    tour: list = field(default_factory=list)
    # Moving camera: per-frame homography (row-major 3x3) mapping that frame's pixels onto the calibration frame.
    # Empty = camera assumed still. A frame missing from a non-empty map = camera pose unknown (view blocked).
    stab: dict = field(default_factory=dict)
    _next_id: int = 1

    # ---- pixel -> world, honouring camera motion ----
    def _H(self, f: int) -> np.ndarray | None:
        h = self.stab.get(int(f))
        return None if h is None else np.asarray(h, float).reshape(3, 3)

    def to_ref(self, xy: np.ndarray, frames) -> np.ndarray:
        """Pixels on frame(s) `frames` -> pixels on the calibration frame (identity when the camera is still)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if not self.stab:
            return xy
        fr = np.broadcast_to(np.asarray(frames), (len(xy),))
        out = np.full_like(xy, np.nan)
        for f in np.unique(fr):
            H = self._H(f)
            if H is None:
                continue
            m = fr == f
            q = np.c_[xy[m], np.ones(m.sum())] @ H.T
            out[m] = q[:, :2] / q[:, 2:3]
        return out

    def from_ref(self, xy: np.ndarray, f: int) -> np.ndarray | None:
        """Calibration-frame pixels -> pixels on frame f (to draw calibration overlays on a moving camera)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if not self.stab:
            return xy
        H = self._H(f)
        if H is None:
            return None
        q = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(H).T
        return q[:, :2] / q[:, 2:3]

    def pos(self, t: "Track", f: int) -> tuple[float, float] | None:
        """Where a point is drawn on frame f. Fixed points live in calibration-frame coordinates when the
        camera moves, so they stay stuck to the scene."""
        p = t.at(f)
        if p is None or not (t.static and self.stab):
            return p
        q = self.from_ref([p], f)
        return None if q is None else (float(q[0, 0]), float(q[0, 1]))

    def world(self, xy: np.ndarray, frames) -> np.ndarray:
        """Pixels on the given frame(s) -> world coordinates."""
        return self.calib.to_world(self.to_ref(xy, frames))

    def new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def track(self, tid: int) -> Track | None:
        return next((t for t in self.tracks if t.id == tid), None)

    def add_track(self, name: str | None = None, static: bool = False) -> Track:
        tid = self.new_id()
        n = sum(1 for t in self.tracks)
        t = Track(tid, name or f"P{n + 1}", PALETTE[n % len(PALETTE)], static=static)
        self.tracks.append(t)
        return t

    def remove_track(self, tid: int) -> None:
        self.tracks = [t for t in self.tracks if t.id != tid]
        self.measures = [m for m in self.measures if tid not in m.refs]

    def add_measure(self, kind: str, refs: list[int]) -> Measure:
        mid = self.new_id()
        names = [self.track(r).name for r in refs if self.track(r)]
        label = {"angle": "∠", "segment": "⦣", "distance": "↔"}[kind]
        m = Measure(mid, kind, refs, name=f"{label} {'-'.join(names)}", color=PALETTE[(len(self.measures) + 3) % len(PALETTE)])
        self.measures.append(m)
        return m

    def snapshot(self) -> dict:
        return copy.deepcopy(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "video_path": self.video_path,
            "capture_fps": self.capture_fps,
            "filter_hz": self.filter_hz,
            "trail": self.trail,
            "next_id": self._next_id,
            "calib": self.calib.__dict__,
            "tracks": [
                {**{k: v for k, v in t.__dict__.items() if k != "pts"},
                 "pts": [[f, *p] for f, p in sorted(t.pts.items())]}
                for t in self.tracks
            ],
            "measures": [m.__dict__ for m in self.measures],
            "title": self.title,
            "blurb": self.blurb,
            "tour": self.tour,
            "stab": {str(k): v for k, v in self.stab.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        p = cls()
        p.video_path = d.get("video_path", "")
        p.capture_fps = d.get("capture_fps", 0.0)
        p.filter_hz = d.get("filter_hz", 0.0)
        p.trail = d.get("trail", 30)
        p._next_id = d.get("next_id", 1)
        p.calib = Calibration(**d.get("calib", {}))
        for td in d.get("tracks", []):
            td = dict(td)
            pts = td.pop("pts", [])
            t = Track(**td)
            t.pts = {int(r[0]): (float(r[1]), float(r[2]), float(r[3]), int(r[4])) for r in pts}
            p.tracks.append(t)
        p.measures = [Measure(**m) for m in d.get("measures", [])]
        p.title = d.get("title", "")
        p.blurb = d.get("blurb", "")
        p.tour = d.get("tour", [])
        p.stab = {int(k): v for k, v in d.get("stab", {}).items()}
        return p

    def by_name(self, name: str):
        """Track or measure with this name (tours refer to things by name)."""
        return next((t for t in self.tracks if t.name == name), None) or \
            next((m for m in self.measures if m.name == name), None)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path) as f:
            return cls.from_dict(json.load(f))
