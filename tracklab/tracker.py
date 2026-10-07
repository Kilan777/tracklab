"""Point trackers + the run loop that drives them over a video.

Template tracker: normalised cross-correlation against two templates at once —
the previous frame's appearance (follows gradual change) and the keyframe the
user placed (pins it so it can't slowly drift off the marker). Search window
follows a constant-velocity prediction and grows with speed, so fast objects
don't fall out of it. A low match score means "I'm not sure" and the run stops
there instead of silently drifting the way Kinovea does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .model import MANUAL, Track
from .video import SequentialReader


def auto_radii(w: int, h: int) -> tuple[int, int]:
    tr = int(np.clip(round(min(w, h) * 0.018), 6, 40))
    return tr, max(3 * tr, 16)


def _prep(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(frame, (3, 3), 0)


def _patch(img: np.ndarray, x: float, y: float, r: int) -> np.ndarray:
    return cv2.getRectSubPix(img, (2 * r + 1, 2 * r + 1), (float(x), float(y))).astype(np.float32)


def _subpix(resp: np.ndarray, ix: int, iy: int) -> tuple[float, float]:
    """Parabolic peak interpolation around integer max."""
    dx = dy = 0.0
    h, w = resp.shape
    if 0 < ix < w - 1:
        l, c, r = resp[iy, ix - 1], resp[iy, ix], resp[iy, ix + 1]
        den = l - 2 * c + r
        if den < 0:
            dx = float(np.clip(0.5 * (l - r) / den, -0.5, 0.5))
    if 0 < iy < h - 1:
        u, c, d = resp[iy - 1, ix], resp[iy, ix], resp[iy + 1, ix]
        den = u - 2 * c + d
        if den < 0:
            dy = float(np.clip(0.5 * (u - d) / den, -0.5, 0.5))
    return dx, dy


def _rot(vx: float, vy: float, a: float) -> tuple[float, float]:
    c, s = math.cos(a), math.sin(a)
    return vx * c - vy * s, vx * s + vy * c


def _ncc1(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


# Tunables (see tests/pedal.py --sweep)
AMBIG_MARGIN = 0.1  # second candidate within this of the best => ambiguous
AMBIG_PENALTY = 0.25
BRIGHT_W = 1.0


class TemplateTracker:
    """NCC tracker with a motion model (velocity + turn rate, camera-shake compensated) and an
    appearance memory: keyframes the user placed plus confident views it has seen. Periodic
    motion (pedalling, running) repeats its appearances, so the memory keeps it locked on."""

    MAX_PINNED, MAX_AUTO = 8, 8

    def __init__(self, frame: np.ndarray, x: float, y: float, tr: int, sr: int):
        self.tr, self.sr = tr, sr
        self.bank: list[tuple[np.ndarray, bool]] = []  # (template, pinned by user)
        self.reinit(frame, x, y)

    # -- setup -------------------------------------------------------------------------------
    def reinit(self, frame: np.ndarray, x: float, y: float, keep_velocity: bool = False) -> None:
        img = _prep(frame)
        self.x, self.y = x, y
        self.missed = 0
        if not keep_velocity:
            self.vx = self.vy = self.turn = 0.0
            self.age = 0
            self.good_v = None
        self.last_good = (x, y, math.hypot(self.vx, self.vy))
        patch = _patch(img, x, y, self.tr)
        self.cur = patch.copy()
        self.flat = float(patch.std()) < 2.0  # textureless template: NCC is meaningless
        self.remember(patch, pinned=True)

    def seed_velocity(self, vx: float, vy: float, turn: float = 0.0) -> None:
        self.vx, self.vy, self.turn = vx, vy, turn
        self.good_v = (vx, vy, turn)
        self.last_good = (self.x, self.y, math.hypot(vx, vy))
        self.age = 2

    def remember(self, patch: np.ndarray, pinned: bool = False) -> None:
        if self.bank and not pinned:
            if max(_ncc1(patch, b) for b, _ in self.bank) > 0.88:
                return  # already know this look
            pins = [b for b, p in self.bank if p]
            if pins and max(_ncc1(patch, b) for b in pins) < 0.5:
                return  # too unlike anything the user confirmed: don't teach ourselves a mistake
        self.bank.append((patch, pinned))
        for flag, cap in ((True, self.MAX_PINNED), (False, self.MAX_AUTO)):
            idx = [i for i, (_, p) in enumerate(self.bank) if p == flag]
            if len(idx) > cap:
                del self.bank[idx[0]]

    # -- tracking ----------------------------------------------------------------------------
    def step(self, frame: np.ndarray, accept: float, g: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float, float, bool]:
        """Returns (x, y, conf, ok). `g` = camera motion since the last frame. Not ok => coasting."""
        img = _prep(frame)
        h, w = img.shape[:2]
        pvx, pvy = _rot(self.vx, self.vy, self.turn)
        px, py = self.x + g[0] + pvx, self.y + g[1] + pvy
        speed = math.hypot(pvx, pvy)
        if self.age < 2:
            sr = 3 * self.sr  # no idea of the velocity yet: look wide
        else:
            sr = max(self.sr, 1.2 * speed + self.tr)
        sr = int(min(sr * (1 + 0.3 * self.missed), 8 * self.sr))
        cx, cy = round(px), round(py)
        R = sr + self.tr
        region = _patch(img, cx, cy, R)
        r_cur = cv2.matchTemplate(region, self.cur, cv2.TM_CCOEFF_NORMED)
        r_bank = None
        for b, _ in self.bank:
            r = cv2.matchTemplate(region, b, cv2.TM_CCOEFF_NORMED)
            r_bank = r if r_bank is None else np.maximum(r_bank, r)
        # Confidence must agree with *verified* looks (user keyframes / confident views), not just
        # the last frame — otherwise a tracker that slid onto the background stays "confident".
        sim = 0.4 * r_cur + 0.6 * r_bank
        resp = sim
        # NCC ignores brightness, so a shaded look-alike (the other foot) scores the same. Penalise it.
        gray = region.mean(axis=2)
        k = 2 * self.tr + 1
        mean_map = cv2.blur(gray, (k, k))[self.tr:self.tr + 2 * sr + 1, self.tr:self.tr + 2 * sr + 1]
        bright = BRIGHT_W * np.abs(mean_map - float(self.cur.mean())) / 128.0
        yy, xx = np.mgrid[-sr:sr + 1, -sr:sr + 1]
        dist = np.sqrt(xx * xx + yy * yy) / max(sr, 1)
        resp = resp - bright - 0.12 * dist
        _, best, _, (ix, iy) = cv2.minMaxLoc(resp)
        # Ambiguity: a second, separate candidate nearly as good means "don't guess".
        sup = resp.copy()
        cv2.circle(sup, (ix, iy), max(3, int(self.tr * 1.2)), float(resp.min()), -1)
        second = float(sup.max())
        ambiguous = best - second < AMBIG_MARGIN
        dx, dy = _subpix(resp, ix, iy)
        nx, ny = cx + ix - sr + dx, cy + iy - sr + dy
        conf = float(sim[iy, ix]) - float(bright[iy, ix]) - (AMBIG_PENALTY if ambiguous else 0.0)
        if self.flat:
            conf = min(conf, 0.5)
        if not (0 <= nx < w and 0 <= ny < h):
            conf = 0.0
        if self.missed:
            # Re-acquiring after an occlusion: must clearly look like something remembered, and be somewhere
            # it could plausibly have got to — otherwise a look-alike across the frame gets grabbed.
            conf = min(conf, float(r_bank[iy, ix]))
            accept = max(accept + 0.1, 0.7)
            lx, ly, lv = self.last_good
            reach = 2 * self.sr + 1.2 * lv * self.missed
            if math.hypot(nx - g[0] * self.missed - lx, ny - g[1] * self.missed - ly) > reach:
                conf = min(conf, accept - 0.01)
        self.guess = (nx, ny)  # best candidate, offered to the user if we give up here
        if conf < accept:
            if self.missed == 0 and self.good_v is not None:
                # Going behind something: the visible part slows down/shifts, so the last few velocity
                # estimates are biased. Coast on the last velocity measured while clearly visible.
                self.x, self.y = self.x - pvx, self.y - pvy
                self.vx, self.vy, self.turn = self.good_v
                pvx, pvy = _rot(self.vx, self.vy, self.turn)
                px, py = self.x + g[0] + pvx, self.y + g[1] + pvy
            self.missed += 1
            self.x, self.y = px, py
            return px, py, conf, False
        steps = self.missed + 1
        mvx = (nx - g[0] - (self.x - pvx * self.missed)) / steps
        mvy = (ny - g[1] - (self.y - pvy * self.missed)) / steps
        # Turn rate (curved paths): angle between old and new velocity, smoothed.
        if math.hypot(self.vx, self.vy) > 2 and math.hypot(mvx, mvy) > 2 and self.age >= 1:
            da = math.atan2(mvy, mvx) - math.atan2(self.vy, self.vx)
            da = (da + math.pi) % (2 * math.pi) - math.pi
            self.turn = 0.5 * self.turn + 0.5 * float(np.clip(da, -1.2, 1.2))
        else:
            self.turn *= 0.5
        if self.age == 0:
            self.vx, self.vy = mvx, mvy
        else:
            self.vx, self.vy = 0.3 * pvx + 0.7 * mvx, 0.3 * pvy + 0.7 * mvy
        if conf > 0.85:
            self.good_v = (self.vx, self.vy, self.turn)
        self.x, self.y = nx, ny
        self.missed = 0
        self.age += 1
        self.last_good = (nx, ny, math.hypot(self.vx, self.vy))
        if conf > 0.7:
            patch = _patch(img, nx, ny, self.tr)
            self.cur = 0.5 * self.cur + 0.5 * patch
            if conf > 0.85:  # long-term memory only from clearly visible views (not half-hidden ones)
                self.remember(patch)
        return nx, ny, conf, True


class CSRTTracker:
    """OpenCV CSRT: slower, but copes with rotation/scale/deformation (whole objects, limbs)."""

    def __init__(self, frame: np.ndarray, x: float, y: float, tr: int, sr: int):
        self.tr = tr
        self.reinit(frame, x, y)

    def reinit(self, frame: np.ndarray, x: float, y: float, keep_velocity: bool = False) -> None:
        self.t = cv2.TrackerCSRT_create()
        r = self.tr
        self.t.init(frame, (int(round(x - r)), int(round(y - r)), 2 * r + 1, 2 * r + 1))
        self.ref = _patch(frame, x, y, r)
        self.x, self.y = x, y

    def seed_velocity(self, vx: float, vy: float, turn: float = 0.0) -> None:
        pass

    def remember(self, patch: np.ndarray, pinned: bool = False) -> None:
        pass

    def step(self, frame: np.ndarray, accept: float, g=(0.0, 0.0)) -> tuple[float, float, float, bool]:
        ok, (bx, by, bw, bh) = self.t.update(frame)
        if not ok:
            return self.x, self.y, 0.0, False
        x, y = bx + bw / 2, by + bh / 2
        p = _patch(frame, x, y, self.tr)
        ncc = float(cv2.matchTemplate(p, self.ref, cv2.TM_CCOEFF_NORMED)[0, 0])
        self.x, self.y = x, y
        # CSRT already said "ok"; NCC to the keyframe only lowers the reported confidence.
        return x, y, float(np.clip(0.5 + 0.5 * ncc, 0.5, 1.0)), True


def structure_radius(frame: np.ndarray, x: float, y: float, r0: int) -> int:
    """Smallest template radius (>= r0, <= 3*r0) whose patch has enough edges to be distinctive.

    A point inside a plain blob (a knee, a ball) gets a bigger patch that reaches its outline."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    R = 3 * r0
    roi = cv2.getRectSubPix(g, (2 * R + 1, 2 * R + 1), (float(x), float(y))).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(roi, cv2.CV_32F, 1, 0), cv2.Sobel(roi, cv2.CV_32F, 0, 1))
    strong = mag > max(40.0, float(np.percentile(mag, 75)))
    r = r0
    while r < R:
        win = strong[R - r:R + r + 1, R - r:R + r + 1]
        # Need edges in at least two directions' worth of the patch: rows and columns both covered.
        if win.mean() > 0.12 and win.any(axis=0).mean() > 0.5 and win.any(axis=1).mean() > 0.5:
            break
        r = int(r * 1.25) + 1
    return min(r, R)


class BlobTracker:
    """Follows a patch of colour (a ball, a coloured marker, a tumbling object): finds the region that has
    the reference colour near the predicted position and takes its centre. Shape and rotation don't matter."""

    def __init__(self, frame: np.ndarray, x: float, y: float, tr: int, sr: int):
        self.tr, self.sr = tr, sr
        self.reinit(frame, x, y)

    def _lab(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)

    def reinit(self, frame: np.ndarray, x: float, y: float, keep_velocity: bool = False) -> None:
        R = 3 * self.tr
        lab = self._lab(cv2.getRectSubPix(frame, (2 * R + 1, 2 * R + 1), (float(x), float(y))))
        # Object = the colour around the click; background = the window's border.
        c = lab[R - 2:R + 3, R - 2:R + 3].reshape(-1, 3).mean(axis=0)
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        d_obj = np.linalg.norm(lab - c, axis=2)
        d_bg = float(np.median(np.linalg.norm(border - c, axis=1)))
        self.thr = max(8.0, 0.5 * d_bg)
        n, lab_i, st, cen = cv2.connectedComponentsWithStats((d_obj < self.thr).astype(np.uint8), 8)
        k = lab_i[R, R] if lab_i[R, R] > 0 else (1 + int(np.argmax(st[1:, 4])) if n > 1 else 0)
        self.color = lab[lab_i == k].mean(axis=0) if k > 0 else c
        self.area = float(st[k, 4]) if k > 0 else float(np.pi * self.tr ** 2)
        if k > 0:
            x, y = x - R + cen[k][0], y - R + cen[k][1]
        self.x, self.y = x, y
        if not keep_velocity:
            self.vx = self.vy = 0.0
            self.age = 0
        self.missed = 0
        self.guess = None

    def seed_velocity(self, vx: float, vy: float, turn: float = 0.0) -> None:
        self.vx, self.vy, self.age = vx, vy, 2

    def remember(self, patch: np.ndarray, pinned: bool = False) -> None:
        pass

    def step(self, frame: np.ndarray, accept: float, g=(0.0, 0.0)) -> tuple[float, float, float, bool]:
        px, py = self.x + g[0] + self.vx, self.y + g[1] + self.vy
        speed = math.hypot(self.vx, self.vy)
        sr = int(min((3 * self.sr if self.age < 2 else max(self.sr, 1.5 * speed + 2 * self.tr)) * (1 + 0.3 * self.missed),
                     8 * self.sr))
        R = sr + 2 * self.tr
        lab = self._lab(cv2.getRectSubPix(frame, (2 * R + 1, 2 * R + 1), (float(px), float(py))))
        mask = (np.linalg.norm(lab - self.color, axis=2) < self.thr * 1.3).astype(np.uint8)
        n, lab_i, st, cen = cv2.connectedComponentsWithStats(mask, 8)
        best, bscore = None, -1.0
        for k in range(1, n):
            a = float(st[k, 4])
            if a < 0.15 * self.area or a > 6 * self.area:
                continue
            d = math.hypot(cen[k][0] - R, cen[k][1] - R) / max(R, 1)
            size = 1 - min(1.0, abs(math.log(a / self.area)) / math.log(6))
            score = 0.6 * size + 0.4 * (1 - min(1.0, d))
            if score > bscore:
                best, bscore = k, score
        h, w = frame.shape[:2]
        if best is None:
            conf, nx, ny = 0.0, px, py
        else:
            nx, ny = px - R + cen[best][0], py - R + cen[best][1]
            conf = float(bscore)
            if not (0 <= nx < w and 0 <= ny < h):
                conf = 0.0
        self.guess = (nx, ny)
        if conf < accept:
            self.missed += 1
            self.x, self.y = px, py
            return px, py, conf, False
        steps = self.missed + 1
        mvx = (nx - g[0] - (self.x - self.vx * self.missed)) / steps
        mvy = (ny - g[1] - (self.y - self.vy * self.missed)) / steps
        self.vx, self.vy = (mvx, mvy) if self.age == 0 else (0.3 * self.vx + 0.7 * mvx, 0.3 * self.vy + 0.7 * mvy)
        self.x, self.y, self.missed = nx, ny, 0
        self.age += 1
        a = float(st[best, 4])
        self.area = 0.8 * self.area + 0.2 * a  # objects change size a little (perspective, tumbling)
        return nx, ny, conf, True


def make_tracker(track: Track, frame: np.ndarray, x: float, y: float):
    h, w = frame.shape[:2]
    atr, asr = auto_radii(w, h)
    tr = track.template_r or structure_radius(frame, x, y, atr)
    sr = track.search_r or max(asr, 2 * tr)
    if track.method == "blob":
        return BlobTracker(frame, x, y, track.template_r or atr, sr)
    cls = CSRTTracker if track.method == "csrt" else TemplateTracker
    return cls(frame, x, y, tr, sr)


@dataclass
class RunResult:
    last_frame: int
    lost: list[tuple[int, int]]  # (track id, frame)
    cancelled: bool
    frames_done: int
    guesses: dict | None = None  # track id -> (x, y) best candidate on its lost frame


def run_tracking(
    path: str,
    tracks: list[Track],
    start: int,
    direction: int,
    n_frames: int,
    emit: Callable[[int, list[tuple[int, int, float, float, float]]], None],
    cancelled: Callable[[], bool],
    threshold: float = 0.65,
    stop_at: int | None = None,
    stop_on_loss: bool = True,
    bridge: int = 10,
    stab: dict | None = None,
) -> RunResult:
    """Track `tracks` from `start` (where each must have a point) one frame at a time.

    `emit(frame, [(track_id, frame, x, y, conf), ...])` is called per frame with new
    auto points. Manual keyframes encountered on the way re-seed their tracker.
    `stop_at` (inclusive) bounds the run; by default it goes to the end / start.
    `bridge`: frames a lost point may coast on its predicted path and be re-acquired
    (brief occlusions). Bridged frames are left empty, never invented.
    """
    reader = SequentialReader(path)
    wp = _Warp(stab) if stab else None
    end = (n_frames - 1) if direction > 0 else 0
    if stop_at is not None:
        end = min(end, stop_at) if direction > 0 else max(end, stop_at)

    frames_iter = _frames(reader, start, end, direction)
    try:
        first = next(frames_iter)[1]
    except StopIteration:
        reader.close()
        return RunResult(start, [], False, 0)
    if wp:
        first = wp.frame(start, first)
        if first is None:
            reader.close()
            return RunResult(start, [(t.id, start) for t in tracks], False, 0)

    active: dict[int, tuple[Track, object]] = {}
    for t in tracks:
        p = t.at(start)
        if p is not None and not t.static:
            if wp:
                p = wp.to_ref(start, *p)
            tk = make_tracker(t, first, *p)
            _seed(tk, t, start, direction, path, wp)
            active[t.id] = (t, tk)
    lost: list[tuple[int, int]] = []
    missing: dict[int, int] = {}  # track id -> first frame it went missing
    guesses: dict[int, tuple[float, float]] = {}
    f = start
    done = 0
    cam = CameraMotion(first)
    for f, frame in frames_iter:
        if cancelled():
            reader.close()
            return RunResult(f - direction, lost, True, done, guesses)
        if wp:
            # Track in the reference frame's view: camera pans/zooms vanish, the scene holds still.
            frame = wp.frame(f, frame)
        g = cam.update(frame) if frame is not None else (0.0, 0.0)
        out = []
        for tid, (t, tk) in list(active.items()):
            existing = t.pts.get(f)
            if frame is None:  # camera pose unknown (view blocked): counts as missing
                gone = missing.setdefault(tid, f)
                if abs(f - gone) >= bridge:
                    lost.append((tid, gone))
                    del active[tid]
                continue
            if existing and existing[3] == MANUAL:
                ex, ey = wp.to_ref(f, existing[0], existing[1]) if wp else existing[:2]
                tk.reinit(frame, ex, ey, keep_velocity=True)
                continue
            x, y, conf, ok = tk.step(frame, threshold, g)
            if ok and wp:
                x, y = wp.from_ref(f, x, y)
            if not ok:
                if tid not in missing and getattr(tk, "guess", None):
                    guesses[tid] = wp.from_ref(f, *tk.guess) if wp else tk.guess
                gone = missing.setdefault(tid, f)
                if abs(f - gone) >= bridge:
                    lost.append((tid, gone))
                    del active[tid]
                continue
            missing.pop(tid, None)
            out.append((tid, f, x, y, conf))
        emit(f, out)
        done += 1
        if not active or (lost and stop_on_loss):
            break
    reader.close()
    # Still coasting at the end of the range counts as lost where it went missing.
    lost += [(tid, gone) for tid, gone in missing.items() if tid in active]
    return RunResult(f, lost, False, done, guesses)


class _Warp:
    """Maps frames/points into the reference frame's view using the per-frame camera homographies."""

    def __init__(self, stab: dict):
        self.H = {int(k): np.asarray(v, float).reshape(3, 3) for k, v in stab.items()}
        self.size = None

    def frame(self, f: int, img: np.ndarray) -> np.ndarray | None:
        H = self.H.get(f)
        if H is None:
            return None
        if self.size is None:
            self.size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, H, self.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def to_ref(self, f: int, x: float, y: float) -> tuple[float, float]:
        q = self.H[f] @ np.array([x, y, 1.0])
        return q[0] / q[2], q[1] / q[2]

    def from_ref(self, f: int, x: float, y: float) -> tuple[float, float]:
        q = np.linalg.solve(self.H[f], np.array([x, y, 1.0]))
        return q[0] / q[2], q[1] / q[2]


class CameraMotion:
    """Frame-to-frame camera translation (handheld shake/pans) by phase correlation on a small copy."""

    W = 480

    def __init__(self, frame: np.ndarray):
        self.scale = max(1.0, frame.shape[1] / self.W)
        self.prev = self._small(frame)
        self.win = cv2.createHanningWindow(self.prev.shape[::-1], cv2.CV_32F)

    def _small(self, frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = g.shape
        return cv2.resize(g, (int(w / self.scale), int(h / self.scale)), interpolation=cv2.INTER_AREA).astype(np.float32)

    def update(self, frame: np.ndarray) -> tuple[float, float]:
        cur = self._small(frame)
        (dx, dy), resp = cv2.phaseCorrelate(self.prev, cur, self.win)
        self.prev = cur
        if resp < 0.15 or math.hypot(dx, dy) > cur.shape[1] * 0.2:
            return 0.0, 0.0  # no clear global motion (or a cut)
        return dx * self.scale, dy * self.scale


def _seed(tk, t: Track, start: int, direction: int, path: str, wp: "_Warp | None" = None) -> None:
    """Give a fresh tracker what we already know: recent velocity and the look of other keyframes."""

    def at(f):
        p = t.pts.get(f)
        if p is None or not wp:
            return p
        if f not in wp.H:
            return None
        return wp.to_ref(f, p[0], p[1])

    p0, p1, p2 = at(start), at(start - direction), at(start - 2 * direction)
    if p0 and p1:
        vx, vy = p0[0] - p1[0], p0[1] - p1[1]
        turn = 0.0
        if p2:
            ux, uy = p1[0] - p2[0], p1[1] - p2[1]
            if math.hypot(ux, uy) > 2 and math.hypot(vx, vy) > 2:
                turn = (math.atan2(vy, vx) - math.atan2(uy, ux) + math.pi) % (2 * math.pi) - math.pi
                turn = float(np.clip(turn, -1.2, 1.2))
        tk.seed_velocity(*_rot(vx, vy, 0.0), turn)
    keys = sorted((f for f, v in t.pts.items() if v[3] == MANUAL and f != start and (not wp or f in wp.H)),
                  key=lambda f: abs(f - start))
    keys = keys[: TemplateTracker.MAX_PINNED - 1]
    if not keys or not isinstance(tk, TemplateTracker):
        return
    rd = SequentialReader(path)
    current = tk.bank[-1]
    for f in sorted(keys):
        img = rd.read(f)
        if img is not None:
            x, y = t.pts[f][0], t.pts[f][1]
            if wp:
                img = wp.frame(f, img)
                x, y = wp.to_ref(f, x, y)
            tk.remember(_patch(_prep(img), x, y, tk.tr), pinned=True)
    rd.close()
    # Keep the start keyframe as the newest pinned entry.
    tk.bank.remove(current)
    tk.bank.append(current)


def _frames(reader: SequentialReader, start: int, end: int, direction: int):
    """Yield (idx, frame) from start to end inclusive; backwards via forward-decoded chunks."""
    if direction > 0:
        for i in range(start, end + 1):
            f = reader.read(i)
            if f is None:
                return
            yield i, f
        return
    CH = 48
    hi = start + 1
    while hi > end:
        lo = max(end, hi - CH)
        chunk = reader.read_range(lo, hi)
        for k in range(len(chunk) - 1, -1, -1):
            yield lo + k, chunk[k]
        hi = lo


def snap_to_feature(frame: np.ndarray, x: float, y: float, r: int) -> tuple[float, float]:
    """Refine a click onto the marker under/near it (blob centroid), else a nearby corner, else the click."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    H, W = g.shape
    R = 2 * r  # look wider than the snap distance so a marker isn't clipped by the window
    x0, y0 = max(0, int(round(x)) - R), max(0, int(round(y)) - R)
    x1, y1 = min(W, int(round(x)) + R + 1), min(H, int(round(y)) + R + 1)
    roi = cv2.GaussianBlur(g[y0:y1, x0:x1], (3, 3), 0)
    if roi.shape[0] < 7 or roi.shape[1] < 7:
        return x, y
    cx, cy = x - x0, y - y0
    t, bright = cv2.threshold(roi, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi, lo = roi[bright > 0], roi[bright == 0]
    if hi.size and lo.size and float(hi.mean()) - float(lo.mean()) > 30:
        best = None
        for mask in (bright, 1 - bright):
            n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            for i in range(1, n):
                bx, by, bw, bh, area = stats[i]
                if area < 5 or bx == 0 or by == 0 or bx + bw >= roi.shape[1] or by + bh >= roi.shape[0]:
                    continue  # touches the ROI edge: background or something bigger than a marker
                ys, xs = np.nonzero(lab == i)
                d = float(np.min(np.hypot(xs - cx, ys - cy)))
                if d <= r * 0.6 and area <= np.pi * r * r * 1.5 and (best is None or d < best[0] or (d == best[0] and area > best[1])):
                    best = (d, area, (lab == i).astype(np.uint8))
        if best is not None:
            m = best[2]
            # Fill holes so ring-shaped markers give their true centre.
            ff = np.pad(1 - m, 1, constant_values=1).astype(np.uint8)
            cv2.floodFill(ff, None, (0, 0), 0)
            m = m | ff[1:-1, 1:-1]
            mo = cv2.moments(m, binaryImage=True)
            if mo["m00"] > 0:
                return x0 + mo["m10"] / mo["m00"], y0 + mo["m01"] / mo["m00"]
    pts = cv2.goodFeaturesToTrack(roi, 5, 0.05, 3)
    if pts is not None:
        pts = pts[:, 0]
        k = int(np.argmin(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
        if np.hypot(pts[k, 0] - cx, pts[k, 1] - cy) <= r * 0.5:
            c = pts[k:k + 1].astype(np.float32).reshape(1, 1, 2)
            cv2.cornerSubPix(roi.astype(np.float32), c, (3, 3), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01))
            return x0 + float(c[0, 0, 0]), y0 + float(c[0, 0, 1])
    return x, y
