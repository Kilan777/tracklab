"""Camera-motion compensation for handheld footage.

For every frame, estimate the homography that maps it onto a reference frame (the one the calibration was
drawn on), from ORB features + RANSAC. Moving things (riders' legs, passers-by) are rejected as outliers; a
handheld camera mostly rotates/zooms, which a homography describes exactly. Frames where the view is
blocked get no entry, and measurements on them are left empty rather than guessed.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

WORK_W = 960  # features are found on a downscaled copy


class _Feat:
    def __init__(self, n: int = 3000):
        # SIFT: slower than ORB but far more repeatable/precise, which is what measurements need.
        self.det = cv2.SIFT_create(nfeatures=n, contrastThreshold=0.02)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def detect(self, gray: np.ndarray, mask: np.ndarray | None = None):
        return self.det.detectAndCompute(gray, mask)

    def homography(self, a, b, min_inliers: int) -> tuple[np.ndarray | None, int]:
        """Homography mapping keypoints of `a` onto `b` (each a (kp, des) pair)."""
        (ka, da), (kb, db) = a, b
        if da is None or db is None or len(ka) < 20 or len(kb) < 20:
            return None, 0
        pairs = self.bf.knnMatch(da, db, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
        if len(good) < min_inliers:
            return None, 0
        src = np.float32([ka[m.queryIdx].pt for m in good])
        dst = np.float32([kb[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 1.5, maxIters=5000, confidence=0.9999)
        n = int(mask.sum()) if mask is not None else 0
        if H is None or n < min_inliers or not _sane(H):
            return None, n
        return H, n


def default_roi(pts: np.ndarray, w: int, h: int) -> tuple[float, float, float, float]:
    """Region around the calibration object: its bounding box grown to at least ~45% of the frame."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    cx, cy = pts.mean(axis=0)
    bw = max(np.ptp(pts[:, 0]) * 3, 0.45 * w)
    bh = max(np.ptp(pts[:, 1]) * 2, 0.45 * h)
    return (max(0, cx - bw / 2), max(0, cy - bh / 2), min(w, cx + bw / 2), min(h, cy + bh / 2))


def _sane(H: np.ndarray) -> bool:
    """Reject wild homographies (degenerate fits): limited zoom, shear and perspective."""
    A = H[:2, :2] / H[2, 2]
    s = np.linalg.svd(A, compute_uv=False)
    return 0.4 < s[1] and s[0] < 2.5 and s[0] / s[1] < 1.6 and abs(H[2, 0]) < 2e-3 and abs(H[2, 1]) < 2e-3


def compute(
    path: str,
    ref: int,
    n_frames: int,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    min_inliers: int = 25,
    roi: tuple[float, float, float, float] | None = None,
) -> dict[int, list[float]]:
    """{frame: row-major 3x3 homography frame->ref (full-resolution pixels)}; blocked frames are omitted.

    `roi` (x0, y0, x1, y1 on the reference frame) limits matching to the part of the scene that matters —
    the plane being measured. With a handheld camera that also moves, near and far things shift by different
    amounts (parallax), so aligning on the busy background would misplace the near object."""
    cap = cv2.VideoCapture(path)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    k = WORK_W / w if w > WORK_W else 1.0
    S = np.diag([k, k, 1.0])
    Si = np.diag([1 / k, 1 / k, 1.0])
    feat = _Feat()

    def small(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k != 1.0 else g

    cap.set(cv2.CAP_PROP_POS_FRAMES, ref)
    ok, img = cap.read()
    if not ok:
        return {}
    hs, ws = small(img).shape
    poly = None
    if roi is not None:
        x0, y0, x1, y1 = roi
        poly = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) * k

    def mask_for(Hs_prev):
        """ROI on the reference, carried into the current frame by the last known camera pose."""
        if poly is None:
            return None
        q = poly if Hs_prev is None else cv2.perspectiveTransform(poly[None], np.linalg.inv(Hs_prev))[0]
        m = np.zeros((hs, ws), np.uint8)
        cv2.fillConvexPoly(m, q.astype(np.int32), 255)
        return m

    ref_f = feat.detect(small(img), mask_for(None))
    ref_full = feat.detect(small(img), None) if poly is not None else ref_f
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out: dict[int, list[float]] = {}
    prev = None  # (features, H_small frame->ref) of the previous frame
    last_H = None  # last good pose, to place the ROI
    for f in range(n_frames):
        if cancelled and cancelled():
            break
        ok, img = cap.read()
        if not ok:
            break
        cur = feat.detect(small(img), mask_for(last_H))
        if f == ref:
            Hs = np.eye(3)
        else:
            Hs, n = feat.homography(cur, ref_f, min_inliers)
            if Hs is None and prev is not None:
                # Too different from the reference (zoomed/panned far): chain through the last frame that worked.
                Hp, n = feat.homography(cur, prev[0], min_inliers)
                Hs = prev[1] @ Hp if Hp is not None else None
                if Hs is not None and not _sane(Hs):
                    Hs = None
            if Hs is None and poly is not None:
                # Lost track of where the region is (e.g. the camera was re-aimed while something blocked the
                # view): look at the whole frame to find our place again, then go back to the region.
                full = feat.detect(small(img), None)
                Hs, n = feat.homography(full, ref_full, min_inliers * 2)
        if Hs is not None:
            H = Si @ Hs @ S
            out[f] = (H / H[2, 2]).ravel().tolist()
            last_H = Hs
            prev = (cur, Hs)
        if progress and f % 10 == 0:
            progress(f, n_frames)
    cap.release()
    return out
