"""Frame-exact video access with an LRU cache.

OpenCV's ffmpeg backend decodes everything the system wheel was built with
(H.264, HEVC from phones, MJPEG from high-speed cams, ...). Random access is
done with CAP_PROP_POS_FRAMES; sequential reads skip the seek entirely.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import cv2
import numpy as np


class VideoSource:
    def __init__(self, path: str, cache_mb: int = 1500):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not (1.0 <= self.fps <= 10000):
            self.fps = 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = n if n > 0 else self._count_frames()
        self._next = 0  # index the capture will return on the next read()
        self._lock = threading.RLock()
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        frame_bytes = max(1, self.width * self.height * 3)
        self._cache_max = max(16, (cache_mb * 1024 * 1024) // frame_bytes)
        # Some containers over-report the frame count; trim on first failed read.
        self._verified_end = False

    def _count_frames(self) -> int:
        cap = cv2.VideoCapture(self.path)
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        return max(n, 1)

    def frame(self, idx: int) -> np.ndarray | None:
        """BGR frame `idx`, or None if past the end."""
        idx = max(0, min(idx, self.frame_count - 1))
        with self._lock:
            img = self._cache.get(idx)
            if img is not None:
                self._cache.move_to_end(idx)
                return img
            # Short forward hop: decode through instead of seeking (exact and fast).
            if not (0 <= idx - self._next <= 12):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self._next = idx
            img = None
            while self._next <= idx:
                ok, f = self._cap.read()
                if not ok:
                    # Reported length was too long: clamp and retry the last frame.
                    if self._next > 0 and self._next < self.frame_count:
                        self.frame_count = self._next
                        self._next = 0
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        return self.frame(self.frame_count - 1)
                    return None
                self._put(self._next, f)
                img = f
                self._next += 1
            return img

    def _put(self, idx: int, img: np.ndarray) -> None:
        self._cache[idx] = img
        self._cache.move_to_end(idx)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def close(self) -> None:
        self._cap.release()


class SequentialReader:
    """Independent decoder for the tracking worker so it never fights the UI cache."""

    def __init__(self, path: str, shared: VideoSource | None = None):
        self.cap = cv2.VideoCapture(path)
        self.shared = shared
        self.next = 0

    def read(self, idx: int) -> np.ndarray | None:
        if idx != self.next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.next = idx
        ok, f = self.cap.read()
        if not ok:
            return None
        self.next = idx + 1
        return f

    def read_range(self, start: int, stop: int) -> list[np.ndarray]:
        """Frames [start, stop) decoded forward; used for backward tracking chunks."""
        out = []
        for i in range(start, stop):
            f = self.read(i)
            if f is None:
                break
            out.append(f)
        return out

    def close(self) -> None:
        self.cap.release()
