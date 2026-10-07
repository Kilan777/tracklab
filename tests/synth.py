"""Synthetic clip with known ground truth: fast circular marker, projectile, occluded marker."""
import math, sys
import cv2, numpy as np

def make(path, n=240, w=1280, h=720, fps=60):
    rng = np.random.default_rng(0)
    bg = (rng.random((h, w, 3)) * 60 + 80).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    for _ in range(40):
        cv2.rectangle(bg, tuple(rng.integers(0, [w, h])), tuple(rng.integers(0, [w, h])), tuple(int(c) for c in rng.integers(40, 200, 3)), 2)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    gt = []
    for i in range(n):
        t = i / fps
        img = bg.copy()
        # circle: 1.5 rev/s radius 200 -> ~31 px/frame
        a = 2 * math.pi * 1.5 * t
        c = (640 + 200 * math.cos(a), 360 + 200 * math.sin(a))
        # projectile
        p = (100 + 300 * t, 650 - 500 * t + 0.5 * 600 * t * t)
        # slow drifter with an occluder bar crossing at frames 150-160
        d = (900 + 60 * math.sin(t * 2), 150 + 40 * t)
        for (x, y), col in ((c, (0, 0, 255)), (p, (0, 255, 255)), (d, (255, 255, 255))):
            cv2.circle(img, (int(x * 16), int(y * 16)), 11 * 16, (20, 20, 20), -1, cv2.LINE_AA, 4)
            cv2.circle(img, (int(x * 16), int(y * 16)), 7 * 16, col, -1, cv2.LINE_AA, 4)
        if 150 <= i <= 160:
            cv2.rectangle(img, (780, 0), (1000, 720), (90, 90, 90), -1)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        vw.write(img)
        gt.append((c, p, d))
    vw.release()
    return gt

if __name__ == "__main__":
    gt = make(sys.argv[1])
    np.save(sys.argv[1] + ".gt.npy", np.array(gt))
