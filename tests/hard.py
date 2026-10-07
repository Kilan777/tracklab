"""Harder clip: motion blur, lighting flicker, rotating textured target, compression noise."""
import math, sys, time
import cv2, numpy as np
sys.path.insert(0, ".")
from tracklab.model import Project
from tracklab.tracker import run_tracking

path = "tests/hard.mp4"
rng = np.random.default_rng(1)
w, h, n, fps = 1920, 1080, 180, 120
bg = cv2.GaussianBlur((rng.random((h, w, 3)) * 255).astype(np.uint8), (0, 0), 6)
bg = cv2.normalize(bg, None, 60, 190, cv2.NORM_MINMAX)
tex = cv2.resize((rng.random((12, 12, 3)) * 255).astype(np.uint8), (60, 60), interpolation=cv2.INTER_NEAREST)
vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
gt = []
for i in range(n):
    t = i / fps
    img = bg.copy()
    # marker on a swinging "limb": fast, blurred
    a = 0.9 * math.sin(2 * math.pi * 1.2 * t)
    mx, my = 960 + 420 * math.sin(a), 300 + 420 * math.cos(a)
    # rotating textured patch drifting right
    rx, ry, ang = 400 + 300 * t, 800 - 80 * t, 120 * t
    M = cv2.getRotationMatrix2D((30, 30), ang, 1.0)
    rot = cv2.warpAffine(tex, M, (60, 60), borderMode=cv2.BORDER_REFLECT)
    img[int(ry) - 30:int(ry) + 30, int(rx) - 30:int(rx) + 30] = rot
    layer = np.zeros_like(img)
    cv2.circle(layer, (int(mx * 16), int(my * 16)), 14 * 16, (255, 255, 255), -1, cv2.LINE_AA, 4)
    # motion blur along velocity
    vx = 420 * math.cos(a) * 0.9 * 2 * math.pi * 1.2 * math.cos(2 * math.pi * 1.2 * t) / fps
    vy = -420 * math.sin(a) * 0.9 * 2 * math.pi * 1.2 * math.cos(2 * math.pi * 1.2 * t) / fps
    L = max(1, int(math.hypot(vx, vy)))
    k = np.zeros((2 * L + 1, 2 * L + 1), np.float32)
    cv2.line(k, (L - int(vx / 2), L - int(vy / 2)), (L + int(vx / 2), L + int(vy / 2)), 1, 1)
    k /= k.sum()
    layer = cv2.filter2D(layer, -1, k)
    alpha = layer.astype(np.float32) / 255
    img = (img * (1 - alpha) + 240 * alpha).astype(np.float32)
    img *= 0.8 + 0.2 * math.sin(t * 9)  # flicker
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    vw.write(img)
    gt.append(((mx, my), (rx, ry)))
vw.release()
gt = np.array(gt)
for method in ("template", "csrt"):
    p = Project()
    ts = []
    for k, nm in enumerate(["marker", "patch"]):
        tr = p.add_track(nm); tr.method = method; tr.set(0, *gt[0, k])
        if nm == "patch": tr.template_r = 26
        ts.append(tr)
    def emit(f, pts):
        for tid, fr, x, y, c in pts: p.track(tid).set(fr, x, y, c, 0)
    t0 = time.time()
    r = run_tracking(path, ts, 0, 1, n, emit, lambda: False, stop_on_loss=False)
    dt = time.time() - t0
    print(f"--- {method}: {r.frames_done} frames in {dt:.1f}s ({r.frames_done/dt:.0f} fps)  lost={r.lost}")
    for k, t in enumerate(ts):
        fr = [f for f in t.frames() if f > 0]
        err = np.array([np.hypot(*(np.array(t.pts[f][:2]) - gt[f, k])) for f in fr])
        print(f"  {t.name}: {len(fr)}/{n-1} frames, mean err {err.mean():.2f}px, p95 {np.percentile(err,95):.2f}, max {err.max():.2f}, min conf {min(t.pts[f][2] for f in fr):.2f}")
