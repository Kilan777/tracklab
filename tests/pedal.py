"""Pedalling benchmark with ground truth + a simulated user who clicks the right spot whenever tracking stops.

Metrics that matter for usability:
  clicks  – how many times the user had to step in
  silent  – frames that were accepted but wrong (> ERR px) — the thing that makes Kinovea untrustworthy
"""
import math
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from tracklab.model import MANUAL, Project  # noqa: E402
from tracklab.tracker import run_tracking  # noqa: E402

ERR = 12.0


def ik(hip, foot, l1, l2):
    d = np.subtract(foot, hip)
    L = min(np.hypot(*d), l1 + l2 - 1e-3)
    a = math.atan2(d[1], d[0])
    c = (l1 * l1 + L * L - l2 * l2) / (2 * l1 * L)
    b = math.acos(np.clip(c, -1, 1))
    return hip[0] + l1 * math.cos(a - b), hip[1] + l1 * math.sin(a - b)  # knee forward (to the right)


def make(path, n=264, w=1280, h=720, fps=24, cadence=2.2, seed=3):
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur((rng.random((h + 80, w + 80, 3)) * 255).astype(np.uint8), (0, 0), 4)
    bg = cv2.normalize(bg, None, 50, 200, cv2.NORM_MINMAX)
    for _ in range(60):  # clutter, incl. white-ish blobs that look like shoes
        c = tuple(int(v) for v in rng.integers(30, 255, 3))
        cv2.ellipse(bg, tuple(int(v) for v in rng.integers(0, [w + 80, h + 80])),
                    tuple(int(v) for v in rng.integers(8, 40, 2)), float(rng.integers(0, 180)), 0, 360, c, -1)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    crank, R = np.array([640.0, 470.0]), 85.0
    hip = np.array([560.0, 200.0])
    shake = np.zeros(2)
    gt = []
    prev_layer_pos = None
    for i in range(n):
        t = i / fps
        shake = 0.9 * shake + rng.normal(0, 3.0, 2)
        off = np.clip(shake, -30, 30)
        a = 2 * math.pi * cadence * t
        img = bg[40 + int(off[1]):40 + int(off[1]) + h, 40 + int(off[0]):40 + int(off[0]) + w].copy()
        sh = -off + (off - off.astype(int))  # content moves opposite to camera
        feet, knees = [], []
        for k, ph in enumerate((0.0, math.pi)):
            foot = crank + R * np.array([math.cos(a + ph), math.sin(a + ph)])
            knee = np.array(ik(hip, foot, 175, 185))
            feet.append(foot)
            knees.append(knee)
        # far leg first (darker), then bike frame tube, then near leg
        layer = img.copy()
        for k in (1, 0):
            dim = 0.6 if k == 1 else 1.0
            foot, knee = feet[k], knees[k]
            skin = tuple(int(c * dim) for c in (120, 160, 210))
            cv2.line(layer, tuple(hip.astype(int)), tuple(knee.astype(int)), tuple(int(c * dim) for c in (40, 40, 40)), 46)
            cv2.line(layer, tuple(knee.astype(int)), tuple(foot.astype(int)), skin, 34)
            cv2.circle(layer, tuple(knee.astype(int)), 20, skin, -1)
            sock_top = knee + 0.55 * (foot - knee)
            cv2.line(layer, tuple(sock_top.astype(int)), tuple(foot.astype(int)), tuple(int(240 * dim) for _ in range(3)), 30)
            ang = 20 * math.sin(a + (0 if k == 0 else math.pi) + 0.6)
            cv2.ellipse(layer, tuple(foot.astype(int)), (34, 14), ang, 0, 360, tuple(int(235 * dim) for _ in range(3)), -1)
            cv2.ellipse(layer, tuple(foot.astype(int)), (34, 14), ang, 0, 360, (30, 30, 30), 2)
            cv2.line(layer, tuple((foot + [-12, 2]).astype(int)), tuple((foot + [12, 2]).astype(int)), (40, 40, 200), 3)
            if k == 1:
                cv2.line(layer, (480, 330), (760, 470), (20, 200, 230), 22)  # yellow down tube
                cv2.circle(layer, tuple(crank.astype(int)), 40, (30, 30, 30), -1)  # chainring
        # motion blur of the whole moving layer approximated by blending with previous pose
        if prev_layer_pos is not None:
            layer = cv2.addWeighted(layer, 0.65, prev_layer_pos, 0.35, 0)
        prev_layer_pos = layer
        M = np.float32([[1, 0, sh[0]], [0, 1, sh[1]]])
        img = cv2.warpAffine(layer, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        vw.write(img)
        gt.append((feet[0] + sh, knees[0] + sh))
    vw.release()
    return np.array(gt)


def simulate(path, gt, names=("shoe", "knee"), threshold=0.55, bridge=0, verbose=False, **kw):
    n = len(gt)
    p = Project()
    ts = []
    for k, nm in enumerate(names):
        t = p.add_track(nm)
        t.set(0, *gt[0, k])
        for key, v in kw.items():
            setattr(t, key, v)
        ts.append(t)
    clicks = 0
    f = 0
    t0 = time.time()
    while f < n - 1:
        def emit(fr, out):
            for tid, ff, x, y, c in out:
                tr = p.track(tid)
                if ff not in tr.pts or tr.pts[ff][3] != MANUAL:
                    tr.pts[ff] = (x, y, c, 0)
        active = [t for t in ts if t.at(f) is not None]
        r = run_tracking(path, active, f, 1, n, emit, lambda: False, threshold=threshold,
                         stop_on_loss=True, bridge=bridge)
        if not r.lost:
            break
        f = min(l[1] for l in r.lost)
        for tid, lf in r.lost:
            if lf == f:
                k = [t.id for t in ts].index(tid)
                p.track(tid).set(f, *gt[f, k], 1.0, MANUAL)
                clicks += 1
                if verbose:
                    print(f"  click {names[k]} @ {f}")
        if clicks > n:
            break
    res = {}
    for k, t in enumerate(ts):
        auto = [fr for fr, v in t.pts.items() if v[3] != MANUAL]
        err = np.array([np.hypot(*(np.array(t.pts[fr][:2]) - gt[fr, k])) for fr in auto]) if auto else np.array([0.0])
        res[t.name] = dict(silent=int((err > ERR).sum()), mean=float(np.median(err)), auto=len(auto))
    return clicks, res, time.time() - t0


if __name__ == "__main__":
    import itertools
    if "--sweep" in sys.argv:
        tot = {}
        for seed, cad in itertools.product((3, 4, 5), (1.6, 2.2)):
            path = f"tests/pedal_{seed}_{cad}.mp4"
            gt = make(path, seed=seed, cadence=cad)
            for thr in (0.55, 0.6, 0.65, 0.7, 0.75):
                clicks, res, dt = simulate(path, gt, threshold=thr)
                silent = sum(v["silent"] for v in res.values())
                a = tot.setdefault(thr, [0, 0])
                a[0] += clicks
                a[1] += silent
                print(f"seed {seed} cad {cad} thr {thr}: clicks {clicks:3d} silent {silent:3d}", flush=True)
        for thr, (c, sl) in tot.items():
            print(f"TOTAL thr {thr}: clicks {c}  silent {sl}")
        sys.exit()
    path = "tests/pedal.mp4"
    gt = make(path)
    np.save(path + ".gt.npy", gt)
    for thr in (0.55, 0.65, 0.75):
        clicks, res, dt = simulate(path, gt, threshold=thr, verbose="-v" in sys.argv)
        print(f"thr {thr}: clicks {clicks:3d}  " + "  ".join(
            f"{k}: silent {v['silent']:3d} / {v['auto']} auto, median err {v['mean']:.1f}px" for k, v in res.items())
            + f"   ({dt:.1f}s)")
