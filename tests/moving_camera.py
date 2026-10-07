"""Camera-motion compensation check: re-film the floor-plane example with a fake handheld camera
(wobble, pan, zoom, roll) and confirm world positions are still right."""
import math, sys
import cv2, numpy as np
sys.path.insert(0, ".")
from tracklab.model import Project
from tracklab.stabilize import compute, default_roi

src = Project.load("examples/floor_plane.tracklab.json")
cap = cv2.VideoCapture(src.video_path)
W, H = 1280, 720
rng = np.random.default_rng(5)
out = "tests/floor_moving.mp4"
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
cams = []
for i in range(300):
    ok, f = cap.read()
    t = i / 30
    z = 1.08 + 0.08 * math.sin(t * 0.9)            # zoom in/out
    rot = math.radians(2.0 * math.sin(t * 1.3))     # roll
    tx, ty = 40 * math.sin(t * 0.7) + rng.normal(0, 1.5), 25 * math.sin(t * 1.1 + 1) + rng.normal(0, 1.5)
    c, s = math.cos(rot) * z, math.sin(rot) * z
    M = np.array([[c, -s, (1 - c) * W / 2 + s * H / 2 + tx], [s, c, -s * W / 2 + (1 - c) * H / 2 + ty], [0, 0, 1]])
    vw.write(cv2.warpPerspective(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE))
    cams.append(M)
vw.release()
# true robot pixel positions in the moving video
r = src.by_name("Robot")
true_px = np.array([cv2.perspectiveTransform(np.float32([[r.pts[i][:2]]]), cams[i])[0, 0] for i in range(300)])
p = Project(video_path=out)
import copy
p.calib = copy.deepcopy(src.calib)
p.calib.quad = cv2.perspectiveTransform(np.float32([src.calib.quad]), cams[0])[0].tolist()  # drawn on frame 0
p.calib.frame = 0
truth = src.world(np.array([r.pts[i][:2] for i in range(300)]), np.arange(300))
still = p.world(true_px, np.arange(300))
print("without compensation: error mean %.1f cm, max %.1f cm" % (np.nanmean(np.hypot(*(still - truth).T)) * 100,
                                                              np.nanmax(np.hypot(*(still - truth).T)) * 100))
p.stab = compute(out, 0, 300, roi=default_roi(np.array(p.calib.quad), W, H))
w = p.world(true_px, np.arange(300))
e = np.hypot(*(w - truth).T)
print(f"with compensation:    {len(p.stab)}/300 frames, error mean {np.nanmean(e)*100:.1f} cm, max {np.nanmax(e)*100:.1f} cm")
if "-v" in sys.argv:
    pts = np.float32([[[200, 200], [1000, 600], [640, 360]]])
    for i in (1, 2, 5, 30, 100, 200, 299):
        Ht = cams[0] @ np.linalg.inv(cams[i])
        if i not in p.stab:
            print(i, "not registered"); continue
        He = np.array(p.stab[i]).reshape(3, 3)
        d = np.abs(cv2.perspectiveTransform(pts, Ht) - cv2.perspectiveTransform(pts, He)).max()
        print(i, "max corner disagreement px", round(float(d), 2))
