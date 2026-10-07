import sys, time, numpy as np
sys.path.insert(0, ".")
from tracklab.model import Project
from tracklab.tracker import run_tracking
from tracklab.kinematics import Kinematics
path = sys.argv[1]
gt = np.load(path + ".gt.npy")
p = Project()
ts = []
for k, name in enumerate(["circle", "proj", "occl"]):
    t = p.add_track(name); t.set(0, *gt[0, k]); ts.append(t)
got = {}
def emit(f, pts):
    for tid, fr, x, y, c in pts: p.track(tid).set(fr, x, y, c, 0)
t0 = time.time()
r = run_tracking(path, ts, 0, 1, len(gt), emit, lambda: False, stop_on_loss=False, bridge=int(sys.argv[2]) if len(sys.argv)>2 else 10)
print("run", r, f"{time.time()-t0:.2f}s")
for k, t in enumerate(ts):
    fr = [f for f in t.frames() if f > 0]
    err = np.array([np.hypot(*(np.array(t.pts[f][:2]) - gt[f, k])) for f in fr])
    print(t.name, "frames", len(fr), "last", max(fr) if fr else None, "mean err %.3f max %.3f" % (err.mean(), err.max()), "lost_at", [l for l in r.lost if l[0]==t.id])
kin = Kinematics(p, len(gt), 60)
print("cutoffs", kin.cutoff_used)
sp = kin.tracks[ts[0].id]["speed"]
print("circle speed px/s median", np.nanmedian(sp), "expected", 2*np.pi*1.5*200)
print("occl gap", [f for f in range(140,175) if f not in ts[2].pts]); print("circle gap", [f for f in range(140,200) if f not in ts[0].pts])
