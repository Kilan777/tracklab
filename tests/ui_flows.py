"""Correction flows: drag-to-fix re-tracks to the next keyframe; Esc on a lost point skips it and continues."""
import os, sys, time
os.environ.setdefault("TRACKLAB_SETTINGS", "tracklab-tests")
import numpy as np
from PySide6.QtCore import Qt, QEventLoop
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv[:1])
from tracklab.ui.main import MainWindow

def pump(ms=50):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents(QEventLoop.AllEvents, 20)
def wait():
    t0 = time.time()
    while w.tracking and time.time() - t0 < 60: pump(20)
    pump(50)

vid = "tests/synth.mp4"
if os.path.exists(vid + ".tracklab.json"): os.remove(vid + ".tracklab.json")
gt = np.load(vid + ".gt.npy")
w = MainWindow(); w.show(); w.open_path(vid)
w.opts.update(bridge=0, stop_on_loss=True, retrack=True)
w.set_tool("point")
for k in range(3): w.canvas_click(*gt[0, k], Qt.NoModifier)
w.set_tool("select")
w.track_run(1); wait()
print("1) lost at", w.frame, w._lost_msg[:30])
w.escape(); wait()
print("2) after Esc-skip: frame", w.frame, "lost:", [(t.name, t.lost_at) for t in w.project.tracks])
# corrupt P1 at frame 50 by dragging it 20px away -> retrack forward should follow the wrong spot or get lost; then drag back
t1 = w.project.tracks[0]
w.seek(50); pump()
before = len(t1.pts)
w.push_undo(); w.drag_to(("track", t1.id), *(gt[50, 0] + [0.3, -0.2]), Qt.NoModifier); w.drag_done(("track", t1.id), True); wait()
err = [np.hypot(*(np.array(t1.pts[f][:2]) - gt[f, 0])) for f in range(51, 150) if f in t1.pts]
print("3) drag-fix at 50 re-tracked", len(err), "frames, mean err %.2f" % np.mean(err), "keyframe kind", t1.pts[50][3])
# backward tracking from a mid keyframe on a fresh point
w.seek(120); pump(); w.set_tool("point"); w.canvas_click(*gt[120, 2], Qt.NoModifier); w.set_tool("select")
w.track_run(-1, ids=[w.selected]); wait()
t4 = w.project.track(w.selected)
err = [np.hypot(*(np.array(t4.pts[f][:2]) - gt[f, 2])) for f in t4.frames()]
print("4) backward: frames", min(t4.frames()), "-", max(t4.frames()), "mean err %.2f" % np.mean(err))
print("undo depth", len(w._undo))
