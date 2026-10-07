"""Drive the real window headlessly: add points, track (incl. a loss + fix), measure, calibrate, export."""
import os, sys, time
os.environ.setdefault("TRACKLAB_SETTINGS", "tracklab-tests")
import numpy as np
from PySide6.QtCore import Qt, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
app = QApplication(sys.argv[:1])
from tracklab.__main__ import apply_theme
apply_theme(app)
from tracklab.ui.main import MainWindow

def pump(ms=50):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents(QEventLoop.AllEvents, 20)

def wait_track(timeout=60):
    t0 = time.time()
    while w.tracking and time.time() - t0 < timeout:
        pump(20)
    assert not w.tracking, "tracking hung"

vid = sys.argv[1]
side = vid + ".tracklab.json"
if os.path.exists(side): os.remove(side)
w = MainWindow(); w.resize(1500, 950); w.show()
w.opts.update(threshold=0.55, bridge=0, stop_on_loss=True, retrack=True, snap=True)
w.open_path(vid); pump(200)
gt = np.load(vid + ".gt.npy")
w.set_tool("point")
for k in range(3):
    x, y = gt[0, k] + np.array([3.0, -2.5])   # sloppy clicks: snapping should fix them
    w.canvas_click(x, y, Qt.NoModifier)
for t, k in zip(w.project.tracks, range(3)):
    print("snap err", t.name, np.round(np.hypot(*(np.array(t.at(0)) - gt[0, k])), 2))
w.set_tool("select")
w.grab().save(f"{OUT}/1_points.png")
w.track_run(1); wait_track()
print("lost msg:", w._lost_msg, "frame", w.frame, "place_target", w.place_target)
w.grab().save(f"{OUT}/2_lost.png")
# Fix: click ground truth of the lost track -> auto-resumes
k = [t.id for t in w.project.tracks].index(w.place_target)
w.canvas_click(*gt[w.frame, k], Qt.NoModifier); pump(100); wait_track()
print("after fix: frame", w.frame, "msg", w._lost_msg)
for t in w.project.tracks:
    print(t.name, len(t.pts), "lost_at", t.lost_at)
# angle between the three points, distance, scale line
w.seek(40); pump()
w.set_tool("angle")
for t in w.project.tracks:
    w.canvas_click(*t.at(40), Qt.NoModifier)
w.set_tool("distance")
w.canvas_click(*w.project.tracks[0].at(40), Qt.NoModifier)
w.canvas_click(*w.project.tracks[2].at(40), Qt.NoModifier)
c = w.project.calib
c.mode, c.line, c.length, c.unit, c.origin = "line", [[100, 700], [500, 700]], 2.0, "m", [100, 700]
w.set_tool("select"); w.changed(); w.select_track(w.project.tracks[0].id); pump(300)
w.plots.qty.setCurrentIndex(2); pump(200)
w.grab().save(f"{OUT}/3_measures.png")
# zoom + loupe
w.canvas.zoom_at(3, w.canvas.to_screen(*w.project.tracks[0].at(40))); w.set_tool("point")
w.canvas.mouse_img = w.canvas.to_img(w.canvas.to_screen(*w.project.tracks[0].at(40))); w.canvas.loupe = True
w.canvas.repaint(); w.grab().save(f"{OUT}/4_zoom_loupe.png"); w.canvas.fit(); w.set_tool("select")
kin = w.kin()
sp = kin.tracks[w.project.tracks[0].id]["speed"]
print("circle speed m/s median", np.nanmedian(sp), "expected", 2*np.pi*1.5*200 * 2/400)
# undo/redo
n = len(w.project.measures); w.undo(); print("undo measures", n, "->", len(w.project.measures)); w.redo()
# exports (bypass dialogs)
from PySide6.QtWidgets import QFileDialog
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (f"{OUT}/out.csv", ""))
w.export_csv()
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (f"{OUT}/out.mp4", ""))
w.in_out = (0, 30); w.export_video(); w.in_out = None
print(open(f"{OUT}/out.csv").read().splitlines()[1][:300])
print("mp4 size", os.path.getsize(f"{OUT}/out.mp4"))
w.save_sidecar(); print("sidecar", os.path.exists(side))
# reopen restores
w2 = MainWindow(); w2.open_path(vid); print("restored tracks", len(w2.project.tracks), "measures", len(w2.project.measures))
