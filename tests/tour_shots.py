"""Walk every example's tour in the real window and screenshot each step (offscreen)."""
import os, sys, time
os.environ.setdefault("TRACKLAB_SETTINGS", "tracklab-tests")
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication
OUT = sys.argv[1]
only = sys.argv[2:] 
os.makedirs(OUT, exist_ok=True)
app = QApplication([])
from tracklab.__main__ import apply_theme
apply_theme(app)
from tracklab.ui.main import MainWindow
from tracklab.ui.welcome import examples
def pump(ms=100):
    end = time.time() + ms / 1000
    while time.time() < end: app.processEvents(QEventLoop.AllEvents, 20)
w = MainWindow(); w.resize(1500, 950); w.show(); pump(300)
w.grab().save(f"{OUT}/t0_welcome.png")
for path, title, _ in examples():
    key = os.path.basename(path).split(".")[0]
    if only and key not in only: continue
    w.open_path(path); pump(500)
    tour = w.sidebar.tour
    assert tour.isVisible(), "tour panel not shown"
    for i in range(len(tour.steps)):
        tour.go(i); pump(350)
        w.grab().save(f"{OUT}/t_{key}_{i:02d}.png")
    print(key, "steps", len(tour.steps), "ok")
