"""'Scale from gravity' on the synthetic physics lab: the true scale is 300 px/m with gravity straight down."""
import os, sys
os.environ.setdefault("TRACKLAB_SETTINGS", "tracklab-tests")
from PySide6.QtWidgets import QApplication
app = QApplication([])
sys.path.insert(0, ".")
from tracklab.ui.main import MainWindow
import math
w = MainWindow()
w.open_path("examples/physics_lab.tracklab.json")
w.sidebar.tour.stop()
w.select_track(w.project.by_name("Ball").id)
w.in_out = (5, 85)
w.calibrate_from_gravity()
c = w.project.calib
ppm = math.dist(*c.line)
print(f"scale {ppm:.1f} px/m (true 300.0), axis tilt {math.degrees(c.axis_angle):.2f}° (true 0), note: {c.note}")
assert abs(ppm - 300) < 6 and abs(math.degrees(c.axis_angle)) < 1
