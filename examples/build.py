"""Builds the bundled example projects (videos + finished analyses + guided tours).

    uv run python examples/build.py

Synthetic scenes are rendered from exact physics so the numbers in the example are checkable
(the physics lab really measures g = 9.81 m/s²). Points are tracked with the real tracker; when it
stops, the builder answers with the true position — exactly what a user clicking would do.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
from tracklab.model import INTERP, MANUAL, Project  # noqa: E402
from tracklab.tracker import run_tracking  # noqa: E402


def track_with_oracle(path, p, names, gt, n, bridge=10, stop_when_gone=True):
    """Track `names` (already placed on frame 0) through n frames; on a stop, place the true point."""
    W, H = cv2.VideoCapture(path).get(3), cv2.VideoCapture(path).get(4)
    for name, k in names.items():
        t = p.by_name(name)
        f = 0
        while f < n - 1:
            def emit(fr, out, t=t):
                for _, ff, x, y, c in out:
                    if ff not in t.pts or t.pts[ff][3] != MANUAL:
                        t.pts[ff] = (x, y, c, 0)
            r = run_tracking(path, [t], f, 1, n, emit, lambda: False, bridge=bridge)
            if not r.lost:
                break
            lf = r.lost[0][1]
            x, y = gt[lf, k]
            if stop_when_gone and not (0 <= x < W and 0 <= y < H):
                t.lost_at = lf  # left the picture: leave it "lost" (shown in red on the timeline)
                break
            t.set(lf, x, y, 1.0, MANUAL)
            f = lf
        err = [math.dist(t.pts[f][:2], gt[f, k]) for f in t.pts]
        print(f"  {name}: {len(t.pts)} frames, mean err {np.mean(err):.2f}px, max {np.max(err):.2f}px")


# ============================================================================ physics lab
def physics_lab():
    W, H, FPS_FILE, FPS_CAP, S = 1280, 720, 30, 240, 300.0
    OX, OY = 100.0, 616.0  # origin (launch point on the floor), px

    def px(x, y):
        return OX + x * S, OY - y * S

    rng = np.random.default_rng(7)
    bg = np.full((H, W, 3), (196, 214, 226), np.uint8)
    bg = np.clip(bg + cv2.GaussianBlur(rng.normal(0, 14, (H, W, 3)), (0, 0), 2), 0, 255).astype(np.uint8)
    for x0, y0, x1, y1, c in ((640, 60, 820, 230, (150, 120, 90)), (180, 90, 300, 200, (90, 140, 170)),
                              (900, 420, 1010, 540, (120, 160, 120))):
        cv2.rectangle(bg, (x0, y0), (x1, y1), c, -1)
        cv2.rectangle(bg, (x0 + 10, y0 + 10), (x1 - 10, y1 - 10), tuple(min(255, v + 50) for v in c), 3)
    floor = np.clip(np.full((H - int(OY), W, 3), 70, np.float64) + rng.normal(0, 9, (H - int(OY), W, 3)), 0, 255)
    bg[int(OY):] = cv2.GaussianBlur(floor.astype(np.uint8), (0, 0), 1.2)
    cv2.line(bg, (0, int(OY)), (W, int(OY)), (40, 40, 40), 2)
    # metre stick lying on the floor edge, 10 cm stripes
    for i in range(10):
        a, b = int(px(i / 10, 0)[0]), int(px((i + 1) / 10, 0)[0])
        cv2.rectangle(bg, (a, int(OY) + 2), (b, int(OY) + 16), (30, 210, 240) if i % 2 == 0 else (25, 25, 25), -1)
    cv2.putText(bg, "1 m", (int(px(0.43, 0)[0]), int(OY) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    # occluding post
    post = (470, 170, 520, int(OY))

    n = 384  # 1.6 s of real time
    dt = 1 / FPS_CAP
    # ball: thrown from 0.30 m, bounces, rolls off the right edge
    x, y, vx, vy, r = 0.0, 0.30, 3.2, 4.0, 0.04
    ball = []
    for i in range(n):
        ball.append(px(x, y))
        for _ in range(40):
            h = dt / 40
            vy -= 9.81 * h
            x += vx * h
            y += vy * h
            if y < r and vy < 0:
                y, vy, vx = r, -0.55 * vy, 0.85 * vx
    # pendulum (exact, RK4)
    L, th, om = 0.6, 0.45, 0.0
    pivot = px(3.4, 1.58)
    bob = []
    for i in range(n):
        bob.append((pivot[0] + L * S * math.sin(th), pivot[1] + L * S * math.cos(th)))
        for _ in range(20):
            h = dt / 20
            f = lambda a: -9.81 / L * math.sin(a)  # noqa: E731
            k1t, k1o = om, f(th)
            k2t, k2o = om + h / 2 * k1o, f(th + h / 2 * k1t)
            k3t, k3o = om + h / 2 * k2o, f(th + h / 2 * k2t)
            k4t, k4o = om + h * k3o, f(th + h * k3t)
            th += h / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
            om += h / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)
    path = os.path.join(ROOT, "physics_lab.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS_FILE, (W, H))
    sh = 16
    P = lambda q: (int(round(q[0] * sh)), int(round(q[1] * sh)))  # noqa: E731
    for i in range(n):
        img = bg.copy()
        # pendulum
        cv2.line(img, P(pivot), P(bob[i]), (50, 50, 50), 2, cv2.LINE_AA, 4)
        cv2.rectangle(img, (int(pivot[0]) - 24, int(pivot[1]) - 14), (int(pivot[0]) + 24, int(pivot[1]) + 10), (90, 90, 95), -1)
        cv2.circle(img, P(pivot), 5 * sh, (235, 235, 235), -1, cv2.LINE_AA, 4)
        cv2.circle(img, P(bob[i]), 17 * sh, (30, 30, 200), -1, cv2.LINE_AA, 4)
        cv2.circle(img, P(bob[i]), 5 * sh, (240, 240, 240), -1, cv2.LINE_AA, 4)
        # ball, then the post in front of it
        cv2.circle(img, P(ball[i]), 12 * sh, (20, 120, 245), -1, cv2.LINE_AA, 4)
        cv2.circle(img, P(ball[i]), 12 * sh, (10, 60, 140), 2, cv2.LINE_AA, 4)
        cv2.rectangle(img, post[:2], post[2:], (60, 75, 90), -1)
        cv2.rectangle(img, (post[0] + 6, post[1]), (post[0] + 12, post[3]), (90, 105, 120), -1)
        cv2.putText(img, f"REC 240 fps  {i / FPS_CAP:5.3f} s", (W - 330, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 200), 2)
        img = np.clip(img + rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)
        vw.write(img)
    vw.release()
    gt = np.stack([np.array(ball), np.array(bob)], axis=1)

    p = Project(video_path=path, capture_fps=FPS_CAP, trail=60)
    p.title = "Physics lab — slow-mo throw & pendulum"
    p.blurb = "Measures gravity from a thrown ball, a pendulum's swing, occlusion bridging, slow-motion timing."
    tb = p.add_track("Ball")
    tb.color = "#d95926"
    tb.set(0, *ball[0])
    pv = p.add_track("Pivot", static=True)
    pv.color = "#9085e9"
    pv.set(0, *pivot)
    bb = p.add_track("Bob")
    bb.color = "#3987e5"
    bb.set(0, *bob[0])
    print("physics lab:")
    track_with_oracle(path, p, {"Ball": 0, "Bob": 1}, gt, n, bridge=25)
    c = p.calib
    c.mode, c.line, c.length, c.unit = "line", [[OX, OY + 9], [OX + S, OY + 9]], 1.0, "m"
    c.origin, c.axis_angle, c.y_up = [OX, OY], 0.0, True
    tilt = p.add_measure("segment", [pv.id, bb.id])
    tilt.name, tilt.color, tilt.signed = "Pendulum angle", "#ffffff", False
    dist = p.add_measure("distance", [pv.id, bb.id])
    dist.name, dist.color = "String length", "#199e70"
    hidden = [f for f in range(1, max(tb.pts)) if f not in tb.pts]
    gap = (hidden[0], hidden[-1]) if hidden else (100, 106)
    lost = tb.lost_at or max(tb.pts)
    p.tour = [
        dict(title="A finished analysis", seek=0, fit=True, spot=["canvas"], text=
             "This clip was filmed in <b>slow motion at 240 fps</b>: a ball is thrown and a pendulum swings. "
             "Everything is already tracked and measured. Press <b>Space</b> to play it (and again to pause). "
             "Use <b>Next</b> to walk through what's here."),
        dict(title="Slow-motion: real time", spot=["time"], text=
             "The file plays at 30 fps but was recorded at 240 fps. <b>Capture rate</b> (right panel, "
             "<i>Time &amp; filtering</i>) is set to 240 so seconds and speeds are real — otherwise every speed "
             "would be 8× too slow. Set this whenever you use a phone's slow-mo mode."),
        dict(title="Scale & origin", seek=0, zoom=[250, 590, 2.2], spot=["calib"], text=
             "The yellow line on the <b>metre stick</b> is the scale: 300 pixels = 1.00 m. The arrows are the axes: "
             "<b>(0, 0)</b> is where the ball was thrown from, <b>x</b> (red) is metres to the right, <b>y</b> "
             "(green) is metres <b>up</b>. So the ball's y is its height above the launch point. "
             "<br><b>Try it:</b> drag a yellow end of the line — every number updates. <b>Ctrl+Z</b> puts it back."),
        dict(title="The ball's path", seek=150, fit=True, select="Ball", plot={"q": "y", "show": ["Ball"]},
             spot=["plots", "canvas"], text=
             "The orange trail is the ball's path. The graph below shows its <b>height</b> over time — a parabola. "
             "<b>Hover</b> the graph to read values; <b>click</b> it to jump the video to that moment."),
        dict(title="Measuring gravity", select="Ball", plot={"q": "vy", "show": ["Ball"]}, seek=60,
             spot=["plots"], text=
             "This is the ball's <b>vertical speed</b>. While it flies it falls in a straight line — the slope is "
             "gravity. Hover two points on the line: it drops about <b>9.8 m/s every second</b>. The jumps are "
             "the bounces."),
        dict(title="Hidden behind the post", seek=gap[0] + (gap[1] - gap[0]) // 2,
             zoom=[*ball[gap[0] + (gap[1] - gap[0]) // 2], 2.2],
             select="Ball", spot=["timeline"], text=
             f"For frames {gap[0]}–{gap[1]} the ball is behind the post. The tracker coasted along the predicted "
             "path and picked it up again on the other side. Those frames are left <b>empty</b> (see the gap in the "
             "orange lane on the timeline) rather than made up. <i>Edit → Fill gaps</i> can interpolate them."),
        dict(title="Lost: left the picture", seek=lost, fit=True, select="Ball", spot=["timeline", "canvas"], text=
             f"At frame {lost} the ball rolls out of the picture, so the tracker stops — marked <b>red</b> on the "
             "timeline and \"last seen here\" on the video. If a point is lost while it's still visible, just click "
             "where it is (or press Enter to take the dashed guess) and tracking carries on."),
        dict(title="A fixed point", seek=0, select="Pivot", spot=["points"], text=
             "<b>Pivot</b> is a <i>fixed point</i> (square marker): it never moves, so it isn't tracked — see "
             "\"Fixed point\" ticked in the point settings. Use fixed points for anything stationary you want to "
             "measure against."),
        dict(title="Pendulum angle", select="Bob", plot={"q": "angle", "show": ["Pendulum angle"]}, seek=100,
             spot=["measures", "plots"], text=
             "<b>Pendulum angle</b> is a <i>tilt</i> measurement: the angle of the Pivot→Bob line against the "
             "x-axis, so hanging straight down is −90°. It swings between about −64° and −116° — a smooth wave on "
             "the graph. Measurements are listed under <i>Measurements</i>; untick one to hide it."),
        dict(title="String length", plot={"q": "distance", "show": ["String length"]}, spot=["readout", "plots"],
             text="<b>String length</b> is a <i>distance</i> measurement. It stays at 0.60 m the whole time — a good "
             "sanity check that the tracking and scale are right. Live values for the current frame are in "
             "<i>At this frame</i>."),
        dict(title="Smoothing", select="Ball", plot={"q": "accel", "show": ["Ball"], "raw": True}, seek=60,
             spot=["time", "plots"], text=
             "Speeds and accelerations come from differentiating positions, which amplifies jitter, so the app "
             "applies a zero-lag low-pass filter (<i>Smoothing: Auto</i>). Flying, the ball's acceleration reads "
             "≈ 9.8 m/s² — gravity again."),
        dict(title="Playback speed & loops", seek=0, fit=True, range=[0, 200], spot=["transport", "timeline"], text=
             "The bar under the video: ◀ / ▶| step one frame (or the ← → keys; Shift for 10), the 1× box changes "
             "playback speed, and <b>[</b> <b>]</b> set a range. A range is set now (frames 0–200, shaded blue on "
             "the timeline): playback loops inside it and tracking stops at its ends. <b>✕</b> (or \\) clears it."),
        dict(title="Get the numbers out", spot=["act:export"], text=
             "<b>Export CSV</b> (Ctrl+E) saves every frame's positions, speeds and measurements as a spreadsheet. "
             "<b>Export video</b> renders the clip with all these overlays. That's the tour — open your own video "
             "from <i>File → Open</i>, or try another example from <i>Help → Examples</i>."),
    ]
    p.save(os.path.join(ROOT, "physics_lab.tracklab.json"))


# ============================================================================ floor plane
def floor_plane():
    W, H, FPS, n = 1280, 720, 30, 300
    PPM = 200.0  # texture px per metre
    X0, X1, Y0, Y1 = -1.4, 4.4, -0.8, 3.6
    TW, TH = int((X1 - X0) * PPM), int((Y1 - Y0) * PPM)
    world = np.array([[0, 2], [3, 2], [3, 0], [0, 0]], np.float32)  # TL TR BR BL of the taped rectangle
    image = np.array([[452, 318], [858, 318], [1010, 610], [300, 610]], np.float32)
    Hw = cv2.getPerspectiveTransform(world, image)  # world (m) -> image px
    T = np.array([[1 / PPM, 0, X0], [0, -1 / PPM, Y1], [0, 0, 1]])  # texture px -> world
    M = Hw @ T

    def to_tex(x, y):
        return (x - X0) * PPM, (Y1 - y) * PPM

    def to_img(x, y):
        q = Hw @ np.array([x, y, 1.0])
        return q[0] / q[2], q[1] / q[2]

    rng = np.random.default_rng(11)
    tex = np.clip(np.full((TH, TW, 3), (150, 155, 160), np.float64) + cv2.GaussianBlur(rng.normal(0, 18, (TH, TW, 3)), (0, 0), 3), 0, 255).astype(np.uint8)
    for k in np.arange(math.ceil(X0 * 2) / 2, X1, 0.5):
        u = int(to_tex(k, 0)[0])
        cv2.line(tex, (u, 0), (u, TH), (125, 130, 135), 2)
    for k in np.arange(math.ceil(Y0 * 2) / 2, Y1, 0.5):
        v = int(to_tex(0, k)[1])
        cv2.line(tex, (0, v), (TW, v), (125, 130, 135), 2)
    corners = [to_tex(*c) for c in world]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        cv2.line(tex, tuple(int(v) for v in a), tuple(int(v) for v in b), (40, 200, 235), 10)
    dock = (-0.7, 1.0)
    du, dv = to_tex(*dock)
    cv2.rectangle(tex, (int(du - 40), int(dv - 40)), (int(du + 40), int(dv + 40)), (60, 170, 60), -1)
    cv2.circle(tex, (int(du), int(dv)), 12, (245, 245, 245), -1)
    # robot path: rounded loop around the tape at 0.8 m/s
    s = np.linspace(0, 2 * math.pi, 4000)
    a, b, e = 1.75, 1.25, 4
    loop = np.c_[1.5 + a * np.sign(np.cos(s)) * np.abs(np.cos(s)) ** (2 / e),
                 1.0 + b * np.sign(np.sin(s)) * np.abs(np.sin(s)) ** (2 / e)]
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(loop, axis=0).T))]
    speed = 0.8
    pos = [np.array([np.interp(speed * i / FPS, seg, loop[:, 0]), np.interp(speed * i / FPS, seg, loop[:, 1])]) for i in range(n)]
    sky = np.zeros((H, W, 3), np.uint8)
    for yy in range(H):
        sky[yy] = (95 + yy // 12, 88 + yy // 14, 80 + yy // 16)
    path = os.path.join(ROOT, "floor_plane.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    mask_tex = np.full((TH, TW), 255, np.uint8)
    mask = cv2.warpPerspective(mask_tex, M @ np.eye(3), (W, H), flags=cv2.INTER_NEAREST) > 0
    gt = []
    for i in range(n):
        t2 = tex.copy()
        x, y = pos[i]
        nx, ny = pos[min(i + 1, n - 1)] - pos[max(i - 1, 0)]
        hd = math.atan2(ny, nx)
        u, v = to_tex(x, y)
        cv2.circle(t2, (int(u), int(v)), int(0.16 * PPM), (150, 70, 30), -1, cv2.LINE_AA)
        tip = (int(u + math.cos(hd) * 0.13 * PPM), int(v - math.sin(hd) * 0.13 * PPM))
        cv2.line(t2, (int(u), int(v)), tip, (40, 40, 40), 6)
        cv2.circle(t2, (int(round(u)), int(round(v))), int(0.045 * PPM), (250, 250, 250), -1, cv2.LINE_AA)
        img = sky.copy()
        warped = cv2.warpPerspective(t2, M, (W, H), flags=cv2.INTER_LINEAR)
        img[mask] = warped[mask]
        img = np.clip(img + rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)
        vw.write(img)
        gt.append([to_img(x, y)])
    vw.release()
    gt = np.array(gt)

    p = Project(video_path=path, trail=90)
    p.title = "Floor plane — perspective calibration"
    p.blurb = "A camera at an angle: a 4-corner plane calibration turns pixels into true floor metres."
    r = p.add_track("Robot")
    r.color = "#3987e5"
    r.set(0, *gt[0, 0])
    d = p.add_track("Dock", static=True)
    d.color = "#199e70"
    d.set(0, *to_img(*dock))
    print("floor plane:")
    track_with_oracle(path, p, {"Robot": 0}, gt, n)
    c = p.calib
    c.mode, c.quad, c.width, c.height, c.unit = "plane", image.tolist(), 3.0, 2.0, "m"
    m = p.add_measure("distance", [d.id, r.id])
    m.name, m.color = "Distance to dock", "#e66767"
    far = int(np.argmin(gt[:, 0, 1]))
    near = int(np.argmax(gt[:, 0, 1]))
    p.tour = [
        dict(title="A camera at an angle", seek=0, fit=True, spot=["canvas"], text=
             "A small robot drives a loop on the floor at a steady <b>0.8 m/s</b>. The camera looks at the floor "
             "at an angle, so things further away look smaller and move fewer pixels. Press <b>Space</b> to play."),
        dict(title="Plane calibration", spot=["calib", "canvas"], text=
             "The yellow outline is a <b>plane calibration</b>: the 4 corners of the taped rectangle, which is "
             "3 m × 2 m in real life. The grid shows how the app now understands the floor in perspective. "
             "To make one: <i>Plane</i> tool, click the corners top-left → top-right → bottom-right → bottom-left, "
             "type the real size."),
        dict(title="Real speed, not pixel speed", select="Robot", plot={"q": "speed", "show": ["Robot"]},
             spot=["plots"], text=
             "The robot's <b>speed</b> reads a flat ≈ 0.8 m/s all the way round — even though on screen it crawls "
             "at the far side and races at the near side. That's the perspective correction at work."),
        dict(title="Far side vs near side", seek=far, select="Robot", spot=["readout"], text=
             f"Frame {far}: robot at the far side. Compare the <i>At this frame</i> readout here with the next step "
             "(near side) — the speed in m/s is the same."),
        dict(title="Near side", seek=near, select="Robot", spot=["readout"], text=
             f"Frame {near}: robot at the near side, moving many more pixels per frame — same real speed."),
        dict(title="What x and y mean here", plot={"q": "x", "show": ["Robot"]}, spot=["plots", "canvas"], text=
             "With a plane calibration, x and y are positions <b>on the floor</b>, in metres: <b>(0, 0)</b> is the "
             "taped rectangle's near-left corner, <b>x</b> runs along its 3 m side, <b>y</b> along its 2 m side "
             "(away from the camera). The robot loops between about x = −0.25…3.25 m and y = −0.25…2.25 m — "
             "and that's exactly the loop it was programmed to drive (checked: within 2.5 cm)."),
        dict(title="Distance to the dock", plot={"q": "distance", "show": ["Distance to dock"]},
             spot=["measures", "plots"], text=
             "<b>Dock</b> is a fixed point (green square). <i>Distance to dock</i> measures from it to the robot "
             "in real metres on every frame."),
        dict(title="Try it: move a corner", seek=0, fit=True, spot=["canvas"], text=
             "Drag one of the yellow corners somewhere else and watch the grid and the speed graph change — a bad "
             "calibration gives wrong numbers. <b>Ctrl+Z</b> undoes it."),
    ]
    p.save(os.path.join(ROOT, "floor_plane.tracklab.json"))


# ============================================================================ cycling (real footage)
def fill_short_gaps(p: Project, t, max_gap: int = 6) -> int:
    """Cubic interpolation over short holes, done in camera-steadied pixels; marked INTERP (orange)."""
    from scipy.interpolate import CubicSpline

    fr = np.array(sorted(t.pts))
    ref = p.to_ref(np.array([t.pts[f][:2] for f in fr]), fr)
    ok = ~np.isnan(ref).any(axis=1)
    fr, ref = fr[ok], ref[ok]
    cs = CubicSpline(fr, ref)
    n = 0
    for a, b in zip(fr, fr[1:]):
        if 1 < b - a <= max_gap + 1:
            for f in range(a + 1, b):
                q = p.from_ref(cs(f)[None], f)
                if q is not None:
                    t.pts[f] = (float(q[0, 0]), float(q[0, 1]), 0.5, INTERP)
                    n += 1
    return n


def cycling():
    """Whole clip. Leg tracked with the app's tracker (keyframes in cycling_keyframes.json, run with camera-motion
    compensation, every 6th frame checked by eye). The bike faces ~54° away from the camera, so a plain scale
    line would be wrong: calibration is a Wheel (circle) calibration on the front tyre's sidewall stripe."""
    src = os.path.expanduser("~/Videos/cycling/tdf2022_jumbo_visma_warmup.webm")
    dst = os.path.join(ROOT, "tdf2022_jumbo_visma_warmup.webm")
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    p = Project.load(os.path.join(ROOT, "cycling_tracked.json"))
    p.video_path = dst
    with open(os.path.join(ROOT, "cycling_stab.json")) as f:
        p.stab = {int(k): v for k, v in json.load(f).items()}
    p.title = "Cycling — full pedalling analysis (Tour de France warm-up)"
    p.blurb = ("Real handheld footage, whole 30 s clip: hip, knee and foot tracked; knee angle, crank angle, cadence, "
               "foot speed; wheel calibration corrects the bike being at an angle; camera motion compensated.")
    blocked = [(125, 137), (484, 521)]  # someone walks past / a team car drives through: view blocked
    for t in p.tracks:
        for a, b in blocked:
            for f in range(a, b + 1):
                t.pts.pop(f, None)
    cols = {"Hip": "#3987e5", "Knee": "#e66767", "Shoe": "#c98500"}
    filled = {}
    for t in p.tracks:
        t.color = cols[t.name]
        t.lost_at = None
        filled[t.name] = fill_short_gaps(p, t)
    # Wheel calibration from the front tyre's tan sidewall stripe (ellipse fitted to it on frame 0).
    cx, cy, W, H, ang = np.load(os.path.join(ROOT, "cycling_wheel_ellipse.npy"))
    a = math.radians(ang)
    u, v, c = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)]), np.array([cx, cy])
    cal = p.calib
    cal.mode, cal.unit, cal.frame = "circle", "m", 0
    cal.quad = [list(map(float, q)) for q in (c - v * H / 2, c + u * W / 2, c + v * H / 2, c - u * W / 2)]
    cal.diameter = 0.64  # sidewall stripe of a 700x25c road tyre (outer tyre ≈ 0.67 m; stripe sits ~1.5 cm inside)
    cal.line, cal.origin = [], None
    # Pedal-circle centre (≈ crank axle): circle fitted to the foot's path in the bike's plane.
    shoe = p.by_name("Shoe")
    fr = np.array([f for f, rec in shoe.pts.items() if rec[3] != INTERP])
    ref = p.to_ref(np.array([shoe.pts[f][:2] for f in fr]), fr)
    w = cal.to_world(ref[~np.isnan(ref).any(axis=1)])
    A = np.c_[2 * w[:, 0], 2 * w[:, 1], np.ones(len(w))]
    sol = np.linalg.lstsq(A, (w ** 2).sum(axis=1), rcond=None)[0]
    Hi = np.linalg.inv(cal.homography())
    q = Hi @ np.array([sol[0], sol[1], 1.0])
    crank = p.add_track("Crank", static=True)
    crank.color = "#9085e9"
    crank.set(0, float(q[0] / q[2]), float(q[1] / q[2]))
    ids = {t.name: t.id for t in p.tracks}
    p.measures = []
    m = p.add_measure("angle", [ids["Hip"], ids["Knee"], ids["Shoe"]])
    m.name, m.color = "Knee angle", "#ffffff"
    m = p.add_measure("segment", [ids["Crank"], ids["Shoe"]])
    m.name, m.color, m.visible = "Crank angle", "#9085e9", False
    m = p.add_measure("distance", [ids["Hip"], ids["Shoe"]])
    m.name, m.color, m.visible = "Hip–foot", "#d55181", False
    p.trail = 16
    from tracklab.kinematics import Kinematics

    kin = Kinematics(p, 722, 24.0)
    ka = kin.measures[p.by_name("Knee angle").id]["value"]
    rate = kin.measures[p.by_name("Crank angle").id]["rate"]
    rpm = abs(float(np.nanmedian(rate))) / 6
    spd = float(np.nanmedian(kin.tracks[ids["Shoe"]]["speed"]))
    fmin, fmax = int(np.nanargmin(ka[150:260])) + 150, int(np.nanargmax(ka[150:260])) + 150
    cov = {t.name: len(t.pts) for t in p.tracks if not t.static}
    print("cycling: coverage", cov, "filled", filled, f"knee {np.nanmin(ka):.0f}-{np.nanmax(ka):.0f}°, "
          f"cadence {rpm:.0f} rpm, foot speed {spd:.2f} m/s")
    interp = next((f for f, rec in shoe.pts.items() if rec[3] == INTERP and 150 < f < 300), 160)
    p.tour = [
        dict(title="A finished analysis", seek=150, fit=True, spot=["canvas"], text=
             "Real handheld footage from the Tour de France: the yellow-jersey rider's leg is tracked for the "
             "<b>whole 30-second clip</b> — <b>Hip</b>, <b>Knee</b> and the <b>Shoe</b> (on its dials). "
             "Press <b>Space</b> to play. Use <b>Next</b> to walk through it."),
        dict(title="The timeline: what's measured where", seek=130, select="Shoe", spot=["timeline"], text=
             "One lane per point. <b>Own colour</b> = tracked confidently. <b>Orange</b> = less certain: blurry "
             "frames where the match was weaker, plus a few per stroke filled in by interpolation. <b>Empty</b> = not measured: around 5 s someone walks in front of the "
             "camera and around 20 s a team car drives through — no point guessing there, so the graphs have gaps "
             "too. <b>White ticks</b> = keyframes placed by hand to restart tracking after those."),
        dict(title="Why there's an ellipse on the wheel", seek=0, zoom=[1026, 742, 2.2], spot=["calib", "canvas"],
             text="The bike isn't side-on: it's turned about <b>54°</b> away from the camera (the round wheel looks "
             "like a tall, thin ellipse). A simple scale line would make every sideways distance ~40% too short "
             "and the angles wrong. So this uses a <b>Wheel calibration</b>: the 4 yellow dots sit on the tyre's tan "
             "stripe (0.64 m across). From the ellipse the app works out the bike's angle and measures <i>in the "
             "bike's plane</i>, as if filmed side-on."),
        dict(title="What x and y mean here", seek=0, zoom=[1026, 742, 1.4], spot=["canvas", "readout"], text=
             "Positions are in metres in the bike's plane. <b>(0, 0)</b> is where the front wheel touches the "
             "ground. <b>y</b> (green arrow) is height above the ground. <b>x</b> (red arrow) runs along the bike, "
             "<b>towards the back wheel</b> (this rider faces left). So Shoe y ≈ 0.25–0.6 m is how high the foot "
             "is."),
        dict(title="The camera moves — the app follows it", seek=300, fit=True, spot=["calib", "canvas"], text=
             "This was filmed handheld, so the picture shifts and zooms. <b>Camera moves — follow it</b> (ticked "
             "in Calibration) measures that on every frame and cancels it out. Watch the yellow wheel outline: it "
             "stays glued to the wheel while the camera moves. Without this, the camera's own movement would be "
             "counted as the rider moving."),
        dict(title="Knee angle", seek=fmin, select="Knee", plot={"q": "angle", "show": ["Knee angle"]},
             spot=["plots", "canvas"], text=
             f"The white arc: the angle at the knee between hip and foot, in the bike's plane. It swings between "
             f"about <b>{np.nanpercentile(ka, 5):.0f}°</b> (most bent, top of the stroke) and "
             f"<b>{np.nanpercentile(ka, 95):.0f}°</b> (most straight, bottom) every pedal stroke. This frame is the "
             "most bent. Hover the graph to read values; click it to jump the video there."),
        dict(title="Most extended", seek=fmax, select="Knee", plot={"q": "angle", "show": ["Knee angle"]},
             spot=["readout"], text=
             "Leg at its straightest. All current values are under <i>At this frame</i>. (It's measured to the "
             "shoe's dials rather than the ankle bone — the white sock can't be tracked reliably — so treat it as "
             "a few degrees approximate.)"),
        dict(title="Cadence from the crank angle", plot={"q": "angle_rate", "show": ["Crank angle"]},
             spot=["plots", "measures"], text=
             f"<b>Crank</b> is a fixed point at the centre of the foot's circle (≈ the crank axle). <i>Crank angle</i> "
             f"is the tilt of Crank→Shoe; the graph shows how fast it turns: about <b>{abs(np.nanmedian(rate)):.0f}°/s"
             f"</b>, i.e. <b>{rpm:.0f} rpm</b> — a typical warm-up cadence. (360°/s = 60 rpm.)"),
        dict(title="Foot speed", select="Shoe", plot={"q": "speed", "show": ["Shoe"]}, spot=["plots"], text=
             f"Because it's calibrated, speeds are real: the foot moves at about <b>{spd:.1f} m/s</b> around its "
             "circle. Switch the dropdown to Position Y to see the foot going up and down."),
        dict(title="Filled-in frames", seek=interp, zoom=[*shoe.pts[interp][:2], 2.0], select="Shoe",
             spot=["timeline", "canvas"], text=
             "At the back of each stroke the foot is too blurred to match, so a few frames per stroke were filled "
             "by interpolation (<i>Edit → Fill gaps</i>) — shown orange on the timeline and as a small dot instead "
             "of a ring on the video. Long gaps (blocked view) are left empty on purpose."),
        dict(title="How the tracker is set up", seek=200, fit=True, select="Knee", spot=["points", "tracking"], text=
             "With a point selected you see two boxes: the <b>solid box</b> is the patch it looks for each frame, the "
             "<b>dotted box</b> is how far it searches. <i>Tracking</i> settings: <b>Min. match</b> (how sure it must "
             "be before it stops to ask), <b>Bridge gaps</b> (coast through short hidden spells), snapping clicks onto "
             "markers, and re-tracking after you fix something."),
        dict(title="Right-click for more", seek=200, select="Knee", spot=["canvas"], text=
             "<b>Right-click a point</b> for: delete it on this frame, clear everything after here, fill gaps, or "
             "track only this point. Right-click empty space to add a point. Right-drag pans; the wheel zooms; "
             "<b>F</b> fits."),
        dict(title="Try it: track something new", seek=150, fit=True, tool="point", spot=["tool:point", "act:track"],
             text="The <b>+ Point</b> tool is on. Click the green-jersey rider's white shoe (left of the yellow rider), "
             "then press <b>Track ▶</b> (T). If it stops and asks, press <b>Enter</b> to accept its guess or click the "
             "right spot. Esc returns to Select. Ctrl+Z undoes anything."),
    ]
    p.save(os.path.join(ROOT, "cycling.tracklab.json"))


# ============================================================================ crash test (real footage)
def crash_test():
    """NHTSA frontal crash test of a 2006 Honda Ridgeline (public domain), the side-on high-speed view.
    Known answers to check against: test speed 35 mph = 56.3 km/h, wheelbase 3.10 m (122.0 in)."""
    path = os.path.join(ROOT, "crash_test_side.mp4")
    if not os.path.exists(path):
        raise SystemExit("crash_test_side.mp4 missing: cut frames 1448-1634 of NHTSA_Crash_Test_of_the_2006_Honda_"
                         "Ridgeline.ogv (Wikimedia Commons) — see README")
    with open(os.path.join(ROOT, "crash_keyframes.json")) as f:
        seeds = json.load(f)
    n = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
    p = Project(video_path=path, trail=40)
    p.title = "Crash test — speed, deceleration, crush (NHTSA)"
    p.blurb = "Real high-speed footage of a 56 km/h barrier crash: the app measures the impact speed, g-forces and crush."
    cols = {"Placard": "#3987e5", "Rear wheel": "#199e70", "Front wheel": "#e66767"}
    for name, kf in seeds.items():
        t = p.add_track(name)
        t.color = cols[name]
        t.template_r = 46 if "wheel" in name else 30  # whole wheel: the spokes spin, the outline doesn't
        for f, xy in kf.items():
            t.set(int(f), *xy, 1.0, MANUAL)

        def emit(fr, out, t=t):
            for _, ff, x, y, c in out:
                if ff not in t.pts or t.pts[ff][3] != MANUAL:
                    t.pts[ff] = (x, y, c, 0)
        run_tracking(path, [t], 0, 1, n, emit, lambda: False, threshold=0.6, bridge=5)
    rw, fw, pl = p.by_name("Rear wheel"), p.by_name("Front wheel"), p.by_name("Placard")
    c = p.calib
    c.mode, c.unit, c.length = "line", "m", 3.10
    c.line = [list(rw.pts[0][:2]), list(fw.pts[0][:2])]
    c.origin = list(rw.pts[0][:2])
    c.note = "wheelbase of a 2006 Ridgeline = 3.10 m"
    # The film shows the crash camera's own clock: 186 video frames span -0.0498 s ... +0.4946 s.
    p.capture_fps = 186 / (0.4946 + 0.0498)
    p.filter_hz = 30.0  # see the "Deceleration" tour step: peaks depend on this
    m = p.add_measure("distance", [rw.id, fw.id])
    m.name, m.color = "Wheelbase", "#ffffff"
    m = p.add_measure("segment", [rw.id, fw.id])
    m.name, m.color, m.visible = "Pitch", "#c98500", False
    from tracklab.kinematics import Kinematics

    kin = Kinematics(p, n, 29.97)
    d = kin.tracks[pl.id]
    tc = kin.t - 0.0498  # crash clock: 0 = first contact
    pre = float(np.nanmedian(d["vx"][(tc > -0.045) & (tc < -0.005)]))
    g = np.abs(d["ax"]) / 9.81
    ipk = int(np.nanargmax(g))
    crush = float(np.nanmax(d["x"][tc > 0]) - d["x"][tc > 0][0])
    istop = int(np.nanargmax(d["x"]))
    t_stop = float(tc[istop])
    avg_g = pre / t_stop / 9.81
    reb = float(np.nanmedian(d["vx"][istop + 10:]))
    wb = kin.measures[p.by_name("Wheelbase").id]["value"]
    i0 = int(np.argmin(np.abs(tc)))
    print(f"crash: impact speed {pre:.2f} m/s = {pre * 3.6:.1f} km/h (test: 56.3), peak {g[ipk]:.0f} g at "
          f"{tc[ipk] * 1000:.0f} ms, crush {crush:.2f} m, rebound {abs(reb):.1f} m/s, wheelbase {wb[0]:.2f}->{np.nanmin(wb):.2f} m")
    p.tour = [
        dict(title="A real crash test", seek=0, fit=True, spot=["canvas"], text=
             "Official NHTSA footage: a pickup driven into a solid barrier at <b>35 mph (56 km/h)</b>, filmed "
             "side-on with a high-speed camera. Three things are tracked: the white <b>placard</b> on the door "
             "(the body), and both <b>wheels</b>. Press <b>Space</b> to play — it's half a second of real time."),
        dict(title="Real time from the film's own clock", spot=["time"], text=
             "The video plays at 30 fps, but each frame is 2.9 ms of real time (the clock printed under the "
             "picture runs from −0.050 s to +0.495 s over 186 frames). So <b>Capture rate</b> is set to "
             "<b>341.7 fps</b>. Get this wrong and every speed is wrong by the same factor."),
        dict(title="Scale from the wheelbase", seek=0, fit=True, spot=["calib", "canvas"], text=
             "The yellow line joins the two wheel centres. A 2006 Ridgeline's wheelbase is <b>3.10 m</b>, which "
             "gives the scale (139 px = 1 m). (0, 0) is the rear wheel centre; <b>x</b> points forward (toward "
             "the barrier), <b>y</b> up."),
        dict(title="Check: impact speed", seek=8, select="Placard", plot={"q": "vx", "show": ["Placard"]},
             spot=["plots", "readout"], text=
             f"Forward speed of the body. Before it hits, the app measures <b>{pre:.1f} m/s = {pre * 3.6:.1f} km/h</b> "
             "— the test is run at 56.3 km/h, so the tracking, the scale and the timing all check out. Then it "
             "drops to zero in about 70 ms and goes slightly negative: the truck <b>bounces back</b> "
             f"at about {abs(reb):.1f} m/s."),
        dict(title="Deceleration in g", seek=ipk, select="Placard", plot={"q": "ax", "show": ["Placard"]},
             spot=["plots", "time"], text=
             f"Acceleration along x in m/s² (÷ 9.81 = g). The truck goes from {pre:.1f} m/s to a stop in "
             f"<b>{t_stop * 1000:.0f} ms</b> — an <b>average of {avg_g:.0f} g</b>; that number is solid. The peak "
             f"here is about <b>{g[ipk]:.0f} g</b> at {tc[ipk] * 1000:.0f} ms, but peaks are fragile: acceleration "
             "comes from differentiating position twice, so it depends on <i>Smoothing</i> (set to 30 Hz here; try "
             "15 or 60 in <i>Time &amp; filtering</i> — the peak moves between ~25 and ~45 g). Crash labs use "
             "accelerometers for exact peaks."),
        dict(title="Crush distance", seek=istop, select="Placard", plot={"q": "x", "show": ["Placard"]},
             spot=["plots", "canvas"], text=
             f"Position of the body. After touching the barrier it keeps going another <b>{crush:.2f} m</b> — "
             "that's how far the front end crushed. The trail shows the path; this frame is maximum crush."),
        dict(title="The front wheel is pushed back", seek=istop, plot={"q": "distance", "show": ["Wheelbase"]},
             spot=["measures", "plots"], text=
             f"<b>Wheelbase</b> = distance between the two wheel centres. It starts at 3.10 m and shrinks to "
             f"<b>{np.nanmin(wb):.2f} m</b>: the crash shoves the front wheel ~{(3.10 - np.nanmin(wb)) * 100:.0f} cm "
             "backwards. <i>Pitch</i> (hidden; tick it) shows the nose diving."),
        dict(title="Whole wheels, not spokes", seek=20, zoom=[*fw.pts[20][:2], 3.0], select="Front wheel",
             spot=["points", "canvas"], text=
             "The wheels spin fast, so a small patch on the spokes would lose track. The wheel points use a big "
             "template (solid box) covering the whole wheel — its round outline doesn't change as it spins. Set "
             "this per point with <i>Template</i> in the point's settings."),
        dict(title="Get the numbers out", spot=["act:export"], text=
             "<b>Export CSV</b> (Ctrl+E) writes time, positions, velocities and accelerations for every frame. "
             "<b>Export video</b> renders this with the overlays."),
    ]
    p.save(os.path.join(ROOT, "crash_test.tracklab.json"))


# ============================================================================ paper bag (real footage)
def paper_bag():
    """'Ballistische Kurve' by Rhetos (CC BY-SA 4.0): a crumpled paper bag thrown in front of a chalk grid,
    filmed at 200 fps. Frames 1532-1739 of the original (the un-annotated second throw)."""
    path = os.path.join(ROOT, "paper_bag_throw.mp4")
    if not os.path.exists(path):
        raise SystemExit("paper_bag_throw.mp4 missing — cut frames 1532-1739 of Ballistische_Kurve.webm (see README)")
    n = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
    p = Project(video_path=path, capture_fps=200.0, trail=400)
    p.title = "Paper bag throw — air drag (blackboard grid)"
    p.blurb = "Real 200 fps footage: a tumbling paper bag, tracked by colour; shows how air drag bends the path."
    t = p.add_track("Bag")
    t.color, t.method, t.template_r = "#d95926", "blob", 20
    with open(os.path.join(ROOT, "paper_bag_keyframes.json")) as f:
        for fr, xy in json.load(f).items():
            t.set(int(fr), *xy, 1.0, MANUAL)
    k0 = min(t.pts)

    def emit(fr, out):
        for _, ff, x, y, c in out:
            if ff not in t.pts or t.pts[ff][3] != MANUAL:
                t.pts[ff] = (x, y, c, 0)
    r = run_tracking(path, [t], k0, 1, n, emit, lambda: False, threshold=0.55, bridge=12)
    t.lost_at = r.lost[0][1] if r.lost else None
    for f in [f for f, v in t.pts.items() if v[3] == MANUAL or f > 191]:
        t.pts.pop(f)  # keep only free flight: not the bag in the hand, not the landing on the chalk tray
    t.lost_at = None
    p.filter_hz = 6.0  # the bag's real motion is slow; this keeps tumbling jitter out of the accelerations
    # Scale: 31 chalk squares between the outermost vertical grid lines (found from the image, sub-pixel),
    # 5 cm each. Cross-checked against gravity at the top of the flight (see tour).
    c = p.calib
    c.mode, c.unit, c.length = "line", "m", 1.55
    c.line = [[409.0, 862.0], [1559.3, 862.0]]
    first = min(t.pts)
    c.origin = [t.pts[first][0], t.pts[first][1]]
    c.note = "31 grid squares × 5 cm"
    from tracklab.kinematics import Kinematics

    kin = Kinematics(p, n, 30.0)
    d = kin.tracks[t.id]
    fr = np.array(sorted(t.pts))
    y = d["y"][fr]
    ia = int(fr[np.nanargmax(y)])
    tt = fr / 200.0
    wy = d["y_raw"][fr]
    ests = []
    for half in (0.06, 0.08, 0.1, 0.12, 0.14):
        for c0 in (ia - 4, ia - 2, ia, ia + 2, ia + 4):
            m = np.abs(tt - c0 / 200.0) <= half
            ests.append(-2 * np.polyfit(tt[m], wy[m], 2)[0])
    g_apex, g_lo, g_hi = np.median(ests), np.percentile(ests, 16), np.percentile(ests, 84)
    rise = (tt < ia / 200.0 - 0.05)
    fall = (tt > ia / 200.0 + 0.05)
    a_rise = -2 * np.polyfit(tt[rise], wy[rise], 2)[0]
    a_fall = -2 * np.polyfit(tt[fall], wy[fall], 2)[0]
    v0 = float(d["speed"][fr[2]])
    h = float(np.nanmax(y) - y[0])
    last = int(fr[-1])
    vfall = float(d["speed"][last - 3])
    print(f"paper bag: {len(t.pts)} frames, apex frame {ia}, g at apex {g_apex:.2f} ({g_lo:.1f}-{g_hi:.1f}) m/s², rising {a_rise:.1f}, "
          f"falling {a_fall:.1f}, launch {v0:.1f} m/s, rise {h:.2f} m, final {vfall:.1f} m/s")
    p.tour = [
        dict(title="A tumbling paper bag", seek=12, fit=True, spot=["canvas"], text=
             "A crumpled paper bag thrown across a blackboard, filmed at <b>200 frames per second</b> (from the "
             "Mathe-AC maths workshop). The whole path is already tracked — the orange line. Press <b>Space</b> to "
             "play it slowly."),
        dict(title="Tracking something that tumbles", seek=60, zoom=[*t.pts[60][:2], 3.0], select="Bag",
             spot=["points", "canvas"], text=
             "The bag spins and changes shape every frame, so matching a picture of it (the normal tracker) fails. "
             "This point uses the <b>Blob</b> tracker (point settings → <i>Tracker</i>): it follows a patch of "
             "<b>colour</b> — the pale bag on the dark board — and takes its centre. Great for balls, coloured "
             "markers, anything that rotates."),
        dict(title="Scale from the chalk grid", seek=0, fit=True, spot=["calib", "canvas"], text=
             "The yellow line spans <b>31 chalk squares</b> of 5 cm = 1.55 m. (0, 0) is where the bag leaves the "
             "hand; <b>x</b> points right, <b>y</b> up. So <i>Position Y</i> is height above the release point."),
        dict(title="The path isn't a parabola", seek=last, select="Bag", plot={"q": "y", "show": ["Bag"]},
             spot=["plots", "canvas"], text=
             "Without air the path would be a symmetric arch. Here the way down is <b>much steeper</b> than the way "
             f"up: air drag eats the sideways speed. It rises {h:.2f} m in {ia / 200:.2f} s but takes "
             f"{(last - ia) / 200:.2f} s to come back down."),
        dict(title="Vertical speed shows the drag", select="Bag", plot={"q": "vy", "show": ["Bag"]}, seek=ia,
             spot=["plots"], text=
             f"Vertical speed over time. Going up it drops fast (gravity <i>and</i> drag both pull down: about "
             f"<b>{a_rise:.0f} m/s²</b>). Coming down it gains speed slowly (drag now pushes up: only about "
             f"<b>{a_fall:.0f} m/s²</b>). Without air both would be 9.81 m/s²."),
        dict(title="Check: gravity at the top", seek=ia, select="Bag", plot={"q": "ay", "show": ["Bag"]},
             spot=["plots", "readout"], text=
             f"At the very top the bag moves sideways only, so drag can't push it up or down — only gravity acts "
             f"vertically. Fitting the path there gives <b>{g_apex:.1f} m/s²</b> (between {g_lo:.1f} and {g_hi:.1f} "
             "depending on exactly which frames you fit), around the real 9.81: an independent check that the grid "
             "scale is right to within roughly 10%. The tumbling bag's centre wobbles, so this is as precise as it "
             "gets; Smoothing is set to 6 Hz to keep that wobble out of the graphs."),
        dict(title="Speed", select="Bag", plot={"q": "speed", "show": ["Bag"]}, spot=["plots"], text=
             f"It leaves the hand at about <b>{v0:.1f} m/s</b>, slows to its minimum just after the top, and falls "
             f"at about {vfall:.1f} m/s at the end — heading for a terminal speed where drag balances its weight."),
        dict(title="Scale from gravity (no ruler needed)", seek=ia, spot=["calib"], text=
             "No ruler in your shot? Select a point that's flying, put the playhead in the flight (or mark it with "
             "<b>[</b> <b>]</b>) and click <b>Scale from gravity</b>: the app fits the fall and uses 9.81 m/s² as the "
             "ruler. Only accurate when air drag is small (a ball, not a paper bag!)."),
    ]
    p.save(os.path.join(ROOT, "paper_bag.tracklab.json"))


if __name__ == "__main__":
    which = sys.argv[1:] or ["physics", "floor", "cycling", "crash", "bag"]
    if "physics" in which:
        physics_lab()
    if "floor" in which:
        floor_plane()
    if "cycling" in which:
        cycling()
    if "crash" in which:
        crash_test()
    if "bag" in which:
        paper_bag()
