# TrackLab

Video motion tracking & measurement — what Kinovea does, minus the fighting with it.

```
tracklab [video.mp4]        # or launch "TrackLab" from the app menu
```

It opens on a start screen with **five finished examples**, each with a guided tour (Back / Show me / Next) that
jumps to the right frame, sets up the graph and spotlights the control it's talking about. Three are real footage,
and each example is checked against a known answer:

| Example | What it shows | Checked against |
|---|---|---|
| Cycling (Tour de France warm-up, handheld, 30 s) | whole-clip tracking, camera-motion compensation, **wheel calibration** for a bike filmed at 54°, knee angle, crank angle, cadence | plausible geometry (saddle ~1.0 m, cadence 87 rpm); gaps where the view is blocked |
| Crash test (NHTSA, public domain) | slow-mo clock → capture rate, scale from the wheelbase, speed, g-forces, crush | **impact speed 56.4 km/h vs. the official 56.3** |
| Paper bag throw (Rhetos, CC BY-SA 4.0, 200 fps) | **Blob** (colour) tracking of a tumbling object, grid scale, air drag | gravity at the apex 9.1 m/s² (8.1–10.0) vs 9.81 |
| Physics lab (simulated) | slow-mo, metre stick, gravity, occlusion, lost point, fixed point, smoothing | g = 9.81 m/s², string 0.600 m (exact) |
| Floor plane (simulated) | 4-corner perspective plane, speed in m/s, position on the floor | 0.80 m/s, path within 2.5 cm |

Help → Examples reopens them (always fresh); Help → Restart tour replays a tour. `examples/build.py` rebuilds them.
Real clips are cut from Wikimedia Commons originals (NHTSA_Crash_Test_of_the_2006_Honda_Ridgeline.ogv frames
1448–1634; Ballistische_Kurve.webm frames 1532–1739; TdF_2022_Castelnau-Magnoac_Échauffement_Jumbo-Visma.webm by F123,
CC BY-SA 4.0).

**Coordinates.** x/y are in the calibration's units. Scale line: (0,0) at the Origin, x along the red arrow, y up
(green). Plane: positions on that surface, (0,0) at its bottom-left corner. Wheel: positions in the wheel's plane,
(0,0) where the wheel touches the ground, y = height. With "Camera moves" on, every frame is mapped back onto the frame
the calibration was drawn on, so the camera's own motion isn't counted.

For your own videos, the **Guide** panel on the right walks you through it: each step has a button and ticks itself off.

## Why it's nicer than Kinovea

- **Tracking that tells you when it's unsure.** Matches each point against both the last frame and the keyframe you
  placed (no slow drift), predicts motion so fast stuff stays in the search window, and **stops on the exact frame it
  loses confidence** with a red banner and a dashed best guess — press Enter to accept the guess or click where the
  point is, and it carries on. Esc skips that point and keeps tracking the rest.
- **Learns as you correct it.** Every click becomes a remembered appearance, so periodic motion (pedalling, running)
  needs fewer clicks each cycle. Motion model handles curved paths; handheld camera shake is compensated.
- **Bridges short occlusions** (configurable) and leaves the hidden frames empty rather than inventing them.
- **Fix by dragging.** Drag any point on any frame → it becomes a keyframe and everything after it is re-tracked up to
  your next keyframe. Loupe shows the actual pixels while you drag.
- **Clicks snap onto markers** (blob centre / corner, sub-pixel). Shift = place exactly.
- **Three trackers**: Template (precise), CSRT (shape changes), Blob (follows a colour: balls, tumbling things).
- **Moving camera**: tick "Camera moves" and the app measures the pan/zoom on every frame and cancels it out.
- **Scale from gravity** when there's no ruler in shot; **Wheel** calibration for bikes filmed at an angle.
- **Timeline lanes** per point: tracked, low-confidence (orange), keyframes (white ticks), lost (red).
- **Calibration**: scale line, or 4-point plane (perspective-correct), movable origin + rotatable axes.
- **Measurements**: 3-point angles, segment tilt, distances — live on the video, plotted over time.
- **Kinematics**: position/velocity/acceleration with a zero-lag Butterworth low-pass; cutoff picked automatically by
  residual analysis (or set it). Slow-mo capture-rate override.
- Plays anything ffmpeg does (incl. iPhone HEVC), frame-exact stepping, unlimited undo, **autosaves next to the video**
  and restores when you reopen it. Exports CSV, an overlay video, or a PNG.

## Keys

| | | | |
|---|---|---|---|
| Space | play/pause | ←/→ (Shift ×10) | step frames |
| T / Shift+T | track fwd/back | Ctrl+T | track selected point only |
| P A G D C O | point, angle, tilt, distance, scale, origin | V | select |
| Q | place selected point on this frame | Del / Shift+Del | delete point here / whole track |
| Ctrl+←/→ | prev/next keyframe or gap | [ ] \ | range start/end/clear |
| Wheel | zoom at cursor | right/middle drag | pan |
| Enter | accept the tracker's guess when it stops | Esc | skip a lost point / cancel |
| Ctrl+Z / Ctrl+Shift+Z | undo/redo | F1 | help |

## Dev

```
uv run tracklab
uv run python tests/synth.py tests/synth.mp4 && uv run python tests/test_tracker.py tests/synth.mp4
QT_QPA_PLATFORM=offscreen uv run python tests/ui_smoke.py tests/synth.mp4 /tmp/shots
QT_QPA_PLATFORM=offscreen uv run python tests/ui_flows.py
uv run python tests/hard.py     # blur / flicker / rotation stress test
uv run python tests/pedal.py    # pedalling benchmark: clicks needed + silent errors (--sweep for 6 clips)
QT_QPA_PLATFORM=offscreen uv run python tests/tour_shots.py /tmp/tour   # screenshot every tour step
```

Layout: `tracklab/tracker.py` (trackers + run loop), `video.py` (frame-exact decode + cache), `kinematics.py`
(filtering, derivatives), `model.py` (project/JSON), `render.py` (overlays, shared by canvas and export), `ui/`.
