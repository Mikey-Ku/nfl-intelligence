# Repository guide

Every file in the project, what it is for, and how the pieces fit. Read this before the
code. [SHOWCASE.md](SHOWCASE.md) says what the project achieves; this says how it is built.

## The shape of it

```
                     capture.py ──► pipeline.py (PlayTracker) ──► analysis.py (AnalyzedTracker) ──► overlay / cli
                                        │                                 │
                   per frame:   detect ─ track ─ teams ─ fieldmask ─ registration ─ camera
                                                                          │
                   per play:    playinfo ─ personnel ─ anchor ─ fieldnumbers ─ fieldtrack ─ roles ─ value ─ coverage ─ decision
                                                                          │
                   output:      schema (CSV) ─ stats ─ panel ─ topdown ─ draw ─ replay
```

Two objects do the work. `PlayTracker` (pipeline.py) takes one frame and returns a
`FrameResult`: boxes, track ids, teams, and feet in the mapping's own yards. `PlayAnalyzer`
(analysis.py) watches those results across a shot and turns them into football: which play,
where on the real field, the snap, who is who, the value. `AnalyzedTracker` wraps the two so
every command sees one object with `process(frame, idx, t)`.

Everything downstream talks through two small dataclasses in schema.py: `TrackedPlayer`
(one person in one frame) and `FrameResult` (one frame). The CSV written by `nfl run` is
those rows, so anything built on the Big Data Bowl layout can read it.

## Source files, in the order a frame meets them

### Getting frames

| File | What it does |
|---|---|
| `capture.py` | Three frame sources with the same `for idx, t, frame in source` shape. `WindowSource` grabs a window by id through Quartz (works when covered; HiDPI gives 2x pixels), `ScreenSource` grabs a screen region with mss, `VideoSource` reads a clip and, if `nfl capture` wrote a `.times.csv` beside it, uses the real capture times instead of the file's frame rate. `--start/--stop` live here. |
| `imaging.py` | `shrink`: a fast resize for 4K frames with the same geometry as `cv2.resize`. Block-average by a whole number, then one linear resample. Replacing OpenCV's area resize with it took the pipeline from 10 to 18 fps. |

### Finding people

| File | What it does |
|---|---|
| `detect.py` | `PlayerDetector`: a pretrained YOLO (`yolo11s` by default) on Apple's GPU, keeping the COCO `person` class; optional tiling for small far-sideline players. Fine-tuned weights plug in through `weights=`. |
| `track.py` | `PlayerTracker`: stable ids across frames from the `trackers` package. BoT-SORT by default, because it compensates for the camera panning; ByteTrack as the cheap option. |
| `teams.py` | Two teams from jersey colour. A tight torso crop, the median Lab colour of pixels that aren't grass, k-means into two clusters (team 0 is the darker jersey), votes per track so one bad frame can't flip a player, refits as the sample grows. `look()` labels a box with no track. |
| `fieldmask.py` | Where the grass is. The grass colour is measured, not assumed, so green jerseys survive. Gives the turf band, the painted sidelines (for the bench filter), and the field crop the detector runs on. |

### Mapping the field

| File | What it does |
|---|---|
| `registration.py` | Pixels to yards from the paint alone. Finds the painted lines, keeps the biggest family meeting at one vanishing point (the yard lines), numbers them so equal steps in number are equal steps on the field, finds the hash-mark rows and checks their dashes land on whole yards, then fits a camera. `FieldRegistrar` follows that camera through a shot: optical flow carries the mapping to the next frame, the paint refits pan, tilt and zoom. The biggest file; the module docstring walks through the steps. |
| `camera.py` | The camera model: level, fixed position, pan, tilt and zoom. Six unknowns, so yard lines plus one hash row are enough. A vectorised grid search finds the start, `scipy.optimize.least_squares` refines it. `looks_across` declines the end-zone angle, where the long lens breaks the model. |
| `field.py` | The field in yards (Big Data Bowl convention: x 0-120, y 0-53.3), landmark names like `L40-H1`, a hand-made `FieldCalibration` from clicked points, and `draw_field` for the minimap. |
| `calibrate.py` | `nfl calibrate`: click landmarks on a frame to build a homography by hand. The fallback when registration can't lock. |

### Reading the screen

| File | What it does |
|---|---|
| `ocr.py` | Text off an image with Apple's Vision framework, on device. Returns each piece of text with its box and confidence. |
| `playinfo.py` | Which play this is. Parses NFL Pro's header ("Q4 08:21 / 2ND & 7 / GB ON ATL 15") and the page's filter chips into a `PlayHeader`, finds its row in nflverse play-by-play, and joins FTN's charting. The `PlayRecord` carries the real yard line, down and distance, shotgun flag, hash mark, who was thrown to, and the result. |
| `personnel.py` | NFL Pro's "On Field Players" list, read at full size. Gives the personnel grouping ("11 personnel", "nickel"), how many linemen and eligibles to expect, and names for receivers the formation identifies beyond doubt. |
| `nflverse.py` | `nfl data`: fetches the season's public play-by-play and charting from nflverse's GitHub releases. No account needed. |

### Turning the picture into a play

| File | What it does |
|---|---|
| `fieldnumbers.py` | Lays the field flat through the mapping (never mirrored, digits can't be read backwards) and reads the big painted numbers. Each number's arrow points at the nearer goal line, which pins the mapping's yards to real yard lines. |
| `anchor.py` | Puts the play on the real field: the line of scrimmage from where the two teams face each other, the direction from the painted numbers, the end zone, the huddle or the formation, and the yard line from the play-by-play when it is known. `Anchor.to_absolute` converts the mapping's yards to the field's. |
| `fieldtrack.py` | Tracking in yards. `FieldTracker` matches each frame's detections to known players with a Hungarian assignment; a player hidden behind another rides with him until they part. `SnapDetector` calls the snap when a still formation starts moving, then walks back to the moment it began. |
| `roles.py` | Who is who at the snap: side of the ball first, jersey second, then the quarterback by where he stands behind the ball, linemen by the box, rushers by the line. Each side is held to eleven, so officials fall out. `update_roles` corrects roles from what players do after the snap. `describe_look` writes the formation in words. |
| `value.py` | The play value. Seventeen position-only features per receiver; a catch model and a yards model (gradient boosting, trained in value_train.py); the play value is the best receiver's expected yards; `typical()` is the baseline by time since the snap and stretch of field. `what_changed` writes the one-line reason for a swing. |
| `coverage.py` | Coverage recognition: man or zone, and which shell, from how the defenders stand and move after the snap. Labels come from nflverse participation data. |
| `decision.py` | Grades the throw, experimentally. The play-by-play says who was thrown to and how deep; that finds the target; a flight time fitted to 3,351 real throws works back to the moment of the throw; every receiver is scored at that moment. A grade is `firm` only if the target's rank holds half a second either side. |
| `analysis.py` | The conductor. `PlayAnalyzer.update` runs the per-play steps in order, only speaks when sure (no mapping, no anchor, no snap: no value), replays the sideline read on the end-zone view, and keeps a play log. `AnalysisView` is the face the panel talks to. `AnalyzedTracker` wraps `PlayTracker` and back-fills every frame of a shot into the final coordinates and roles when the shot ends. |

### Showing it

| File | What it does |
|---|---|
| `schema.py` | `TrackedPlayer`, `FrameResult`, and the CSV in and out. The one contract every stage shares. |
| `stats.py` | `LiveStats`: play segmentation, players per team, tracking health, and the value history with its biggest swings. Everything football-specific is plugged in as a function, so the panel works with or without the analysis. |
| `panel.py` | Draws the side panel with QPainter from a `StatsSnapshot`. One renderer for the live window, the test images and the tests. |
| `overlay.py` | The live windows: a transparent click-through overlay over the video and a docked panel, with a worker thread running the pipeline. |
| `topdown.py` | The top-down view of an anchored play on a slice of the real field, and `TopDownWriter` for `nfl run --topdown`. |
| `draw.py` | OpenCV drawing: boxes and labels on a frame, the minimap. |
| `replay.py` | `nfl replay`: a top-down video from a saved CSV. |

### Training and checking

| File | What it does |
|---|---|
| `bdb.py` | Loads Big Data Bowl 2026 tracking one play at a time, flipped so the offense always moves +x, joined with outcomes. |
| `value_train.py` | `nfl value-train`: fits the catch and yards models on weeks 1-14, reports on 15-18, and computes the typical-play curves (`nfl value-baseline`). |
| `roles_check.py` | `nfl roles-check`: marks role assignment against 2,572 real snaps, with and without the charted hints. |
| `plays_check.py` | `nfl plays-check`: marks a play log against the official charting (alignment, backs, box, rushers). |
| `evaluate.py` | `nfl eval`: position error, recall, identity switches and MOTA against a hand-labelled CSV. No truth file exists yet. |
| `train.py` | `nfl prelabel` and `nfl train`: keyframes with detector boxes to correct, a dataset YAML, a two-stage YOLO fine-tune. |
| `demo.py` | `nfl value-demo`: replays a Big Data Bowl play with the panel, on the league's own tracking. Where the value model was first watched. |
| `cli.py` | The `nfl` command. Every command is a `cmd_*` function; `_build_live` is where the tracker and the analysis are put together; `_compose` draws the film, top-down and panel side by side for test images and the demo video. |

### Outside `src/`

| Path | What it is |
|---|---|
| `scripts/make_showcase.py` | Builds the two showcase videos from chosen plays with captions. |
| `tests/` | 133 tests, synthetic data only, no footage. `formations.py` is the shared synthetic play (a shotgun formation and a film of it); `conftest.py` runs Qt offscreen. Each `test_*.py` matches a module or a stage. |
| `docs/SHOWCASE.md` | What it achieves, with every number's source. |
| `docs/PRIOR_ART.md` | What exists elsewhere and why the detector is the bottleneck. |
| `BRIEF.md`, `README.md`, `CLAUDE.md` | Scope, how to run it, rules for working in the repo. |
| `data/`, `models/`, `outputs/` | Film, public data, trained models and results. Never committed. |

## How a play goes through

1. A frame arrives. `fieldmask` finds the grass; the detector runs on that crop; BoT-SORT
   keeps ids; `teams` colours them; `registration` maps feet to the mapping's yards.
2. `PlayAnalyzer` reads the screen on the first mapped frames: the play, the roster.
3. Every few frames before the snap it tries to anchor: where the teams face each other,
   which way the painted numbers say the goal line lies, how far the play-by-play says.
4. Once the offense is at the line, the snap detector watches for the formation to move.
5. At the snap, roles are named from where everyone stood a tenth of a second earlier.
6. From then on every receiver is scored each frame, roles are corrected by what players
   do, lost players are re-found, the panel shows the value, and six seconds in the throw
   is graded.
7. When the shot ends, every frame of it is back-filled into the final anchor and roles, the
   play goes into the log, and what was learned is kept for the end-zone view of the same play.

## Conventions

- Absolute field coordinates follow the Big Data Bowl: the offense moves +x, the goal line it
  attacks is at x = 110, y runs across the field from the offense's right.
- The mapping's own yards (before anchoring) are relative to the first yard line seen.
  `FrameResult.anchored` says which a row is in.
- Roles: `QB`, `REC`, `OL`, `DEF`, `DL`, `OFF` (officials and anyone who doesn't fit).
- Labels: `QB`, `R1..R5` across the field from the offense's left, `D1..`, or a last name.
- Nothing in the analysis guesses. When it can't tell, the panel says what is missing.
