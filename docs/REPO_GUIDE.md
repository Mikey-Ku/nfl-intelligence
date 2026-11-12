# Repository guide

Every file in the project, what it is for, and how the pieces fit. Read this before the
code. [SHOWCASE.md](SHOWCASE.md) says what the project achieves; this says how it is built.

## The shape of it

```
                     capture.py ──► pipeline.py (PlayTracker) ──► analysis.py (AnalyzedTracker) ──► overlay / cli
                                        │                                 │
                   per frame:   detect ─ track ─ teams ─ fieldmask ─ registration ─ camera
                                                                          │
                   per play:    playinfo ─ personnel ─ anchor ─ fieldnumbers ─ fieldtrack ─ roles ─ jersey ─ looks ─ value ─ coverage ─ decision
                                                                          │
                   output:      schema (CSV) ─ stats ─ panel ─ topdown ─ draw ─ replay ─ gametable
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
| `teams.py` | Two teams from jersey colour. A tight torso crop, the median Lab colour of pixels that aren't grass, k-means into two clusters (team 0 is the darker jersey), votes per track so one bad frame can't flip a player, refits as the sample grows. `look()` labels a box with no track. Good enough to tell the sides apart before the snap; after it, `kits.py` is what the tracker goes by. |
| `fieldmask.py` | Where the grass is. The grass colour is measured, not assumed, so green jerseys survive. Gives the turf band, the field crop the detector runs on, and the bench filter: the painted sidelines in the picture, or, once the field is mapped, the white border measured in yards through the mapping (`border_edges`). |

### Mapping the field

| File | What it does |
|---|---|
| `registration.py` | Pixels to yards from the paint alone. Finds the painted lines, keeps the biggest family meeting at one vanishing point (the yard lines), numbers them so equal steps in number are equal steps on the field, finds the hash-mark rows and checks their dashes land on whole yards, then fits a camera. `FieldRegistrar` follows that camera through a shot: optical flow carries the mapping to the next frame, the paint refits pan, tilt and zoom. The sidelines are the inner edge of the white border; the first time one shows, the camera's position is solved again with it, and that position is carried to later shots. The biggest file; the module docstring walks through the steps. |
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
| `kits.py` | Which team a box shows, taught by the line-up. Before the snap the teams stand on opposite sides of the ball, so each box is a labelled picture of one kit (helmet, jersey and pants, as colour histograms of three bands of the box). A small classifier is fitted on the last 2.5 s of those at the snap and reads every later box, whatever the image tracker calls it. On a player clear of the pile it agrees with his side on practically every frame. |
| `fieldtrack.py` | Tracking in yards. `FieldTracker` matches each frame's detections to known players with a Hungarian assignment; a detection stays with the player who had its image track, a box the image tracker has not confirmed never starts a player, a second track started on a man takes over his number when the first stops, and a player hidden behind another rides with him until they part. Going without a box for a frame has a price (3.5 yd), so a player whose box is missing does not take his neighbour's and push him onto a third; and a player is not put on a box that shows the other team's kit while anything else will do, which is what stops him following the image tracker's box when it slides to the man he is blocking. `SnapDetector` calls the snap when a still formation starts moving, then walks back to the moment it began. |
| `roles.py` | Who is who at the snap: side of the ball first, jersey second, then the quarterback by where he stands behind the ball, linemen by the box, rushers by the line. Each side is held to eleven, so officials fall out. `update_roles` corrects roles from what players do after the snap. `describe_look` writes the formation in words. |
| `jersey.py` | Who a track is, read off his shirt. `torso_crop`, `NumberNet` (a pretrained ResNet-18 with a tens head and a units head), `NumberVotes` adding up the frames the reader is sure of, `assign` giving each roster number to at most one track on clear evidence, and `ShotNumbers`, the piece the analyzer calls. The roster from the screen makes it a 22-way choice. |
| `looks.py` | Knowing a named player again by his look. `LookEmbedder` (ResNet-18 body features in three parts, fine-tuned by `looks_train.py`), `LookBook` (a running mean look per team and number for the game, saved and loaded), `ShotLooks` (each track's recent looks), and `recognise_many`, a frame-by-frame vote that names a man only when he is clearly ahead of the runner-up. |
| `value.py` | The play value. Seventeen position-only features per receiver; a catch model and a yards model (gradient boosting, trained in value_train.py); the play value is the best receiver's expected yards; `typical()` is the baseline by time since the snap and stretch of field. `what_changed` writes the one-line reason for a swing. |
| `coverage.py` | Coverage recognition: man or zone, and which shell, from how the defenders stand and move after the snap. Labels come from nflverse participation data. |
| `decision.py` | Grades the throw, experimentally. The play-by-play says who was thrown to and how deep; that finds the target; a flight time fitted to 3,351 real throws works back to the moment of the throw; every receiver is scored at that moment. A grade is `firm` only if the target's rank holds half a second either side. |
| `analysis.py` | The conductor. `PlayAnalyzer.update` runs the per-play steps in order, only speaks when sure (no mapping, no anchor, no snap: no value), reads numbers and looks every frame and names players from them every 15 mapped frames (a number beats a look beats the formation's guess), replays the sideline read on the end-zone view, and keeps a play log. `AnalysisView` is the face the panel talks to. `AnalyzedTracker` wraps `PlayTracker` and back-fills every frame of a shot into the final coordinates and roles when the shot ends. |

### Showing it

| File | What it does |
|---|---|
| `schema.py` | `TrackedPlayer`, `FrameResult`, and the CSV in and out. The one contract every stage shares. |
| `stats.py` | `LiveStats`: play segmentation, players per team, tracking health, the value history with its biggest swings, and the panel's blocks (`PanelSection`) when the caller has them. Everything football-specific is plugged in as a function, so the panel works with or without the analysis. |
| `panel.py` | Draws the side panel with QPainter from a `StatsSnapshot`. One renderer for the live window, the test images and the tests. With a play read it draws the play, its value and swings, then a block for the offense and one for the defense (formation or look, grouping, what it is doing now, the eleven by position with the players the film has found in white); with tracking alone it draws the counts per jersey group and the tracking health. |
| `overlay.py` | The live windows: a transparent click-through overlay over the video and a docked panel, with a worker thread running the pipeline. |
| `topdown.py` | The top-down view of an anchored play on a slice of the real field, and `TopDownWriter` for `nfl run --topdown`. |
| `draw.py` | OpenCV drawing: boxes and labels on a frame, the minimap. |
| `replay.py` | `nfl replay`: a top-down video from a saved CSV. |

### Training and checking

| File | What it does |
|---|---|
| `bdb.py` | Loads Big Data Bowl 2026 tracking one play at a time, flipped so the offense always moves +x, joined with outcomes. |
| `value_train.py` | `nfl value-train`: fits the catch and yards models on weeks 1-14, reports on 15-18, and computes the typical-play curves (`nfl value-baseline`). |
| `jersey_mark.py` | Marks torso crops readable or nothing-to-read by hand, a page of one track's crops at a time (`marks.csv`); `jersey_train` trains marked crops as plain examples. The way to teach the reader dark jerseys. |
| `jersey_train.py` | Trains the number reader on torso crops in `data/jersey/` (labels.csv + crops/), one number per track. Each crop may be its track's number or "nothing to read", each track must show its number sometimes, and half of every batch is made-up torsos so the digits are learned as digits. |
| `looks_train.py` | Fine-tunes the look embedding as a cosine classifier over the (team, number) identities a run named by number; the labels come free from the run's CSV and play log. |
| `gametable.py` | `nfl game-table`: one table for a game from runs' CSVs and play logs, a row per player per play (who, how known, snap position and depth, top speed, distance run, moments as the best throw) and per-player totals. |
| `roles_check.py` | `nfl roles-check`: marks role assignment against 2,572 real snaps, with and without the charted hints. |
| `plays_check.py` | `nfl plays-check`: marks a play log against the official charting (alignment, backs, box, rushers). |
| `truth.py` | `nfl truth`: click where each player stands on frames of a run, half a second apart; clicks become yards through the run's own mapping, numbers carry by nearness; writes truth in the tracking schema. |
| `evaluate.py` | `nfl eval`: position error (with its steady offset and the spread around it), recall, identity switches and MOTA against that truth. Players are paired after the offset is taken out, and identity is the field tracker's `slot`. |
| `train.py` | `nfl prelabel` and `nfl train`: keyframes with detector boxes to correct, a dataset YAML, a two-stage YOLO fine-tune. |
| `boxes.py` | `nfl boxes`: the window for correcting those boxes (draw, move, resize, delete, player or referee, zoom), reading and writing the YOLO text files in place. |
| `demo.py` | `nfl value-demo`: replays a Big Data Bowl play with the panel, on the league's own tracking. Where the value model was first watched. |
| `cli.py` | The `nfl` command. Every command is a `cmd_*` function; `_build_live` is where the tracker and the analysis are put together; `_compose` draws the film, top-down and panel side by side for test images and the demo video. |

### Outside `src/`

| Path | What it is |
|---|---|
| `scripts/make_showcase.py` | Builds the two showcase videos from chosen plays with captions. |
| `tests/` | 210 tests, synthetic data only, no footage. `formations.py` is the shared synthetic play (a shotgun formation and a film of it); `conftest.py` runs Qt offscreen. Each `test_*.py` matches a module or a stage. |
| `README.md` | The front page: what it does, what it looks like, quick start. |
| `docs/USAGE.md` | Every command and flag, output files, training, how to know it is working. |
| `docs/STATUS.md` | Where each capability stands, with its measured number, what is next, known limits. |
| `docs/SHOWCASE.md` | What it achieves, with every number's source. |
| `docs/PRIOR_ART.md` | What exists elsewhere and why the detector is the bottleneck. |
| `docs/BRIEF.md` | The original scope. |
| `docs/media/` | Pictures for the README: top-down renders, panel images, and one eight-second GIF of film beside the tracking (the only footage in the repository, committed on purpose). |
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
   is graded. Jersey numbers are read all along; a number read clearly puts the roster name
   on the player, and a player named once is known by his look on later plays.
7. When the shot ends, every frame of it is back-filled into the final anchor and roles, the
   play goes into the log, and what was learned is kept for the end-zone view of the same play.

## Conventions

- Absolute field coordinates follow the Big Data Bowl: the offense moves +x, the goal line it
  attacks is at x = 110, y runs across the field from the offense's right.
- The mapping's own yards (before anchoring) are relative to the first yard line seen.
  `FrameResult.anchored` says which a row is in.
- Roles: `QB`, `REC`, `OL`, `DEF`, `DL`, `OFF` (officials and anyone who doesn't fit).
- Labels: `QB`, `R1..R5` across the field from the offense's left, `D1..`, or a last name; `R?`/`D?`
  when a number read took the formation's guessed name off a player.
- Nothing in the analysis guesses. When it can't tell, the panel says what is missing.
