# NFL Intelligence

Computer-vision pipeline that turns All-22 film into top-down player tracking. Personal
learning project; see `BRIEF.md` for scope and `README.md` for how to run it.

## Rules

- **Footage never enters the repo.** `data/`, `outputs/` and every video extension are
  gitignored. Never commit frames, crops, or clips either, including in tests or docs.
- Python 3.12 via `uv`. Run things with `uv run nfl ...` and tests with `uv run pytest`.
- Vision models start pretrained and get fine-tuned. Don't propose training from scratch.
- At run time the only input is video (screen capture of the NFL Pro player, or clip files).
  The value model behind the plus/minus is trained on public NFL Big Data Bowl tracking data
  (decided 2026-09-27); that data lives in `data/bdb/`, outcomes in `data/nflverse/`, trained
  models in `models/`. None of it is ever committed.
- Commit and push at pipeline stage boundaries or when asked; the repo is private
  (`Mikey-Ku/nfl-intelligence`). Never push footage, frames or weights.

## Layout

- `src/nfl_intel/capture.py` — screen-region / window capture (mss + Quartz) and video files
- `src/nfl_intel/detect.py` — YOLO player detector (pretrained `yolo11s`, MPS on Apple silicon)
- `src/nfl_intel/track.py` — BoT-SORT (camera-motion compensation) / ByteTrack giving stable IDs
- `src/nfl_intel/teams.py` — jersey-color clustering into two teams, smoothed per track
- `src/nfl_intel/field.py` — field model in yards, homography, landmark names, field drawing
- `src/nfl_intel/camera.py` — level pan/tilt/zoom camera model; fits a camera to yard-line and hash-row constraints
- `src/nfl_intel/registration.py` — field registration: paint → yard lines → numbering → hash rows → camera; `FieldRegistrar` follows the camera through a shot
- `src/nfl_intel/fieldmask.py` — turf band, painted sidelines, grass colour; drives the field crop and the bench filter
- `src/nfl_intel/pipeline.py` — `PlayTracker`: one frame in, `FrameResult` out
- `src/nfl_intel/ocr.py` — text off a frame with Apple Vision (on-device)
- `src/nfl_intel/playinfo.py` — NFL Pro's play header → the play's row in nflverse play-by-play and FTN charting
- `src/nfl_intel/anchor.py` — pins the relative mapping to the real field: line of scrimmage, direction, end-zone finder
- `src/nfl_intel/fieldtrack.py` — tracking in yards (Hungarian assignment) and snap detection
- `src/nfl_intel/roles.py` — QB / REC / OL / DEF / DL from the formation (+ charted shotgun/hash hints), corrected by what players do after the snap; the look in words
- `src/nfl_intel/roles_check.py` — marks role assignment against Big Data Bowl snaps (`nfl roles-check`)
- `src/nfl_intel/nflverse.py` — fetches the season's public play-by-play / charting (`nfl data`)
- `src/nfl_intel/imaging.py` — `shrink`: fast resize for 4K frames with the geometry of cv2.resize
- `src/nfl_intel/analysis.py` — `PlayAnalyzer` (play → anchor → snap → roles → value) and `AnalyzedTracker`, which wraps `PlayTracker`
- `src/nfl_intel/topdown.py` — top-down view of an anchored play, `TopDownWriter` for `nfl run --topdown`
- `src/nfl_intel/stats.py` — `LiveStats`: play segmentation, team counts, tracking health, value/plus-minus hooks
- `src/nfl_intel/panel.py` — renders the stats side panel (QPainter), shared by the window, test shots and tests
- `src/nfl_intel/overlay.py` — transparent overlay window + docked stats panel window (PySide6)
- `src/nfl_intel/bdb.py` — Big Data Bowl 2026 loader (normalised so offense moves +x) and nflverse outcomes
- `src/nfl_intel/value.py` — pass value: per-receiver features, catch + yards models, live kinematics
- `src/nfl_intel/coverage.py` — coverage recognition (man/zone + shell) from defender positions, labels from nflverse participation
- `src/nfl_intel/value_train.py` — trains/evaluates the value models (weeks 1-14 train, 15-18 test)
- `src/nfl_intel/demo.py` — replays a Big Data Bowl play top-down with the value panel
- `src/nfl_intel/replay.py` — tracking CSV → top-down replay video (`render_topdown` for mapped shots)
- `src/nfl_intel/calibrate.py` — click landmarks on a frame to build a homography
- `src/nfl_intel/evaluate.py` — score a tracking CSV against hand-labeled truth (yards error, recall, ID switches, MOTA)
- `src/nfl_intel/train.py` — keyframe pre-labeling, dataset YAML, two-stage YOLO fine-tune
- `src/nfl_intel/cli.py` — `nfl` command: check, windows, overlay, capture, run, replay, eval, calibrate, prelabel, train, value-train, coverage-train, value-demo, data, roles-check
- `docs/PRIOR_ART.md` — what exists elsewhere, how it runs/tests/views, ranked improvements
