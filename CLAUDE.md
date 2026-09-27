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
- `src/nfl_intel/detect.py` — YOLO player detector (pretrained, MPS on Apple silicon)
- `src/nfl_intel/track.py` — BoT-SORT (camera-motion compensation) / ByteTrack giving stable IDs
- `src/nfl_intel/teams.py` — jersey-color clustering into two teams, smoothed per track
- `src/nfl_intel/field.py` — field model in yards, homography, landmark names, field drawing
- `src/nfl_intel/fieldmask.py` — turf band, painted sidelines, grass colour; drives the field crop and the bench filter
- `src/nfl_intel/pipeline.py` — `PlayTracker`: one frame in, `FrameResult` out
- `src/nfl_intel/stats.py` — `LiveStats`: play segmentation, team counts, tracking health, value/plus-minus hooks
- `src/nfl_intel/panel.py` — renders the stats side panel (QPainter), shared by the window, test shots and tests
- `src/nfl_intel/overlay.py` — transparent overlay window + docked stats panel window (PySide6)
- `src/nfl_intel/bdb.py` — Big Data Bowl 2026 loader (normalised so offense moves +x) and nflverse outcomes
- `src/nfl_intel/value.py` — pass value: per-receiver features, catch + yards models, live kinematics
- `src/nfl_intel/value_train.py` — trains/evaluates the value models (weeks 1-14 train, 15-18 test)
- `src/nfl_intel/demo.py` — replays a Big Data Bowl play top-down with the value panel
- `src/nfl_intel/replay.py` — tracking CSV → top-down replay video
- `src/nfl_intel/calibrate.py` — click landmarks on a frame to build a homography
- `src/nfl_intel/evaluate.py` — score a tracking CSV against hand-labeled truth (yards error, recall, ID switches, MOTA)
- `src/nfl_intel/train.py` — keyframe pre-labeling, dataset YAML, two-stage YOLO fine-tune
- `src/nfl_intel/cli.py` — `nfl` command: check, windows, overlay, capture, run, replay, eval, calibrate, prelabel, train, value-train, value-demo
- `docs/PRIOR_ART.md` — what exists elsewhere, how it runs/tests/views, ranked improvements
