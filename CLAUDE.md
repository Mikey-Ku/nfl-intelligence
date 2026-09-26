# NFL Intelligence

Computer-vision pipeline that turns All-22 film into top-down player tracking. Personal
learning project; see `BRIEF.md` for scope and `README.md` for how to run it.

## Rules

- **Footage never enters the repo.** `data/`, `outputs/` and every video extension are
  gitignored. Never commit frames, crops, or clips either, including in tests or docs.
- Python 3.12 via `uv`. Run things with `uv run nfl ...` and tests with `uv run pytest`.
- Vision models start pretrained and get fine-tuned. Don't propose training from scratch.
- Input is video only (screen capture of the NFL Pro player, or clip files). Public tracking
  data may be used to evaluate, never as model input.
- Commit and push at pipeline stage boundaries or when asked; the repo is private
  (`Mikey-Ku/nfl-intelligence`). Never push footage, frames or weights.

## Layout

- `src/nfl_intel/capture.py` — screen-region / window capture (mss + Quartz) and video files
- `src/nfl_intel/detect.py` — YOLO player detector (pretrained, MPS on Apple silicon)
- `src/nfl_intel/track.py` — ByteTrack wrapper giving stable IDs
- `src/nfl_intel/teams.py` — jersey-color clustering into two teams, smoothed per track
- `src/nfl_intel/field.py` — field model in yards, homography, landmark names, field drawing
- `src/nfl_intel/pipeline.py` — `PlayTracker`: one frame in, `FrameResult` out
- `src/nfl_intel/overlay.py` — transparent always-on-top overlay window (PySide6)
- `src/nfl_intel/replay.py` — tracking CSV → top-down replay video
- `src/nfl_intel/calibrate.py` — click landmarks on a frame to build a homography
- `src/nfl_intel/evaluate.py` — score a tracking CSV against hand-labeled truth (yards error, recall, ID switches, MOTA)
- `src/nfl_intel/cli.py` — `nfl` command: check, windows, overlay, run, replay, eval, calibrate
