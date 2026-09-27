# NFL Intelligence

Turns All-22 game film into top-down tracking data with a computer-vision pipeline:
detect every player, keep a stable ID on each one from line-set to whistle, split the
teams by jersey colour, and map feet from pixels to yards on the field. Personal learning
project; see [BRIEF.md](BRIEF.md) for scope.

```
screen / clip ─► YOLO (pretrained) ─► ByteTrack ─► jersey k-means ─► homography ─► CSV + overlay + replay
```

## Setup

```bash
uv sync --all-groups
uv run nfl check
```

`nfl check` loads the weights (downloaded on first run), times inference on the Apple GPU,
and saves a test screen grab. If the grab is blank or window titles are missing, give your
terminal app **Screen Recording** permission in System Settings → Privacy & Security, then
restart it.

## Live overlay on NFL Pro

Play a clip in the browser, then:

```bash
uv run nfl windows            # find the player window
uv run nfl overlay --window "Chrome" --out outputs/live.csv
```

A transparent, click-through window sits on top of the player and draws each tracked
player's box, ID and team colour. A stats panel docks beside the video (left by default,
`--panel-side right`, `--no-panel`): players on the field per team, tracking health, and the
play value with its plus/minus once calibration and the value model exist. The window is
captured by ID, so the overlay never sees its own boxes and other windows can cover it.

`--shots outputs/shots` saves a test image (annotated field + panel) every few seconds.
`--seconds 40` stops on its own; Ctrl-C also ends cleanly and writes the CSV.

To iterate without NFL Pro open, save a raw clip once and re-run on it:

```bash
uv run nfl capture --window "NFL Pro" --seconds 20 --out data/clips/atl-gb-td.mp4
uv run nfl run data/clips/atl-gb-td.mp4 --shots outputs/shots
```

## Clip files

```bash
uv run nfl run data/clips/*.mp4 --save-video          # one CSV per clip in outputs/
uv run nfl replay outputs/play01_sideline.csv          # top-down replay video
uv run nfl eval outputs/play01_sideline.csv data/truth/play01.csv   # yards error, recall, ID switches
```

Useful flags on `overlay` and `run`: `--tracker bytetrack` (faster, fixed camera; the default
BoT-SORT compensates for camera pans), `--tiles 2` (2×2 overlapping slices for tiny far-sideline
players), `--weights yolo11s.pt` or your fine-tuned weights.

## The plus/minus (pass value)

For every receiver at every moment, two models estimate what would happen if the ball were
thrown to him now: the chance it's caught and the yards if it is. The play value is the best
receiver's expected yards; the plus/minus compares it with a typical play at the same moment
after the snap.

```bash
uv run nfl value-train                 # ~1 min: trains on 2023 weeks 1-14, tests on 15-18
uv run nfl coverage-train              # man/zone and coverage shell from defender movement
uv run nfl value-demo --week 16        # replays real plays top-down with the live panel
```

Public label files, all from nflverse releases on GitHub, go in `data/nflverse/`:
`play_by_play_2023.parquet` (results), `pbp_participation_2023.parquet` (pressure, coverage,
formation per play), `ftn_charting_2023.parquet` (blitzers, play action, catchable ball).
Coverage recognition is right on man vs zone 70% of the time at the snap and 77% after two
seconds (always guessing zone: 57%).

Needs the Big Data Bowl 2026 data in `data/bdb/2026-prediction/` and nflverse play-by-play in
`data/nflverse/play_by_play_2023.parquet`. On held-out plays the catch model scores AUC 0.76
and is calibrated within 3 points; throws to the model's top option gained 12.0 yards, second
option 9.4, third or lower 5.8. The data has no linemen, so pressure isn't modelled, and it
runs on sensor tracking until video positions are in yards.

## Fine-tuning the detector

The pretrained COCO `person` class finds players but the detector is the bottleneck on film
(see [docs/PRIOR_ART.md](docs/PRIOR_ART.md)). A few hundred corrected frames go a long way:

```bash
uv run nfl prelabel data/clips/*.mp4 --every 30       # keyframes + detector boxes as YOLO labels
# correct the boxes in X-AnyLabeling (reads/writes YOLO txt), classes: player, referee
uv run nfl train --epochs 40                          # two-stage fine-tune at 1280 on the M5
uv run nfl overlay --window "Chrome" --weights runs/detect/nfl/weights/best.pt
```

## Field calibration

```bash
uv run nfl calibrate --video data/clips/play01_sideline.mp4 --frame 30 --out calibrations/sideline.json
```

Click a field landmark, then type its name in the terminal: `L40-H1` is the left 40 at
the near hash, `50-S2` is midfield at the far sideline, `REZ-S1` is the back of the right
end zone. Four points fit a homography; more points plus RANSAC make it sturdier. The fit
prints its mean error in yards as you go.

## Output

One CSV row per player per frame: `frame, t, track_id, team, conf, x1, y1, x2, y2,
foot_x_px, foot_y_px, x_yd, y_yd`. Field coordinates follow the Big Data Bowl convention
(x 0–120 along the field, y 0–53.3 across it), so anything built on that data runs on ours.

## Tests

```bash
uv run pytest
```
