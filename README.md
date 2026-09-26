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
player's box, ID and team colour, with a top-down minimap in the corner once a
calibration is loaded (`--calib calibrations/sideline.json`). The overlay asks macOS to
leave it out of screen capture so the pipeline doesn't see its own boxes; if boxes ever
start stacking up, run with `--mirror` to get a side window instead.

## Clip files

```bash
uv run nfl run data/clips/*.mp4 --save-video          # one CSV per clip in outputs/
uv run nfl replay outputs/play01_sideline.csv          # top-down replay video
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
