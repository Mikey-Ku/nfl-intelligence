# Using NFL Intelligence

Everything you need to run the pipeline, every command and flag, what it writes, and how
to tell it is working. The [README](../README.md) is the short version.

## Contents

- [Setup](#setup)
- [Live overlay on NFL Pro](#live-overlay-on-nfl-pro)
- [Clip files](#clip-files)
- [The game table](#the-game-table)
- [Manual calibration](#manual-calibration)
- [Command reference](#command-reference)
- [How to know it is working](#how-to-know-it-is-working)
- [Output files](#output-files)
- [Training the models](#training-the-models)
- [Data and models on disk](#data-and-models-on-disk)

## Setup

Requires macOS on Apple silicon, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Mikey-Ku/nfl-intelligence.git
cd nfl-intelligence
uv sync --all-groups
uv run nfl check
```

`nfl check` loads the detector weights (downloaded on first run), times inference on the
Apple GPU and saves a test screen grab. A healthy machine prints something like:

```
nfl-intelligence 0.1.0
python 3.12.x  torch 2.x  mps=True
weights yolo11s.pt loaded on mps in 1.2s
inference on a 1280x720 frame at imgsz=1280: 28 ms  (~36 fps)
screen capture: 800x600 px for a 400x300 pt region (scale 2.0) -> outputs/capture_test.png
windows: 41 (12 on screen), 12 with titles
```

If the grab is blank or no windows have titles, give your terminal app **Screen Recording**
permission in System Settings → Privacy & Security and restart it.

The play lookup and the names need the season's public play-by-play, fetched once:

```bash
uv run nfl data --season 2026
```

## Live overlay on NFL Pro

Play a clip in the browser, then:

```bash
uv run nfl windows                                           # find the player window
uv run nfl overlay --window "Chrome" --out outputs/live.csv --shots outputs/shots
```

A transparent, click-through window sits on top of the player and draws each tracked
player's box, team colour and, from the snap, his name (a jersey-number read, a recognised
look, or the formation's guess: QB, R1, D3). A stats panel docks beside the video (left by
default, `--panel-side right`, `--no-panel`) showing the play, the formation, the play value
with its plus/minus, the best throw, the coverage read, the nearest rusher and tracking
health. The window is captured by ID, so the overlay never sees its own drawing and other
windows can cover it.

Useful flags: `--seconds 40` stops on its own (Ctrl-C also ends cleanly and writes the CSV);
`--record out.mp4` saves the annotated capture; `--mirror` shows the capture in a normal
window instead of an overlay; `--no-analysis` goes back to tracking only.

## Clip files

Save a raw clip once and iterate on it without NFL Pro open:

```bash
uv run nfl capture --window "NFL Pro" --seconds 20 --out data/clips/atl-gb-td.mp4
uv run nfl run data/clips/atl-gb-td.mp4 --shots outputs/shots
```

```bash
uv run nfl run data/clips/*.mp4 --save-video     # one CSV + annotated video per clip in outputs/
uv run nfl run data/clips/*.mp4 --topdown        # + a top-down replay with the panel for every anchored play
uv run nfl run data/clips/*.mp4 --demo           # + film, top-down view and panel side by side in one video
uv run nfl replay outputs/atl-gb-td.csv          # top-down replay from a tracking CSV
```

`--start` / `--stop` cut a clip; `--preview` shows frames as they are tracked.

Flags shared by `run` and `overlay`: `--tracker bytetrack` (faster, fixed camera; the default
BoT-SORT compensates for pans), `--tiles 2` (2×2 overlapping slices for tiny far-sideline
players), `--weights yolo11m.pt` or your own fine-tuned weights, `--no-play-lookup` for film
with no NFL Pro header around it (the painted numbers still give the yard lines and the value
still runs, but down and distance, personnel, names and the result are lost).

The default detector is `yolo11s.pt`: on the test clip the nano model lost four players a
frame to low confidence and the target receiver for most of the play, while the medium one
found one more lineman at three times the cost.

## The game table

```bash
uv run nfl game-table outputs/live-02.csv outputs/live-03.csv --out outputs/game.parquet
```

Joins a game's runs into one table, one row per player per play, with the per-player totals
beside it (`outputs/game.totals.parquet`). Parquet or CSV, by the extension. Columns: the
play (game, quarter, clock, down, distance, teams, type, result, yards, EPA, target,
formation reads), the player (team, side, role, label, name, number, `known_by` ∈ number /
look / formation), frames seen, position and depth off the line at the snap, top speed and
ground covered in the first five seconds, how often he was the best throw and the peak value
when he was.

## Manual calibration

The automatic mapping makes this unnecessary on the sideline angle; it is the fallback.

```bash
uv run nfl calibrate --video data/clips/play01_sideline.mp4 --frame 30 --out calibrations/sideline.json
```

Click a field landmark, then type its name: `L40-H1` is the left 40 at the near hash, `50-S2`
midfield at the far sideline, `REZ-S1` the back of the right end zone. Four points fit a
homography; more plus RANSAC make it sturdier, and the fit prints its mean error in yards as
you go.

## Command reference

All through `uv run nfl <command>`; `--help` on any of them lists the flags.

| Group | Command | What it does |
|---|---|---|
| Setup | `check` | environment check: weights, GPU timing, screen capture, window titles |
| | `data --season 2026` | fetch the season's public play-by-play and charting (nflverse; no account) |
| | `windows [query]` | list capturable windows |
| Run | `overlay --window NAME` | live overlay and stats panel on a playing video |
| | `capture --window NAME` | save the raw window capture to a clip |
| | `run CLIPS...` | track clip files: CSV, play log, optional videos |
| | `replay CSV` | top-down replay video from a tracking CSV |
| | `game-table CSVS... --out` | one table for a game: a row per player per play |
| Check | `plays-check LOGS...` | mark play logs against the play-by-play and the official charting |
| | `roles-check --weeks 1-3` | mark role assignment against real Big Data Bowl snaps |
| | `truth VIDEO CSV --shot N` | click where each player stands, every half second: hand-labelled truth for a play |
| | `eval PRED TRUTH` | score a tracking CSV against that truth: yards error, recall, ID switches among the players it names, MOTA |
| Train | `prelabel CLIPS...` / `boxes` / `train` | keyframes with the detector's boxes, a window to correct them, then a two-stage YOLO fine-tune |
| | `value-train` / `value-baseline` | the pass value models and their typical-play curves |
| | `coverage-train` | man/zone and coverage shell from defender movement |
| | `value-demo --week 16` | replay held-out Big Data Bowl plays top-down with the live panel |
| | `python -m nfl_intel.jersey_train` | the jersey-number reader, on crops of our own film |
| | `python -m nfl_intel.looks_train` | the look embedding, on a run's own number-named players |
| Fallback | `calibrate --video CLIP` | click landmarks on a frame to build a homography by hand |

## How to know it is working

While it runs, `nfl run` and `nfl overlay` narrate each shot as things are found:

```
  shot 3    4.2s  play: Q4 08:21  2nd & 7  GB on ATL 15  (pass, Love to Golden, 15 yd)
  shot 3    4.2s  on the field: GB in 11 personnel: 1 RB, 1 TE, 3 WR; ATL in nickel, 4-2-5
  shot 3    5.0s  line of scrimmage on the 15, 15 from the goal line, settled by the painted numbers
  shot 3    6.3s  lined up: Shotgun, 3 x 1, 1 back | Single high
  shot 3    7.9s  snap at 7.9s: Shotgun, 3 x 1, 1 back | Single high
  shot 3    8.0s  first value 6.1 yd, best throw Golden; receivers Golden, Kraft, R3, Jacobs, R5
  ...
data/clips/live-02.mp4: 1812 frames, 36240 rows -> outputs/live-02.csv  (17.6 fps, 61 track ids, 2 picked up late)
  13 plays worked out -> outputs/live-02.plays.json
```

Each line appears once per shot, as soon as that thing is known. A shot with no `line of
scrimmage` line was not anchored; one with no `snap` line never saw a still formation start
to move.

- **Test shots** (`--shots DIR`): an annotated frame plus the panel every few seconds. The
  fitted grid should sit on the paint, the line of scrimmage on the ball, and names on the
  players from the snap. A grid that drifts off the lines means the camera fit lost its lock.
- **The panel's tracking line** (its last line, when there is room): players in sight,
  picked up late and lost; it turns red when either of the last two passes 3. With
  `--no-analysis` the panel has the full block and a sparkline of players found over the last
  ten seconds.
- **The panel's two blocks**: each side's names should be the ones on NFL Pro's list, and a
  name turns white once the film has found that player (by number, by look, or by where the
  formation puts him).
- **The play log** (`<clip>.plays.json`): `matched` says the play was found in the
  play-by-play, `anchored_by` what settled the field (`painted numbers`, `end zone` or
  `formation`, or two of them), `snap` the snap time, `numbers` and `looks` which names came
  from where, `value` the curve and `peak` its high point. A play with `"matched": false` or
  no `snap` is one the pipeline gave up on, on purpose: it says what is missing rather than
  guessing.
- **Piles**: after the snap, four or more players from both teams packed within two yards
  of each other are drawn as one area, white outline on the film and a shaded circle with a
  head count on the top-down view, with nobody inside it named. A player gets his box and
  name back a quarter of a second after he is clear. In the CSV those rows have `pile` set;
  `slot` still carries the tracker's own guess.
- **`kits` in the play log**: `read_s` is how many player-seconds a named player's box
  showed a clear kit after the snap, `other_side_s` how many of those showed the other
  team's. That second number is identities sitting on the wrong man, measured with no labels:
  it runs at about 0.3 a play, and a play well above that is worth watching with `--shots`.
- **Against the truth**:
  - `uv run nfl plays-check outputs/*.plays.json` marks every play log against FTN's
    charting: quarterback found, backs, box count, rushers, shotgun.
  - `uv run nfl roles-check --weeks 1-3` marks role assignment on 2,572 real snaps.
  - `uv run nfl eval outputs/play.csv data/truth/play.csv` scores yards error, recall,
    ID switches and MOTA against a hand-labelled CSV in the same schema. The error is given
    whole and as the spread around its steady offset (the run puts a player at the bottom of
    his box, a labeller clicks a little higher); ID switches count changes of `slot`.
  - `uv run nfl value-train` retrains the value models and prints the held-out AUC and
    calibration.
- **The code**: `uv run pytest` runs 215 tests on synthetic frames and tracks; no footage
  is needed or used.

### Hand-labelling a play

The number that matters most, how far each tracked player is from where he really stood,
needs truth made by hand. `nfl truth` makes it:

```bash
uv run nfl truth data/clips/live-02.mp4 outputs/live-02.csv --shot 5     # label one play, a frame every 0.5 s
uv run nfl eval outputs/live-02.csv data/truth/live-02.shot5.csv          # score the run against it
```

A window shows each sampled frame. Click the feet of every player on the field (both
teams, not officials), then `n` for the next frame. With `--shot` and the run's play log
beside its CSV, the frames run from a second before the snap to six after it, about 15 of
them and ten minutes of clicking; `--whole-shot` labels the rest too.

| Key | |
|---|---|
| click | a player's feet |
| `n` / `p` | next / previous frame (the file is saved each time) |
| `u` | undo the last click |
| `x` | delete the click under the cursor |
| `i` | renumber the click under the cursor: type the number in the window, Enter (Esc leaves it) |
| `b` | show the run's boxes for reference (off by default so they don't lead the eye) |
| `q` | quit; running the same command again picks the play up at its first unlabelled frame |

Each click becomes yards through the run's own field mapping for that frame, so the score
measures detection and tracking; the mapping's own accuracy is measured separately against
the paint. Numbers carry from frame to frame so identity swaps can be counted: the whole
frame is matched at once to where each man was heading, so the order of clicking does not
matter. A number that is still wrong is fixed with `i`; if another click on the frame has
the number typed, the two trade.

[STATUS.md](STATUS.md) has the current state of every capability with its measured number.

## Output files

Every run writes to `outputs/` (gitignored):

| File | Contents |
|---|---|
| `<clip>.csv` | one row per player per frame: `frame, t, shot, track_id, team, conf, x1, y1, x2, y2, foot_x_px, foot_y_px, x_yd, y_yd, role, label, anchored, slot`. `track_id` is the image tracker's number, which changes whenever it loses a man; `slot` is the number the analysis keeps him under through the play. With `anchored = 1` the yards are on the real field in the Big Data Bowl convention (x 0-120 along the field, offense moving +x; y 0-53.3 across), so anything built on that data runs on ours |
| `<clip>.plays.json` | one entry per play: situation, personnel, formation, who was found and how, the value curve against typical, the closest the rush got, the result |
| `<clip>.annotated.mp4`, `<clip>.demo.mp4`, `<clip>.shot<N>.topdown.mp4` | with `--save-video`, `--demo`, `--topdown` (one top-down video per anchored play) |
| `game.parquet` + `game.totals.parquet` | from `nfl game-table` |

`scripts/make_showcase.py` builds the two demo videos in `outputs/showcase/`: a sideline
cut with the film (for showing in person) and a top-down cut with no film in it (the one
that can be shared).

## Training the models

Every model starts pretrained and is fine-tuned; nothing is trained from scratch.

| Model | Trained on | Command | Lands in |
|---|---|---|---|
| Detector (YOLO) | corrected keyframes from our clips | `nfl prelabel data/clips/*.mp4 --every 30`, then `nfl boxes` to fix them (drag to draw, click to select, `d` delete, `r` player/referee, `z` zoom, `n` next; split piles into one box per man, box the missed quarterback, delete crowd boxes), then `nfl train --epochs 40` (about 45 minutes on an M5) | `runs/detect/nfl/weights/best.pt`; pass it with `--weights`, and only the `player` class is kept, so officials drop out |
| Pass value (catch + yards) | Big Data Bowl 2026 tracking, nflverse outcomes; weeks 1-14 train, 15-18 test | `nfl value-train` (about a minute), `nfl value-baseline` | `models/pass_value.pkl` |
| Coverage (man/zone, shell) | Big Data Bowl + nflverse participation | `nfl coverage-train` | `models/coverage.pkl` |
| Jersey-number reader | torso crops from our film with per-track labels, plus synthetic numbers; crops marked readable or not by hand (`python -m nfl_intel.jersey_mark data/jersey --kind dark`, a page of one track's crops at a time, click green for readable and red for nothing to read) | `python -m nfl_intel.jersey_train data/jersey --out models/jersey.pt` | `models/jersey.pt` |
| Look embedding | a run's own number-named players, as classes | `python -m nfl_intel.looks_train data/clips/live-04.mp4 outputs/live-04.csv` | `models/looks.pt` |

The number reader and look embedding switch on when their files exist; without them `nfl
run` says so and players keep their formation labels. The look model is per game: fine-tune
it on the game's first clip and it names players on the rest.

## Data and models on disk

| Path | What | Source |
|---|---|---|
| `data/clips/` | raw captures and clip files | NFL Pro, captured by you |
| `data/nflverse/` | `play_by_play_<season>.parquet`, `pbp_participation_*.parquet`, `ftn_charting_*.parquet` | `nfl data` (public, nflverse releases) |
| `data/bdb/2026-prediction/` | Big Data Bowl 2026 tracking | Kaggle, competition terms, downloaded by you |
| `data/jersey/` | torso crops and `labels.csv` for the number reader | built from your own clips |
| `data/labels/`, `data/truth/` | corrected detector labels, hand-labelled tracking truth | yours |
| `models/` | every trained model | the commands above |
| `outputs/` | everything a run writes | |

**Footage and anything derived from it stays on this machine.** `data/`, `outputs/`,
`models/`, `runs/` and every video and weights extension are gitignored. Frames, crops, clips
and weights are never committed, including in tests or docs. The tests use synthetic frames.
The one exception is `docs/media/film.gif`, eight seconds of film beside the tracking,
committed on purpose for the README.
