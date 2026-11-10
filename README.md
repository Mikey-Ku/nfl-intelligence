# NFL Intelligence

**Player tracking, formations and a live pass value, read straight off NFL All-22 film.**

![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Apple silicon](https://img.shields.io/badge/runs%20on-Apple%20silicon%20(MPS)-000000?logo=apple&logoColor=white)
![Tests](https://img.shields.io/badge/tests-158%20passing-2ea44f)
![License MIT](https://img.shields.io/badge/license-MIT-blue)

NFL Intelligence is a computer-vision pipeline that watches game film as it plays on screen
and turns it into the kind of analysis the league produces from its sensor data: every
player on a top-down field in real yards, who each one is, the formation, the coverage read,
and a running **play value** with a plus/minus for the pass the quarterback could throw right
now. Film is the only input; everything else is worked out.

Personal learning project, built solo on a MacBook. [BRIEF.md](BRIEF.md) has the scope,
[docs/SHOWCASE.md](docs/SHOWCASE.md) the full account of the results, and
[docs/REPO_GUIDE.md](docs/REPO_GUIDE.md) a tour of the code.

---

## Contents

- [What it does](#what-it-does)
- [Results at a glance](#results-at-a-glance)
- [Quick start](#quick-start)
- [Running it](#running-it)
- [Commands](#commands)
- [How it works](#how-it-works)
- [Project status](#project-status)
- [How to know it is working](#how-to-know-it-is-working)
- [Output files](#output-files)
- [Training the models](#training-the-models)
- [Data, models and what never enters the repo](#data-models-and-what-never-enters-the-repo)
- [Known limits](#known-limits)
- [Repository layout](#repository-layout)

---

## What it does

```
screen / clip ─► YOLO (pretrained) ─► BoT-SORT ─► jersey k-means ─► field mapping ─► yards ─► snap ─► roles ─► value ─► panel
      │                                                                                              ▲
      ├─► OCR: play header ─► play-by-play row ─► real yard line, down & distance, personnel, result ─┘
      └─► OCR: on-field roster ─► jersey-number reader + player looks ─► names on the players
```

Pointed at the NFL Pro player (or a saved clip), the pipeline:

1. **Finds every player** with a pretrained YOLO detector and keeps a stable identity on
   each one through the camera's pans (BoT-SORT).
2. **Splits the teams** by jersey colour and **maps the field** from the painted yard lines
   and hash marks, fitting a pan/tilt/zoom camera and following it frame to frame. Every
   foot on screen becomes a point in yards.
3. **Reads the screen.** Apple's on-device text recognition reads NFL Pro's play header
   ("Q4 08:21 / 2ND & 7 / GB ON ATL 15") and the on-field roster. The header finds the
   play's row in the public play-by-play; the roster gives the 22 names and numbers.
4. **Anchors to the real field** by reading the big painted yard numbers, whose arrows point
   at the nearer goal line, so the yards are the NFL's own coordinates.
5. **Finds the snap and names the roles** (quarterback, receivers, linemen, rushers,
   coverage) from where everyone stands, corrected by what they do after the snap.
6. **Names the players**: a jersey-number reader trained on our own film, and a "look"
   memory that recognises a man named once on later plays by his build, sleeves, socks and
   gloves.
7. **Scores every receiver, every frame**, with models trained on the NFL's Big Data Bowl
   tracking data: the chance a throw to him is caught and the yards if it is. The play
   value is the best throw available; the plus/minus compares it with a typical play from
   the same stretch of field at the same moment.
8. **Shows and saves it**: a transparent overlay on the video, a docked stats panel, a
   top-down replay, a tracking CSV, a play log, and a per-game table of every player on
   every play.

## Results at a glance

Measured on two games of NFL Pro film (about 17 minutes, 45 runs and passes) and checked
against FTN's official charting. Every number below is measured, and
[docs/SHOWCASE.md](docs/SHOWCASE.md) says where each one comes from.

| What | Result |
|---|---|
| Plays identified from the screen and matched to the play-by-play | every run and pass |
| Field anchored to real yard lines by reading the painted numbers | 36 of 40 |
| Quarterback picked out / passes given a live value | 39 of 42 / 22 of 24 |
| Backs, defenders in the box, pass rushers within one of the charting | 39 of 40, 36 of 42, 21 of 24 |
| Players named from their jersey number (held-out plays) | about 3.4 of 22 a play, 36 of 36 checked right |
| Players named from their look, on top of the numbers | about 0.8 more a play, roughly 85% right |
| Catch model on held-out weeks | AUC 0.76, calibrated within 3 points |
| Throughput, 4K capture with full analysis, Apple M5 | 17-18 frames a second |

`scripts/make_showcase.py` builds two demo videos in `outputs/showcase/`: a sideline cut
with the film and a top-down cut with no film in it, which is the one that can be shared.

## Quick start

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

Then either run a clip file:

```bash
uv run nfl run data/clips/my-play.mp4 --shots outputs/shots --topdown
```

or put the live overlay on a video playing in the browser:

```bash
uv run nfl windows                                           # find the player window
uv run nfl overlay --window "Chrome" --out outputs/live.csv
```

The play lookup and the names need the season's public play-by-play, fetched once:

```bash
uv run nfl data --season 2026
```

## Running it

### Live overlay on NFL Pro

```bash
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

### Clip files

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

### Manual calibration (fallback)

```bash
uv run nfl calibrate --video data/clips/play01_sideline.mp4 --frame 30 --out calibrations/sideline.json
```

Click a field landmark, then type its name: `L40-H1` is the left 40 at the near hash, `50-S2`
midfield at the far sideline, `REZ-S1` the back of the right end zone. Four points fit a
homography; more plus RANSAC make it sturdier, and the fit prints its mean error in yards as
you go. The automatic mapping makes this unnecessary on the sideline angle.

### The game table

```bash
uv run nfl game-table outputs/live-02.csv outputs/live-03.csv --out outputs/game.parquet
```

Joins a game's runs into one table, one row per player per play, with the per-player totals
beside it (`outputs/game.totals.parquet`). Parquet or CSV, by the extension.

## Commands

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
| | `eval PRED TRUTH` | score a tracking CSV against hand-labelled truth |
| Train | `prelabel CLIPS...` / `train` | keyframes with detector boxes to correct, then a two-stage YOLO fine-tune |
| | `value-train` / `value-baseline` | the pass value models and their typical-play curves |
| | `coverage-train` | man/zone and coverage shell from defender movement |
| | `value-demo --week 16` | replay held-out Big Data Bowl plays top-down with the live panel |
| | `python -m nfl_intel.jersey_train` | the jersey-number reader, on crops of our own film |
| | `python -m nfl_intel.looks_train` | the look embedding, on a run's own number-named players |
| Fallback | `calibrate --video CLIP` | click landmarks on a frame to build a homography by hand |

## How it works

On a sideline-angle shot the pipeline works the play out on its own, in this order. The
module names point into `src/nfl_intel/`.

| Step | What happens | Where |
|---|---|---|
| 1. Which play | NFL Pro's header is read with Apple's on-device OCR and matched to its row in the nflverse play-by-play: yard line, who has the ball, shotgun, and afterwards what happened | `ocr.py`, `playinfo.py` |
| 2. Where on the field | Paint is found (bright, colourless, thin); the biggest family of lines meeting at one vanishing point is the yard lines; hash rows are checked against whole yards; a level pan/tilt/zoom camera is fitted and followed | `fieldmask.py`, `registration.py`, `camera.py` |
| 3. Real yard lines | The field is laid flat through the camera and the painted numbers are read; their arrows point at the nearer goal line. With the yard line from step 1 that pins the mapping: offense moving +x, goal line at x = 110 | `fieldnumbers.py`, `anchor.py` |
| 4. Tracking in yards | Players are followed on the field, not in the picture (Hungarian assignment); a player hidden behind another rides with him until they part | `fieldtrack.py` |
| 5. The snap and the roles | The snap is a still formation starting to move. Just before it each player gets a role from where he stands, side of the ball first, jersey second; what he does after the snap corrects it | `roles.py` |
| 6. Who is who | The on-field roster is read; the jersey-number reader names what it can and the look book names players it has seen before; the formation names the rest where it leaves no doubt | `personnel.py`, `jersey.py`, `looks.py` |
| 7. The value | Every receiver is scored each frame by the catch and yards models; the panel shows the best throw, the plus/minus and the coverage read | `value.py`, `coverage.py` |
| 8. Out | Overlay, panel, top-down view, CSV, play log, game table | `overlay.py`, `panel.py`, `topdown.py`, `gametable.py` |

The line of scrimmage and the line to gain are drawn the way television does. The panel
also shows the nearest rusher to the quarterback: the value model cannot see the rush (its
training data has no linemen), so it sits beside the value rather than inside it.

**The end-zone angle** is not mapped, but NFL Pro shows each play from the sideline first
and the two views start at the same moment, so the sideline read is replayed in step with it,
marked "read from the sideline view".

**Naming players** is a 22-way choice, not reading a number from 0 to 99, because the roster
is on screen. The number reader is a ResNet-18 with a tens head and a units head on the torso
of every tracked box, trained with weak per-track labels: a crop may show its track's number
*or* nothing to read, and each track only has to show its number sometimes. The frames it is
sure of are added up through the shot and each roster number goes to at most one player on
its side. Once a man is named, his look goes into a book for the game; on later plays a
frame-by-frame vote among the on-field players on his side, in his kind of role, in his
shirt colour, can name him again. A number read always wins over a look. Off-the-shelf text
readers (Apple Vision, PARSeq, TrOCR, PP-OCR) were tried first: none read a dark jersey.

[docs/SHOWCASE.md](docs/SHOWCASE.md) has the longer story, including what was hard and
the two early results that turned out to be wrong.

## Project status

Last updated 2026-11-09. "Measured" means a number checked against charting, tracking data
or by eye on frames; "by eye" means no ground truth exists yet.

| Capability | Status | Measured | How to check |
|---|---|---|---|
| Player detection and tracking | Working, pretrained detector | 17-18 fps on 4K; linemen in a pile still merge | `nfl run --shots`, panel's Tracking block |
| Team split by jersey colour | Working | | overlay colours |
| Field mapping, sideline angle | Working | 70-100% of frames, 0.03-0.08 yd residual | `--shots` shows the fitted grid on the paint |
| Field mapping, end-zone angle | Not mapped | long lens defeats the camera model | panel says "read from the sideline view" |
| Play identified from the screen | Working | every run and pass of 45 | `play: ...` line while running; `matched` in the play log |
| Anchoring to real yard lines | Working | 36 of 40 from the numbers, rest from the formation | `line of scrimmage ... settled by` line while running; `anchored_by` in the play log |
| Snap detection | Working | 0.0-0.2 s late on 4 plays by eye, then corrected | `snap` in the play log; `--topdown` video |
| Roles from the formation | Working | QB 39 of 42 on film; 71% / 96% on 2,572 BDB snaps | `nfl plays-check`, `nfl roles-check` |
| Pass value and plus/minus | Working | catch AUC 0.76; 22 of 24 passes valued | `nfl value-train` prints the held-out scores |
| Coverage read (man/zone, shell) | Working, rough | 70% at the snap, 77% after 2 s (always zone: 57%) | `nfl coverage-train` |
| Names from jersey numbers | Working on light jerseys | 3.4 of 22 a play, 36 of 36 right; dark jerseys hardly read | `numbers` in the play log |
| Names from looks | Working, by eye only | +0.8 a play at ~85% | `looks` in the play log; `game-table` `known_by` |
| Throw grade | Experimental | throw moment within 0.6 s only 69% of the time | marked firm only when the rank holds |
| Game table | Working | 621 player-plays over 42 plays on the test game | `nfl game-table` |
| Detector fine-tuning | Tooling ready, not run | | `nfl prelabel`, `nfl train` |
| Officials | Told apart by count and position, not by sight | | |

**What is next.** Crop-level "readable" labels for dark jerseys so the number reader learns
them (a few hundred crops); fine-tuning the detector on corrected boxes, which most remaining
tracking errors trace back to; and ground truth for the looks so the 85% stops being an eye
estimate.

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
- **The panel's Tracking block**: players tracked (22 plus a few officials on a good shot),
  picked up late and lost (both turn red above 3), and a sparkline of players found over the
  last ten seconds.
- **The play log** (`<clip>.plays.json`): `matched` says the play was found in the
  play-by-play, `anchored_by` what settled the field (`painted numbers`, `end zone` or
  `formation`, or two of them), `snap` the snap time, `numbers` and `looks` which names came from where,
  `value` the curve and `peak` its high point. A play with `"matched": false` or no `snap`
  is one the pipeline gave up on, on purpose: it says what is missing rather than guessing.
- **Against the truth**:
  - `uv run nfl plays-check outputs/*.plays.json` marks every play log against FTN's
    charting: quarterback found, backs, box count, rushers, shotgun.
  - `uv run nfl roles-check --weeks 1-3` marks role assignment on 2,572 real snaps.
  - `uv run nfl eval outputs/play.csv data/truth/play.csv` scores yards error, recall,
    ID switches and MOTA against a hand-labelled CSV in the same schema.
  - `uv run nfl value-train` retrains the value models and prints the held-out AUC and
    calibration.
- **The code**: `uv run pytest` runs 158 tests on synthetic frames and tracks; no footage
  is needed or used.

## Output files

Every run writes to `outputs/` (gitignored):

| File | Contents |
|---|---|
| `<clip>.csv` | one row per player per frame: `frame, t, shot, track_id, team, conf, x1, y1, x2, y2, foot_x_px, foot_y_px, x_yd, y_yd, role, label, anchored`. With `anchored = 1` the yards are on the real field in the Big Data Bowl convention (x 0-120 along the field, offense moving +x; y 0-53.3 across), so anything built on that data runs on ours |
| `<clip>.plays.json` | one entry per play: situation, personnel, formation, who was found and how, the value curve against typical, the closest the rush got, the result |
| `<clip>.annotated.mp4`, `<clip>.demo.mp4`, `<clip>.shot<N>.topdown.mp4` | with `--save-video`, `--demo`, `--topdown` (one top-down video per anchored play) |
| `game.parquet` + `game.totals.parquet` | from `nfl game-table`: per player per play (team, side, role, name, number, `known_by` ∈ number / look / formation, position and depth at the snap, top speed and ground covered in the first five seconds, how often he was the best throw) and per-player totals |

## Training the models

Every model starts pretrained and is fine-tuned; nothing is trained from scratch.

| Model | Trained on | Command | Lands in |
|---|---|---|---|
| Detector (YOLO) | corrected keyframes from our clips | `nfl prelabel data/clips/*.mp4 --every 30`, fix the boxes in X-AnyLabeling, `nfl train --epochs 40` | `runs/detect/nfl/weights/best.pt` |
| Pass value (catch + yards) | Big Data Bowl 2026 tracking, nflverse outcomes; weeks 1-14 train, 15-18 test | `nfl value-train` (about a minute), `nfl value-baseline` | `models/pass_value.pkl` |
| Coverage (man/zone, shell) | Big Data Bowl + nflverse participation | `nfl coverage-train` | `models/coverage.pkl` |
| Jersey-number reader | torso crops from our film with per-track labels, plus synthetic numbers | `python -m nfl_intel.jersey_train data/jersey --out models/jersey.pt` | `models/jersey.pt` |
| Look embedding | a run's own number-named players, as classes | `python -m nfl_intel.looks_train data/clips/live-04.mp4 outputs/live-04.csv` | `models/looks.pt` |

The number reader and look embedding switch on when their files exist; without them `nfl
run` says so and players keep their formation labels. The look model is per game: fine-tune
it on the game's first clip and it names players on the rest.

## Data, models and what never enters the repo

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

## Known limits

- **Linemen in a pile** come out as a few boxes, so heavy formations read roughly (a
  goal-line run charted with seven in the box read eight). Detector fine-tuning is the fix.
- **Dark jerseys** are hardly read by the number reader yet; the labelled training set was
  almost all light shirts.
- **The end-zone angle** is not mapped; the sideline read is replayed over it instead.
- **Run plays** get roles and a formation but no value: the models score passes.
- **The throw grade** cannot get better than the throw moment, and without tracking the ball
  that is within 0.6 s only 69% of the time.
- **Officials** are kept out of the eleven by count and position, not recognised by sight.
- **Values above 20 expected yards** on 4 of 22 passes are plausible but unverified.
- Without the play-by-play file, or on the end-zone angle, the panel says what is missing
  instead of guessing.

## Repository layout

```
src/nfl_intel/
  capture.py        screen / window capture (mss + Quartz) and video files
  detect.py         YOLO player detector (yolo11s, MPS)
  track.py          BoT-SORT / ByteTrack for stable IDs
  teams.py          jersey-colour clustering into two teams
  fieldmask.py      turf band, painted sidelines, grass colour
  registration.py   paint → yard lines → numbering → hash rows → camera; follows the camera through a shot
  camera.py         level pan/tilt/zoom camera model and its fit
  field.py          field model in yards, homography, landmarks, drawing
  fieldnumbers.py   reads the painted yard numbers and their arrows
  anchor.py         pins the mapping to the real field (line of scrimmage, direction, end zone, huddle)
  fieldtrack.py     tracking in yards, snap detection
  roles.py          QB / REC / OL / DEF / DL from the formation and after-snap behaviour
  ocr.py            on-device text recognition (Apple Vision)
  playinfo.py       NFL Pro's play header → nflverse play-by-play and FTN charting
  personnel.py      NFL Pro's on-field roster → personnel grouping and names
  jersey.py         jersey-number reader and roster-constrained voting
  looks.py          player looks: LookBook and per-shot looks
  value.py          pass value: features, catch + yards models, typical-play baseline
  coverage.py       man/zone and shell recognition
  decision.py       throw grade (experimental)
  analysis.py       PlayAnalyzer: play → anchor → snap → roles → names → value; the play log
  pipeline.py       PlayTracker: one frame in, FrameResult out
  stats.py / panel.py / overlay.py / topdown.py / replay.py   the live panel, overlay window and replays
  gametable.py      one table for a game
  jersey_train.py / looks_train.py / value_train.py / train.py   training
  roles_check.py / plays_check.py / evaluate.py                  marking against the truth
  bdb.py / nflverse.py                                           data loaders and fetchers
  demo.py / calibrate.py / draw.py / imaging.py / schema.py      BDB replay, manual calibration, drawing, fast resize, the CSV schema
  cli.py            the nfl command
tests/              158 tests, synthetic frames only
scripts/make_showcase.py   builds the demo videos
docs/               REPO_GUIDE.md (every file), SHOWCASE.md (results), PRIOR_ART.md (what exists elsewhere)
```

## Tests

```bash
uv run pytest
```

## Licence

MIT for the code. Film, tracking data and anything derived from them are not part of the
repository and carry their own terms (NFL Pro, Kaggle's Big Data Bowl, nflverse).
