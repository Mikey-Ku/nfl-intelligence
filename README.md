# NFL Intelligence

Turns All-22 game film into top-down tracking data with a computer-vision pipeline:
detect every player, keep a stable ID on each one from line-set to whistle, split the
teams by jersey colour, and map feet from pixels to yards on the field. Personal learning
project; see [BRIEF.md](BRIEF.md) for scope.

```
screen / clip ─► YOLO (pretrained) ─► BoT-SORT ─► jersey k-means ─► field mapping ─► CSV + overlay + replay
                                                                        │
      play header (OCR) ─► play-by-play row ─► real yard line ─► roles at the snap ─► pass value, live
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
player's box, team colour and, from the snap, his name in the play (QB, R1, D3). A stats
panel docks beside the video (left by default, `--panel-side right`, `--no-panel`): the play,
the formation, the play value with its plus/minus, the best throw, the coverage read, and
tracking health. The window is captured by ID, so the overlay never sees its own boxes and
other windows can cover it. `--no-analysis` goes back to tracking only.

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
uv run nfl run data/clips/*.mp4 --topdown             # + a top-down replay with the panel per play
uv run nfl replay outputs/play01_sideline.csv          # top-down replay video
uv run nfl eval outputs/play01_sideline.csv data/truth/play01.csv   # yards error, recall, ID switches
```

Useful flags on `overlay` and `run`: `--tracker bytetrack` (faster, fixed camera; the default
BoT-SORT compensates for camera pans), `--tiles 2` (2×2 overlapping slices for tiny far-sideline
players), `--weights yolo11m.pt` or your fine-tuned weights. The default is `yolo11s.pt`: on
the test clip the nano model lost four players a frame to low confidence and the target
receiver for most of the play, while the medium one found one more lineman at three times
the cost. With analysis on, a 4K capture runs at about 18 frames a second on the M5.

## The plus/minus (pass value)

For every receiver at every moment, two models estimate what would happen if the ball were
thrown to him now: the chance it's caught and the yards if it is. The play value is the best
receiver's expected yards; the plus/minus compares it with a typical play at the same moment
after the snap.

```bash
uv run nfl value-train                 # ~1 min: trains on 2023 weeks 1-14, tests on 15-18
uv run nfl value-baseline              # only recompute the typical-play curves, by field position
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

## From film to football

On a sideline-angle shot the pipeline works the play out on its own, in this order:

1. **Which play.** NFL Pro's header ("Q4 08:21 / 2ND & 7 / GB ON ATL 15") is read off the
   captured window with Apple's on-device text recognition and matched to its row in the
   nflverse play-by-play (`data/nflverse/play_by_play_<season>.parquet`). That row gives the
   yard line, who has the ball, shotgun or not, and afterwards what happened.
2. **Where on the field.** The field is laid flat through the mapping and the big painted
   yard numbers are read off it; each number's arrow points at the nearer goal line. With
   the yard line from step 1 that pins the mapping to the real field: offense moving +x,
   goal line at x = 110. (No numbers in view: the painted end zone does the same job.)
3. **Tracking in yards.** Players are followed on the field rather than in the picture, so a
   player keeps his identity when the image tracker hands him a new number.
4. **The snap and the roles.** The snap is where a still formation starts moving. Just before
   it, each player gets a role from where he stands: side of the ball first, jersey second.
5. **The value.** From the snap, every receiver is scored by the pass value model and the
   panel shows the best throw, the plus/minus against a typical play from the same stretch
   of field, and the coverage read.
6. **Who is out there.** NFL Pro's "On Field Players" list is read too: the personnel
   grouping ("11 personnel v nickel", "jumbo, 7 linemen") and, where the formation leaves no
   doubt, names on the receivers (a lone back, a lone tight end, the man the ball went to).

The line of scrimmage and the line to gain are drawn on the video the way television does.

**Without the page around the video** (full screen, or another source) there is no header
to read. `--no-play-lookup` runs that way on purpose: the painted numbers give the yard
lines, the huddle or the formation says who has the ball, and the value still runs. What is
lost is everything the play-by-play knows: down and distance, personnel, names, the result.

On the saved clip (Packers-Falcons, 2026 week 3) all three plays were matched and both
sideline shots anchored from the end zone. On Love's touchdown to Golden the quarterback, all
five eligible receivers and four rushers were named (the charting says four rushed), and the
formation read "Shotgun, 3 x 1, 1 back", which is what was on the field. Golden was the
model's best throw for the first second, then the tight end and the back took over as his
defender closed; the charting calls the catch contested. The clip stops 2.2 seconds after
the snap, before the ball is thrown.

The snap is only looked for once the offense is at the line: the first version took the
break of the huddle for the snap on the goal-line run.

How good are the roles? `nfl roles-check` marks them against Big Data Bowl tracking, which
has no linemen at all (a hard version of linemen lost in a pile), with 0.4 yd of noise added:

| 2,572 pass plays, 2023 weeks 1-3 | formation alone | with the play's charting |
|---|---|---|
| quarterback picked out | 71% | 96% |
| route runners named receivers | 92% | 98% |
| coverage men named coverage | 94% | 95% |

The charting (shotgun or under centre, which hash the ball is on) comes with the matched
play. Without it, and with no linemen to show where the ball is, the quarterback is often
confused with the back beside him; on film the linemen that are found close most of that gap.

What it gets wrong today: linemen in a pile come out as a few boxes, so heavy formations
read roughly (the goal-line run was charted with three in the backfield and seven in the
box; we saw two and eight); officials are not told apart from players; the typical-play
names for receivers on the same side can't be told apart without reading jersey numbers.
Run plays get roles and a formation but no value:
the model scores passes. Without the play-by-play file, or on the end-zone angle, the panel
says what is missing instead of guessing.

```bash
uv run nfl data --season 2026          # fetch this week's play-by-play and charting (public, no account)
uv run nfl roles-check --weeks 1-3     # the table above
```

## Fine-tuning the detector

The pretrained COCO `person` class finds players but the detector is the bottleneck on film
(see [docs/PRIOR_ART.md](docs/PRIOR_ART.md)). A few hundred corrected frames go a long way:

```bash
uv run nfl prelabel data/clips/*.mp4 --every 30       # keyframes + detector boxes as YOLO labels
# correct the boxes in X-AnyLabeling (reads/writes YOLO txt), classes: player, referee
uv run nfl train --epochs 40                          # two-stage fine-tune at 1280 on the M5
uv run nfl overlay --window "Chrome" --weights runs/detect/nfl/weights/best.pt
```

## Field mapping (pixels to yards)

Automatic, from the painted lines. `nfl run` and `nfl overlay` do it on their own:

1. Find the paint (bright, colourless, thin), ignoring players, end-zone lettering and logos.
2. The biggest family of lines meeting at one vanishing point is the yard lines; they are
   numbered so equal steps in number are equal steps on the field.
3. Rows of hash marks are found and checked: their dashes must sit on whole yards.
4. A level pan/tilt/zoom camera is fitted to it all, then followed frame to frame.

On the saved clip the sideline angle maps 70-100% of frames with the fitted lines on the paint
(0.03-0.08 yd error against the lines used). The end-zone angle (camera looking down the
field) is **not mapped yet**: through its long lens the scale across the field comes out
wrong, so the pipeline declines to map it. The mapping's own yard numbers start from the
first line seen; anchoring (above) turns them into real yard lines. A first lock takes about
a second; following takes 25-50 ms a frame.

## Manual calibration (fallback)

```bash
uv run nfl calibrate --video data/clips/play01_sideline.mp4 --frame 30 --out calibrations/sideline.json
```

Click a field landmark, then type its name in the terminal: `L40-H1` is the left 40 at
the near hash, `50-S2` is midfield at the far sideline, `REZ-S1` is the back of the right
end zone. Four points fit a homography; more points plus RANSAC make it sturdier. The fit
prints its mean error in yards as you go.

## Output

One CSV row per player per frame: `frame, t, shot, track_id, team, conf, x1, y1, x2, y2,
foot_x_px, foot_y_px, x_yd, y_yd, role, label, anchored`. With `anchored` = 1 the yards are
on the real field in the Big Data Bowl convention (x 0–120 along the field with the offense
moving +x, y 0–53.3 across it), so anything built on that data runs on ours. With 0 they are
relative to the mapping's first yard line.

## Tests

```bash
uv run pytest
```
