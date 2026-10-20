# NFL Intelligence: what it does, how, and how well

A computer-vision pipeline that watches NFL All-22 film as it plays on screen and turns
it into the kind of live analysis the league shows from its sensor data: every player on a
top-down field in real yards, who is who, the formation, the coverage read, and a running
"play value" with a plus/minus for the pass the quarterback could throw right now.

Built solo, 2026-09-25 to 2026-10-09, about 8,400 lines of Python, 134 tests. Private
repository `Mikey-Ku/nfl-intelligence`.

## The one-minute version

Film is the only input at run time. Pointed at the NFL Pro player, the pipeline:

1. **Finds the players** with a pretrained YOLO detector and keeps identities with
   BoT-SORT, which compensates for the camera's pans.
2. **Maps the field.** It finds the painted yard lines and hash marks, fits a level
   pan/tilt/zoom camera to them, and follows that camera frame to frame. Every foot on
   screen becomes a point in yards.
3. **Reads the screen.** Apple's on-device text recognition reads NFL Pro's play header
   ("Q4 08:21 / 2ND & 7 / GB ON ATL 15") and the on-field roster beside the video. The
   header identifies the play's row in the public play-by-play; the roster gives personnel
   groupings and names.
4. **Anchors the yards to the real field** by laying the field flat through the camera
   model and reading the big painted numbers, whose arrows point at the nearer goal line.
5. **Tracks in yards**, finds the snap (a still formation starting to move), and names roles
   from where everyone stands: quarterback, receivers, linemen, rushers, coverage.
6. **Scores every receiver, every frame**, with models trained on the NFL's own tracking
   data (Big Data Bowl 2026): the chance a throw to him is caught and the yards if it is.
   The play value is the best throw available; the plus/minus compares it with a typical
   play from the same stretch of field at the same moment after the snap.

```
screen ─► YOLO ─► BoT-SORT ─► jersey split ─► field mapping ─► yards ─► snap ─► roles ─► value ─► panel
   │
   └─► OCR: play header ─► play-by-play row ─► real yard line, down, distance, personnel, result
```

## What the numbers say

Everything below is measured, and the note says where it came from.

**On film** (two games, about 17 minutes, 45 runs and passes; `nfl plays-check` against
FTN's official charting, 2026 week 3):

| | |
|---|---|
| Plays identified in the play-by-play from the screen | every run and pass |
| Field anchored to real yard lines from the painted numbers | 36 of 40 (the rest from the formation) |
| Quarterback picked out | 39 of 42 |
| Passes given a live value | 22 of 24 |
| Backs in the backfield, within one of the charting | 39 of 40 (exact 15) |
| Defenders in the box, within one | 36 of 42 (exact 16) |
| Pass rushers, within one | 21 of 24 (exact 6) |
| Shotgun vs under centre, from the formation alone | 26 of 40 |
| Throughput on a 3024x1654 capture, Apple M5 | about 17 frames a second |

**The models** (held-out weeks 15-18 of the 2023 season, `nfl value-train`):

| | |
|---|---|
| Catch model AUC | 0.76, calibrated within 3 points |
| Yards gained when the quarterback threw to the model's option 1 / 2 / 3+ | 12.0 / 9.4 / 5.8 |
| Coverage: man vs zone right at the snap / after 2 s | 70% / 77% (always "zone": 57%) |

**Role assignment** (`nfl roles-check`, 2,572 Big Data Bowl pass plays with no linemen in
the data and 0.4 yd of noise, a hard version of film): quarterback picked out 71% from the
formation alone and 96% with the play's charting; route runners named receivers 92% / 98%;
coverage named coverage 94% / 95%.

**Field mapping:** on the sideline angle the fitted grid sits on the paint on 70-100% of
frames with 0.03-0.08 yd residual against the lines used. Two different plays gave the same
camera (about 49 yd up, 23 degrees of tilt), which is good evidence the fit is physical.

## What was hard

- **The field has no sensors.** Positions had to come from paint. The yard-line family is
  found by vanishing point, numbered so equal steps in number are equal steps on the field,
  and checked against the hash marks, whose dashes must land on whole yards.
- **The detector loses people.** Linemen in a pile, a corner pressed on his receiver, the
  quarterback behind the line. Three things made it survivable: tracking in yards with a
  Hungarian assignment instead of trusting the image tracker's identities; a player hidden
  behind another rides with him until they part; a player lost and found again gets his
  role and name back by where he reappears.
- **Telling who is who with no labels.** Side of the ball first, jersey colour second,
  charted hints (shotgun, hash) when the play is known, and corrections from what players
  do after the snap. Marked against 2,572 real snaps, not eyeballed.
- **Speed.** Profiling showed the slow part wasn't the model: OpenCV's area resizes of a 4K
  grab cost 40 ms a frame. A block-average-then-linear resize with the same geometry took
  the pipeline from 10 to 18 fps and moved the mapping by a median of 0.03 yd.
- **Being honest about the snap and the value.** Two early results were wrong (a coasting
  track made the target look like the best throw; the huddle breaking was taken for the
  snap). Both were caught by looking at frames. Since then every claim is checked by eye or
  against charting before it is written down.

## What it doesn't do yet

- **The throw grade is experimental.** The play-by-play says who was thrown to and how
  deep; that picks out the target on about a third of passes. The moment of the throw,
  worked back from depth and a flight time fitted to 3,351 real throws, is within 0.6 s of
  the truth only 69% of the time. Without tracking the ball it won't get better, so a grade
  is marked firm only when the target's rank holds half a second either side, and few are.
- **The end-zone angle** isn't mapped (the long lens defeats the camera model); the panel
  replays the sideline read in step with it instead.
- **Officials** are told apart from players by count and position, not by sight.
- **Values above 20 expected yards** on 4 of 22 passes are plausible (a receiver deep and
  open) but unverified.
- **Detector fine-tuning** on corrected boxes is the next big step; most remaining errors
  trace to the pretrained detector.

## Demo videos

`scripts/make_showcase.py` builds two cuts of six plays from the two games
(`outputs/showcase/`, never committed):

- `sideline.mp4`: the film with the grid, names and lines, the top-down view and the panel.
  It contains NFL film, so it is for showing in person, not for posting.
- `topdown.mp4`: the top-down view and the panel only, no film in it. This is the one that
  can go on a website.

## Data and licences

Film comes from NFL Pro at run time and never enters the repository. Models are trained
on the Big Data Bowl 2026 tracking data (Kaggle, competition terms) and labelled with
nflverse play-by-play and FTN charting (public). None of that is committed either.
