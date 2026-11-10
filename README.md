<h1 align="center">NFL Intelligence</h1>

<p align="center">
  <strong>Live player tracking, formations and a pass value, read straight off NFL game film.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/runs%20on-Apple%20silicon-000000?logo=apple&logoColor=white" alt="Apple silicon">
  <img src="https://img.shields.io/badge/tests-158%20passing-2ea44f" alt="158 tests">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
</p>

<p align="center">
  <img src="docs/media/play.gif" width="880" alt="A play unfolding top-down with the live panel: names on the players, the best throw and its expected yards">
</p>

<p align="center"><sub>A pass play as the pipeline sees it: every player placed on the field in yards, named from his jersey, with the best throw and its expected yards updating live. No film in this picture, only what was read from it.</sub></p>

---

The NFL's tracking data comes from chips in shoulder pads and none of it is public. NFL
Intelligence gets the same kind of picture from the film alone: point it at the All-22 player
and it finds the 22 players, maps the painted field to real yards, reads the play off the
screen, names the players from their jersey numbers, works out the formation and the snap,
and scores every receiver, every frame, with models trained on the league's own tracking data.
It runs live on a MacBook.

## Highlights

<table>
  <tr>
    <td align="center"><h3>17-18 fps</h3>4K capture with full analysis, Apple M5</td>
    <td align="center"><h3>45 / 45</h3>plays identified from the screen and matched to the play-by-play</td>
    <td align="center"><h3>36 / 40</h3>plays anchored to real yard lines by reading the painted numbers</td>
  </tr>
  <tr>
    <td align="center"><h3>39 / 42</h3>quarterbacks picked out at the snap, checked against official charting</td>
    <td align="center"><h3>36 / 36</h3>jersey numbers read right on held-out plays, checked by eye</td>
    <td align="center"><h3>AUC 0.76</h3>catch model on held-out weeks, calibrated within 3 points</td>
  </tr>
</table>

Every number is measured; [docs/SHOWCASE.md](docs/SHOWCASE.md) says where each comes from.

## What you see

<table>
  <tr>
    <td width="62%"><img src="docs/media/presnap.jpg" alt="The formation read before the snap"></td>
    <td><img src="docs/media/panel.jpg" alt="The stats panel mid-play"></td>
  </tr>
  <tr>
    <td><sub><b>Before the snap.</b> The play is read off the screen and matched to the play-by-play, the field is anchored from the painted numbers, and the formation and personnel are already on the panel.</sub></td>
    <td><sub><b>Mid-play.</b> Expected yards of the best throw, the plus/minus against a typical play from the same stretch of field, who is open, the nearest rusher, and tracking health.</sub></td>
  </tr>
</table>

<p align="center">
  <img src="docs/media/names.jpg" width="880" alt="Players named from their jersey numbers on the top-down view">
</p>
<p align="center"><sub><b>Who is who.</b> Names come from a jersey-number reader trained on the film itself, constrained to the 22 on the field. A player named once is known again on later plays by his look.</sub></p>

**The game table.** Every run also lands in one table, a row per player per play, so a
question like "how deep did the safeties line up" or "who was the best option most often"
is one filter away. From one half of Panthers at Browns:

| team | name | # | role | plays | named by number / look | depth at snap (yd) | top speed (yd/s) | moments as best throw |
|---|---|---|---|---|---|---|---|---|
| CAR | Tremble | 82 | REC | 6 | 2 / 0 | -1.7 | 8.0 | 72 |
| CAR | Hubbard | 30 | REC | 8 | 1 / 0 | -5.4 | 8.4 | 14 |
| CAR | Moehrig | 7 | DEF | 7 | 2 / 5 | 5.5 | 10.2 | |
| CLE | Jeudy | 3 | REC | 4 | 2 / 2 | -1.7 | 8.7 | 26 |
| CLE | Hickman | 33 | DEF | 4 | 4 / 0 | 11.9 | 9.8 | |
| CLE | Verse | 8 | DL | 4 | 4 / 0 | 0.8 | 5.8 | |

## How it works

```
screen / clip ─► YOLO ─► BoT-SORT ─► jersey colours ─► field mapping ─► yards ─► snap ─► roles ─► value ─► panel
      │                                                                                        ▲
      ├─► OCR: play header ─► play-by-play row ─► real yard line, down & distance, result ─────┘
      └─► OCR: on-field roster ─► jersey-number reader + player looks ─► names on the players
```

1. **Find and follow the players.** A pretrained YOLO detector and BoT-SORT keep an identity
   on each man through the camera's pans; jersey colour splits the teams.
2. **Map the field from the paint.** The yard lines are found by their shared vanishing
   point, checked against the hash marks, and a pan/tilt/zoom camera is fitted and followed
   frame to frame. Every foot on screen becomes a point in yards.
3. **Read the screen.** On-device text recognition reads the play header and the on-field
   roster. The header finds the play in the public play-by-play; the big painted numbers,
   whose arrows point at the nearer goal line, pin the yards to the real field.
4. **Find the snap and the roles.** A still formation starting to move is the snap. Each
   player gets a role from where he stands, corrected by what he does next.
5. **Name the players.** A number reader trained on the film reads the torsos; a look memory
   recognises a named player on later plays by his build, sleeves, socks and gloves.
6. **Score every receiver, every frame.** Catch and yards models trained on Big Data Bowl
   tracking give the best throw and its expected yards; the plus/minus compares it with a
   typical play at the same moment from the same stretch of field.

Everything is a pretrained model fine-tuned on our own data, and every stage can be checked
against public charting or real tracking. [docs/REPO_GUIDE.md](docs/REPO_GUIDE.md) walks
through the code.

## Quick start

macOS on Apple silicon, Python 3.12, [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run nfl check                                              # weights, GPU, screen capture
uv run nfl data --season 2026                                 # public play-by-play, once
uv run nfl overlay --window "Chrome" --out outputs/live.csv   # live on a playing video
uv run nfl run data/clips/play.mp4 --topdown                  # or on a saved clip
```

[docs/USAGE.md](docs/USAGE.md) has every command and flag, what each run writes, how to train
the models, and how to tell a run is working.

## Documentation

| | |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | Running it: commands, flags, outputs, training, how to know it is working |
| [docs/STATUS.md](docs/STATUS.md) | Where each capability stands, with its measured number |
| [docs/SHOWCASE.md](docs/SHOWCASE.md) | What it achieves and where every number comes from |
| [docs/REPO_GUIDE.md](docs/REPO_GUIDE.md) | Every file and how a play travels through them |
| [docs/PRIOR_ART.md](docs/PRIOR_ART.md) | What exists elsewhere and what was learned from it |
| [docs/BRIEF.md](docs/BRIEF.md) | The original scope |

## Built with

[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) · BoT-SORT · PyTorch on
Apple MPS · OpenCV · Apple Vision (on-device OCR) · PySide6 · Polars · scikit-learn ·
[NFL Big Data Bowl 2026](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction) tracking data ·
[nflverse](https://github.com/nflverse) play-by-play and FTN charting

## Licence

MIT for the code. Film, tracking data and anything derived from them are not in the
repository and carry their own terms. The pictures above are renders of the pipeline's own
output; no footage appears in them.
