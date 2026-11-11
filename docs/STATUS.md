# Project status

Last updated 2026-11-11. "Measured" means a number checked against charting, tracking data
or by eye on frames; "by eye" means no ground truth exists yet. [SHOWCASE.md](SHOWCASE.md)
says where each number comes from; [USAGE.md](USAGE.md#how-to-know-it-is-working) says how
to see each of these for yourself on a run.

## Capabilities

| Capability | Status | Measured | How to check |
|---|---|---|---|
| Player detection and tracking | Working; fine-tuned detector available | 17-18 fps on 4K with full analysis; linemen in a pile still merge | `nfl run --shots`, panel's Tracking block |
| Team split by jersey colour | Working | | overlay colours |
| Field mapping, sideline angle | Working, day and night | 70-100% of frames, 0.03-0.08 yd residual; 0.03-0.12 on the night game | `--shots` shows the fitted grid on the paint |
| Field mapping, end-zone angle | Not mapped | long lens defeats the camera model | panel says "read from the sideline view" |
| Play identified from the screen | Working, both NFL Pro layouts | every run and pass on all three games (49 and 29) | `play: ...` line while running; `matched` in the play log |
| Anchoring to real yard lines | Working | 36 of 40 from the numbers, rest from the formation | `line of scrimmage ... settled by` line; `anchored_by` in the play log |
| Snap detection | Working | 0.0-0.2 s late on 4 plays by eye, then corrected | `snap` in the play log; `--topdown` video |
| Roles from the formation | Working | QB 39 of 42 on film; 71% / 96% on 2,572 BDB snaps | `nfl plays-check`, `nfl roles-check` |
| Pass value and plus/minus | Working | catch AUC 0.76; 22 of 24 passes valued | `nfl value-train` prints the held-out scores |
| Coverage read (man/zone, shell) | Working, rough | 70% at the snap, 77% after 2 s (always zone: 57%) | `nfl coverage-train` |
| Names from jersey numbers | Working on light jerseys | 3.4 of 22 a play, 36 of 36 right; dark jerseys hardly read | `numbers` in the play log |
| Names from looks | Working, by eye only | +0.8 a play at ~85% | `looks` in the play log; `game-table` `known_by` |
| Throw grade | Experimental | throw moment within 0.6 s only 69% of the time | marked firm only when the rank holds |
| Game table | Working | 621 player-plays over 42 plays on the test game | `nfl game-table` |
| Detector fine-tuning | Done once, on 275 corrected frames | players found 80.5% → 91.1% on 55 held-out frames; see below | `nfl prelabel`, `nfl boxes`, `nfl train` |
| Officials | Dropped by sight with the fine-tuned detector; by count and position with the stock one | boxes on an official 53 → 4 on the held-out frames | |

## Role assignment in detail

`nfl roles-check` marks roles against Big Data Bowl tracking, which has no linemen at all (a
hard version of linemen lost in a pile), with 0.4 yd of noise added:

| 2,572 pass plays, 2023 weeks 1-3 | formation alone | with the play's charting |
|---|---|---|
| quarterback picked out | 71% | 96% |
| route runners named receivers | 92% | 98% |
| coverage men named coverage | 94% | 95% |

The charting (shotgun or under centre, which hash the ball is on) comes with the matched
play. Without it, and with no linemen to show where the ball is, the quarterback is often
confused with the back beside him; on film the linemen that are found close most of that gap.

## The fine-tuned detector

`nfl train` on 275 hand-corrected frames (5,565 player boxes, 294 referee boxes) from the
three clips, 220 to train and 55 held out, on 2026-11-10. The weights are in
`runs/detect/nfl/weights/best.pt` and are passed with `--weights`; the default is still the
stock `yolo11s.pt`, which is what a fresh clone has.

| 55 held-out frames | stock | fine-tuned |
|---|---|---|
| players found | 80.5% | 91.1% |
| kept boxes that are players | 75.7% | 88.8% |
| boxes on an official | 53 | 4 |

| the three clips, same code, against the charting | stock | fine-tuned |
|---|---|---|
| quarterback named | 39 of 42 | 40 of 42 |
| passes given a value | 22 of 24 | 23 of 24 |
| throw found | 8 of 24 | 16 of 24 |
| backs in the backfield, exact | 15 of 40 | 30 of 41 |
| defenders in the box, exact | 16 of 42 | 21 of 42 |
| pass rushers, exact (within one) | 6 (21) of 24 | 11 (23) of 24 |
| players named by number | 213 | 228 |
| track identities used (fewer is steadier) | 3,691 | 2,663 |

The held-out frames and the clips are from the same two games the detector trained on, so a
new game will show less. The third game is the test of that.

## The third game

Jaguars at Ravens, Week 9: a Thursday night game in Baltimore, the home team in purple, ten
minutes of film from the first drive (`data/clips/live-06.mp4`, captured 2026-11-10). None
of it was in the detector's training frames or the number reader's crops.

Two things broke on the first run, and both are fixed:

- NFL Pro had redesigned its player since September. The header is now one line on the
  video's top edge ("2ND & 6 BAL 17") that no longer says who has the ball, and the
  on-field list lost its title. No play could be identified. `playinfo` and `personnel` now
  read both layouts; the offense comes from the list or from the play-by-play row.
- Under the lights the yard lines are light green, as colourful as the turf, and the line
  finder only took white. The mapping locked on 3 plays. `register` now tries white lines
  first and light-green ones when that gives no lock, so the three older clips score what
  they did before.

After the fixes, with the fine-tuned detector, against the charting:

| | games it was built on | third game |
|---|---|---|
| runs and passes worked out, of those identified | 43 of 49 | 24 of 29 |
| quarterback named | 40 of 42 | 22 of 24 |
| passes given a value | 23 of 24 | 11 of 12 |
| throw found | 16 of 24 | 8 of 12 |
| backs in the backfield, exact (within one) | 30 (40) of 41 | 11 (23) of 24 |
| defenders in the box, exact (within one) | 21 (36) of 42 | 8 (18) of 24 |
| pass rushers, exact (within one) | 11 (23) of 24 | 4 (9) of 12 |
| players named by number, a play | 5.3 | 0.7 |

What carries over: reading the screen, the mapping, the snap, the quarterback, the value.
What does not yet: exact counts in the box and of rushers, and names. The number reader has
seen almost no purple or teal-trimmed shirts, and the look memory is trained per game and
was not trained on this one. Of the five plays not worked out, two are snaps from the
1-yard line and two are cut off by the ends of the clip.

## Where it stands, in one paragraph

Measured on three games, 78 runs and passes, 27 minutes of film. Reading the screen, pinning the field
and reading the formation are solid, and they held on a game the models had never seen once
two things that game exposed were fixed. The number that matters most, how far each tracked
player is from where he really stood, has not been measured: the mapping's 0.03-0.08 yd
residual is the grid against the paint, not a player against the truth, and no hand-labelled
truth exists yet. So this is a working prototype with its central claim unverified.

## The bar for a first release

The brief's own success criteria, plus the detector:

1. **Truth on 20 plays.** `nfl truth` labels a play in about 20 minutes; `nfl eval` scores
   it. Target from the brief: about 1 yard mean error, at most 2 identity swaps a play.
2. **Detector fine-tuned.** Done 2026-11-10 on 275 corrected frames; the numbers are
   above.
3. **A third game.** Done 2026-11-11: a night game, dark home jerseys, another stadium.
   What broke and the numbers are above.

With those three the README gets a "within X yd on 20 labelled plays" tile and the word
"prototype" can go.

## After that

- Crop-level "readable" labels for dark jerseys (a few hundred crops) so the number reader
  learns them; today it names almost only light shirts.
- Ground truth for the looks, so the 85% stops being an eye estimate.
- Roster fallback from nflverse weekly rosters when NFL Pro's on-field list is not on
  screen.
- The end-zone angle, if ever: the long lens defeats the camera model and the sideline
  read already covers it.

What it will not become: chip-grade tracking (10 Hz at a tenth of a yard) or a ball
tracker. The throw grade stays rough for that reason.

## Known limits

- **Linemen in a pile** come out as a few boxes, so heavy formations read roughly (a
  goal-line run charted with seven in the box read eight).
- **Dark jerseys** are hardly read by the number reader yet; the labelled training set was
  almost all light shirts.
- **The end-zone angle** is not mapped; the sideline read is replayed over it instead.
- **Run plays** get roles and a formation but no value: the models score passes.
- **The throw grade** cannot get better than the throw moment, and without tracking the ball
  that is within 0.6 s only 69% of the time.
- **Officials** are recognised by sight only with the fine-tuned detector; the stock one
  keeps them out of the eleven by count and position.
- **Values above 20 expected yards** on 4 of 22 passes are plausible but unverified.
- Without the play-by-play file, or on the end-zone angle, the panel says what is missing
  instead of guessing.
