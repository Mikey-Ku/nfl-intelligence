# Project status

Last updated 2026-11-09. "Measured" means a number checked against charting, tracking data
or by eye on frames; "by eye" means no ground truth exists yet. [SHOWCASE.md](SHOWCASE.md)
says where each number comes from; [USAGE.md](USAGE.md#how-to-know-it-is-working) says how
to see each of these for yourself on a run.

## Capabilities

| Capability | Status | Measured | How to check |
|---|---|---|---|
| Player detection and tracking | Working, pretrained detector | 17-18 fps on 4K with full analysis; linemen in a pile still merge | `nfl run --shots`, panel's Tracking block |
| Team split by jersey colour | Working | | overlay colours |
| Field mapping, sideline angle | Working | 70-100% of frames, 0.03-0.08 yd residual | `--shots` shows the fitted grid on the paint |
| Field mapping, end-zone angle | Not mapped | long lens defeats the camera model | panel says "read from the sideline view" |
| Play identified from the screen | Working | every run and pass of 45 | `play: ...` line while running; `matched` in the play log |
| Anchoring to real yard lines | Working | 36 of 40 from the numbers, rest from the formation | `line of scrimmage ... settled by` line; `anchored_by` in the play log |
| Snap detection | Working | 0.0-0.2 s late on 4 plays by eye, then corrected | `snap` in the play log; `--topdown` video |
| Roles from the formation | Working | QB 39 of 42 on film; 71% / 96% on 2,572 BDB snaps | `nfl plays-check`, `nfl roles-check` |
| Pass value and plus/minus | Working | catch AUC 0.76; 22 of 24 passes valued | `nfl value-train` prints the held-out scores |
| Coverage read (man/zone, shell) | Working, rough | 70% at the snap, 77% after 2 s (always zone: 57%) | `nfl coverage-train` |
| Names from jersey numbers | Working on light jerseys | 3.4 of 22 a play, 36 of 36 right; dark jerseys hardly read | `numbers` in the play log |
| Names from looks | Working, by eye only | +0.8 a play at ~85% | `looks` in the play log; `game-table` `known_by` |
| Throw grade | Experimental | throw moment within 0.6 s only 69% of the time | marked firm only when the rank holds |
| Game table | Working | 621 player-plays over 42 plays on the test game | `nfl game-table` |
| Detector fine-tuning | Tooling ready, not run | | `nfl prelabel`, `nfl train` |
| Officials | Kept out of the eleven by count and position, not by sight | | |

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

## Where it stands, in one paragraph

Measured on two games, 45 plays, 17 minutes of film. Reading the screen, pinning the field
and reading the formation are solid. The detector is still the stock COCO model, which is
where most tracking errors come from. The number that matters most, how far each tracked
player is from where he really stood, has not been measured: the mapping's 0.03-0.08 yd
residual is the grid against the paint, not a player against the truth, and no hand-labelled
truth exists yet. So this is a working prototype with its central claim unverified.

## The bar for a first release

The brief's own success criteria, plus the detector:

1. **Truth on 20 plays.** `nfl truth` labels a play in about 20 minutes; `nfl eval` scores
   it. Target from the brief: about 1 yard mean error, at most 2 identity swaps a play.
2. **Detector fine-tuned.** 275 frames are pre-labelled in `data/labels/` with the
   correction notes in `data/LABELLING.txt`; none has been corrected yet. An afternoon of
   fixes, then `nfl train`. Most of the remaining tracking errors trace to this step.
3. **A third game**, not more of the same two: a night game, a team with dark home jerseys,
   a different stadium. Captured with `nfl capture`, run with `nfl run`, marked with
   `nfl plays-check` and the game table, looked at through `--shots`.

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
- **Officials** are kept out of the eleven by count and position, not recognised by sight.
- **Values above 20 expected yards** on 4 of 22 passes are plausible but unverified.
- Without the play-by-play file, or on the end-zone angle, the panel says what is missing
  instead of guessing.
