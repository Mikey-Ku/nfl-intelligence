# Project status

Last updated 2026-11-11. "Measured" means a number checked against charting, tracking data
or by eye on frames; "by eye" means no ground truth exists yet. [SHOWCASE.md](SHOWCASE.md)
says where each number comes from; [USAGE.md](USAGE.md#how-to-know-it-is-working) says how
to see each of these for yourself on a run.

## Capabilities

| Capability | Status | Measured | How to check |
|---|---|---|---|
| Player detection and tracking | Working; fine-tuned detector available | 17-18 fps on 4K with full analysis; 0.72-0.98 yd and 84-94% found on five labelled plays from two games; linemen in a pile still merge and trade numbers | `nfl run --shots`, panel's Tracking block |
| Team split by jersey colour | Working | | overlay colours |
| Field mapping, sideline angle | Working, day and night, across the whole width | 70-100% of frames; the far sideline lands within 0.1 yd of where it is and the near one within 0.4 (they were 0.7-3.0 yd inside); the hash marks and the sidelines are each left 0.2-0.35 yd off by the one camera that fits both | `--shots` shows the fitted grid on the paint |
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
| Officials and the bench | Officials dropped by sight with the fine-tuned detector, by count and position with the stock one, and given no box or team either way; the bench kept out by the border measured in yards | boxes on an official 53 → 4 on the held-out frames; a side is never more than eleven, on the panel or in boxes: after the snap a man nobody has named is boxed only while his side is short of eleven, so the official on the sideline is not; the panel's headline counts the two sides only | |

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

After the fixes, with the fine-tuned detector and the steadier identities described below,
against the charting:

| | games it was built on | third game |
|---|---|---|
| runs and passes worked out, of those identified | 43 of 49 | 24 of 29 |
| quarterback named | 42 of 42 | 24 of 24 |
| passes given a value | 24 of 24 | 11 of 12 |
| throw found | 13 of 24 | 5 of 12 |
| backs in the backfield, exact (within one) | 36 (42) of 42 | 17 (23) of 24 |
| defenders in the box, exact (within one) | 23 (41) of 42 | 8 (21) of 24 |
| pass rushers, exact (within one) | 16 (24) of 24 | 5 (9) of 12 |
| players named by number, a play | 5.7 | 0.7 |

What carries over: reading the screen, the mapping, the snap, the quarterback, the value.
What does not yet: exact counts in the box and of rushers, and names. The number reader has
seen almost no purple or teal-trimmed shirts, and the look memory is trained per game and
was not trained on this one. (These are from `outputs/v12`, with the field mapped at its
true width; before that, on `outputs/v8`, the throw was found on 14 of 24 and 8 of 12. The
eight passes that changed, six lost and two gained, each come down to a receiver's track
being lost or named differently in the new run: identity, not the mapping.) Of the five plays not worked out, two are snaps from the
1-yard line, where half the players are lost in the pile, and two are cut off by the ends
of the clip.

## Hand-labelled truth

Five plays so far, four from the Panthers at Browns game the detector was trained on and one
from the Jaguars at Ravens night game it never saw: 14 frames each, from a second before the
snap to six after, every player's feet clicked with `nfl truth` and scored with `nfl eval`
against the same run (`outputs/v12`).

| play | position error | around the steady offset | players found | identity changes |
|---|---|---|---|---|
| Panthers 2nd and 15, a pass (`live-02` shot 3) | 1.02 yd | 0.38 yd | 91.7% | 24 |
| Panthers 1st and 10, a run (`live-02` shot 5) | 0.93 yd | 0.40 yd | 83.8% | 42 |
| Panthers 1st and 10, a pass (`live-04` shot 22) | 0.89 yd | 0.35 yd | 91.8% | 26 |
| Browns 3rd and 2, a run (`live-04` shot 48) | 0.78 yd | 0.34 yd | 94.1% | 24 |
| Ravens 1st and 10, a run, the unseen game (`live-06` shot 22) | 0.84 yd | 0.36 yd | 87.3% | 22 |
| all five | 0.89 yd | 0.37 yd | 89.7% of 1,531 | 28 a play |
| the brief's target | about 1 yd | | | at most 2 |

Position meets the target on all five, and the play from the unseen game sits in the middle
of the others. What this measures is the tracked player against the same player clicked by
hand, both put through the run's own field mapping; an error in the mapping itself does not
show in it. That is how the mapping came to be a tenth too narrow across the field without
this table saying so (see Known limits). With the field at its true width the same errors
in the picture are more yards: these five read 0.72 to 0.98 before, 0.82 together. Most of
the error is one steady offset of 0.71 to 0.92 yd across the field:
the run places a player at the bottom of his box and the labeller clicks a little higher, the
same on every player and every frame. Of 1,377 players the run placed on these frames, 3 were
not players. The stock detector finds 58.9% of the players on the first play.

Identity is the miss. The first play scored 61 changes at first, and looking into it found
the field tracker, whose number the names and roles hang on, less steady than the image
tracker under it: a second, unconfirmed box on a player started a ghost beside him, and the
two traded his detections. A detection now stays with the player who had its image track,
an unconfirmed box starts a player only well clear of everyone, and a second track started
on a man takes over his number. That took the first play from 61 to 21 and lifted the
charting checks on all four clips (backs in the backfield exact 30 to 36 of 42, the box
within one 36 to 41 of 42). Corners, safeties and wide receivers now keep one number through
a pass play. The second play, labelled afterwards, is a run where everyone closes on the
ball, and there the field tracker does no better than the image tracker under it: 43. Some
of the count is the labels, since at half-second steps two linemen crossing are hard to
number by hand, but most of it is the pile.

The second play also showed a play anchored the wrong way round: the Browns marked as the
offense on a Panthers snap from their own 40. The painted arrows were not read there and a
painted 50 was not being used, so the direction was left to a guess from the formation. A 50
and one other number now settle it, and once the paint has settled which way a team is going
in a quarter a later play is not anchored the other way. Three of 66 plays had been the
wrong way round; now every play found in the play-by-play goes the way of its drive. (One
play the play-by-play does not have still disagrees with its neighbours.)

Identity misses the target on all five, passes and runs alike (22 to 42 changes against a
target of 2), so the first play, the one the fixes were worked out on, was the low end and not the norm. More labelled
plays would measure that more finely; they would not change it.

Five plays are a small sample, four of them from one game and one from the unseen game.

## Where it stands, in one paragraph

Measured on three games, 78 runs and passes, 27 minutes of film. Reading the screen, pinning the field
and reading the formation are solid, and they held on a game the models had never seen once
two things that game exposed were fixed. The number that matters most, how far each tracked
player is from the same player clicked by hand, has been measured on five plays from two games: 0.7 to
1.0 yard on every one, most of it a steady offset, and the same on the game no model had
seen. The field mapping under both was a tenth too narrow until 2026-11-12; it now puts
the sidelines within 0.4 yd of where they are. Identities do not hold: 22 to 42 changes a play against a target of 2. So this is a
working prototype whose position claim holds on a small sample and whose identity claim
does not hold yet.

## The bar for a first release: closed 2026-11-11

The brief's own success criteria, plus the detector:

1. **Truth on hand-labelled plays.** The bar was 20 plays; it was closed at five. Target
   from the brief: about 1 yard mean error, at most 2 identity swaps a play. The first is
   met on all five, with too little spread between plays for fifteen more to move it. The
   second is met on none, and that is tracker work, not labelling.
2. **Detector fine-tuned.** Done 2026-11-10 on 275 corrected frames; the numbers are
   above.
3. **A third game.** Done 2026-11-11: a night game, dark home jerseys, another stadium.
   What broke and the numbers are above.

The README carries the result as tiles: 0.89 yd on five labelled plays, and the identity
changes beside it. The word "prototype" stays, because the identity target is not met.
Measuring is finished; what would lift the project from here is in the next section.

## After that

- Identity through the pile: a player keeping one number from the snap to the whistle.
  Tuning the field tracker's gates and smoothing moved nothing, so this needs a different
  approach (the jersey numbers and looks used inside a play, not only between plays). The
  five labelled plays are the test for it.
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

- **The mapping was a tenth too narrow, and is now within 0.4 yd.** Until 2026-11-12 the
  camera was fitted to the yard lines and the two rows of hash marks alone, and the far
  sideline's inner edge landed 0.7 yd (`live-02`), 1.1 yd (`live-06`) and 3.0 yd (`live-04`)
  inside where it is. Two causes. The rows followed are the hash marks' centres, and they
  were taken to be 6.17 yd apart; that is the gap between the marks' inner ends, and the
  centres of two-foot marks painted outward from there are 6.83 yd apart. And two rows in
  the middle of the field do not hold the width of a field 53 yards across: with the
  spacing corrected, the sideline still landed anywhere from 0.8 yd outside to 2.1 yd
  inside. So the border's inner edge is now a constraint on the camera, the camera's
  position is solved again the first time a sideline shows, and that position is carried to
  later shots (it is one camera, bolted down). The far sideline now lands within 0.1 yd on
  all three clips and the near one within 0.4. What is left: one level camera over a flat
  field cannot put the hash marks and the sidelines both exactly right, and each is left
  0.2 to 0.35 yd off. A crowned field would do that; it has not been looked into.
- **The box widths were set on the narrow field.** Who counts as in the box (8.3 yd either
  side of the ball on the line, 5.0 behind it) is the old 7.5 and 4.5 scaled by 6.83/6.17.
  That one factor brought the box count back to what it was (within one on 62 of 66, from
  60); it was not searched for.
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
