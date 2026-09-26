# NFL Intelligence

> Turn All-22 game film into top-down tracking data for every run and pass play, snap to whistle, using a computer-vision pipeline I build myself.

**Domain:** ml-ai
**Created:** 2026-09-25
**Slug:** `nfl-intelligence`

## The problem
The NFL's tracking data (Next Gen Stats) comes from RFID chips in shoulder pads, and none of it is public for the current season. The only way to get player positions for this year's plays is to pull them out of the film. I want to learn computer vision end to end (detection, tracking, camera calibration) by building that pipeline, then use its output to study formations, spacing, and how plays develop.

## Who it's for
Just me. A personal learning project; nothing is redistributed.

## MVP scope
- **Field registration:** map every video frame to field coordinates in yards, for both the sideline and end-zone angles, following camera pans and zooms.
- **Players:** detect all 22 players with a fine-tuned pretrained detector and split them into offense and defense by jersey color.
- **Tracking:** keep a stable ID for each player from line-set to whistle, assign roles (QB, RB, WR, TE, OL, DL, LB, DB) from the pre-snap alignment, and label the offensive formation at the snap.
- **Live overlay:** capture the NFL Pro player window straight off the screen and draw the tracking on top of it in a transparent window, with a top-down minimap in the corner, so any play can be watched with the pipeline running on it.
- **Output:** for each play, write a tracking file (frame, player, team, role, x, y, speed, direction), render an animated top-down replay, and attach the play-by-play summary (down, distance, play type, result).
- **Angle comparison:** hand-label ground truth for a set of test plays, run the sideline and end-zone angles separately, and measure which is more accurate.

## Out of scope (v2 ideas)
- Fusing both angles into one combined track (the MVP compares them first).
- Live within-play value line (win probability / expected points per frame) and credit assignment for the swings.
- Coverage recognition (man/zone, Cover 1/2/3/4/6), QB decision quality, play-call tendencies against defensive looks.
- Reacting to pre-snap movement (shifts, motion, safety rotation), the observable stand-in for audibles.
- Ball tracking (the MVP uses the play-by-play to know who carried or was targeted).
- Naming players via jersey-number reading plus rosters.
- Punts, kickoffs, field goals, and other special teams.
- Broadcast footage, splitting full games into plays, and real-time overlays.

## Success criteria
- On 20 hand-labeled test plays, tracked players are within about 1 yard of their true positions on average, for at least one angle.
- No more than 2 identity swaps per play, line-set to whistle.
- One command turns a play clip into a tracking file and top-down replay in under 5 minutes on my M5, and a short write-up says which angle is more accurate and why.

## Constraints
- Runs locally on an M5 MacBook with 16 GB RAM; Python with `uv`.
- Input is 2026-season All-22 film from NFL Pro, one play per clip, sideline and end zone, mostly time-synced. It reaches the pipeline either as screen capture of the player or as clip files; I supply the footage.
- Must keep up with playback: detection under ~30 ms per frame on the M5 GPU (measured 11 ms with YOLO11n at 1280).
- Footage and anything derived from it stay local: never committed to the repo, never shared.
- Vision models start pretrained and get fine-tuned; nothing is trained from scratch.
- No public tracking data exists for 2026, so ground truth has to be hand-labeled.
- Run and pass plays only.

## Inspirations / prior art
- **NFL Next Gen Stats (AWS):** the tracking comes from RFID sensors, not video. This project gets the same kind of data from film alone.
- **NFL Big Data Bowl:** the tracking-data schema to mirror. It can't serve as the answer key because its plays are from earlier seasons.
- **Roboflow `sports` (soccer):** detection, team clustering by jersey color, and pitch-keypoint homography on broadcast video. This project needs American-football markings (yard lines, numbers, hash marks) and All-22 angles.
- **SoccerNet Game State Reconstruction:** the academic benchmark for turning video into a top-down minimap, and a source of evaluation ideas.
- **Yurko et al., "Going Deep":** within-play valuation from tracking data; the v2 target once tracking works.

## Open questions
- Clip specs: resolution, frame rate, format, and how many clips per week.
- How far out of sync the two angles get, and whether aligning them on the snap is enough.
- Labeling budget: how many plays and keyframes to hand-label, and what the labeling tool looks like.
- Whether the end-zone angle's depth compression makes downfield positions too inaccurate.
- Formation vocabulary: which labels to use (shotgun/pistol/under center, personnel groupings, trips/bunch/empty).
- Whether to allow a paid cloud GPU for fine-tuning, or keep everything local.
- Repo and folder spelling: the folder is `NFL-Intellegence`; the slug above uses `nfl-intelligence`.

## Notes
Started as a broader idea: learn football strategy (formations, spacing, audibles, whether players make the right decisions) and show live, within-play probability swings like the Next Gen Stats broadcast graphics. We looked at building the reasoning model first on Big Data Bowl tracking data, but chose vision first because the vision tech is what I most want to learn. Audibles aren't directly observable on film, so they became "reaction to pre-snap movement" (v2). Everything in the out-of-scope list builds on the tracking output from this MVP.
