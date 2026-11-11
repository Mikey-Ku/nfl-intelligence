"""Hand-labelled truth for a play: click where each player stands, every half second.

The brief's success criteria are "within about a yard on hand-labelled plays, no more than
two identity swaps a play". `nfl eval` scores a run against truth in the same CSV schema;
this is the tool that makes the truth.

    uv run nfl truth data/clips/live-02.mp4 outputs/live-02.csv --shot 5 --every 0.5

It steps through the run's frames of that shot, half a second apart, and shows each one.
Click the feet of every player on the field (both teams; not officials), `n` for the next
frame. Each click is turned into yards through the run's own field mapping for that frame,
fitted from the run's players (their pixel feet and their yards), so the truth is in the
same coordinates the run is scored in. That means the score measures detection and
tracking, not the mapping; the mapping is measured separately against the paint.

Identities: on the first frame clicks are numbered in order. On later frames the whole
frame is matched at once to the frame before (each man to where he was heading, from his
last two positions, within `--reach` yards), so the order of clicking does not matter and
two men side by side do not trade numbers. On 400 Big Data Bowl plays sampled this way, with
0.3 yd of clicking error, 2.5% of numbers come out wrong; one click at a time gave 21%. A man nobody matches gets a number never used in
the play. The number is drawn by the click; a wrong one is fixed with the cursor on it and
`i` (type the number in the window, Enter; Esc leaves it). If another click on the frame
has that number the two trade. A number typed by hand, and every number on a frame once it
is left, stays put.

With `--shot` and the run's play log beside its CSV, labelling starts a second before the
snap and stops six seconds after it: the play itself, about 15 frames. `--whole-shot`, or
`--start` / `--stop`, label more.

Keys: `n` next frame, `p` previous, `u` undo the last click, `x` delete the click under the
cursor, `i` renumber the click under the cursor (else the last), `b` show or hide the run's
boxes (hidden by default so they don't lead the eye), `q` quit. The CSV goes to
`data/truth/<clip>.shot<N>.csv` unless `--out` says otherwise. It is written on every move
between frames and on any exit, and a play that already has a file is picked up at its first
unlabelled frame.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .field import FieldCalibration
from .schema import FrameResult, TrackedPlayer, read_csv, write_csv

WINDOW = "nfl truth"
MIN_POINTS = 4
BEFORE_SNAP, AFTER_SNAP = 1.0, 6.0          # seconds of a play labelled by default
GRAB = 60.0                                 # screen pixels within which the cursor is "on" a click
CARRY = 0.5                                 # how much of a man's last step he is expected to repeat


@dataclass
class Click:
    px: tuple[float, float]
    yd: tuple[float, float] | None
    pid: int
    fixed: bool = False                     # typed by hand, or on a frame already left: not renumbered


@dataclass
class TruthFrame:
    frame: int
    t: float
    shot: int
    anchored: bool
    clicks: list[Click] = field(default_factory=list)


def frame_mapping(res: FrameResult) -> FieldCalibration | None:
    """The run's pixel-to-yards mapping on one frame, refitted from its players."""
    px = [p.foot_px for p in res.players if p.field_yd is not None]
    yd = [p.field_yd for p in res.players if p.field_yd is not None]
    if len(px) < MIN_POINTS:
        return None
    try:
        return FieldCalibration.from_correspondences(np.array(px, dtype=np.float64), np.array(yd, dtype=np.float64))
    except Exception:
        return None


def mappings_for(run: list[FrameResult]) -> dict[int, FieldCalibration | None]:
    """A mapping per frame; a frame with too few mapped players borrows the nearest frame's in the same shot."""
    fitted = {r.frame: frame_mapping(r) for r in run}
    out: dict[int, FieldCalibration | None] = {}
    for r in run:
        m = fitted[r.frame]
        if m is None:
            near = [o for o in run if o.shot == r.shot and fitted[o.frame] is not None]
            if near:
                m = fitted[min(near, key=lambda o: abs(o.frame - r.frame)).frame]
        out[r.frame] = m
    return out


def sample_frames(run: list[FrameResult], every: float, shot: int | None = None,
                  start: float | None = None, stop: float | None = None) -> list[FrameResult]:
    """The run's frames `every` seconds apart, in one shot or a time window."""
    picked: list[FrameResult] = []
    last = None
    for r in run:
        if shot is not None and r.shot != shot:
            continue
        if start is not None and r.t < start:
            continue
        if stop is not None and r.t > stop:
            break
        if last is None or r.t - last >= every - 1e-6:
            picked.append(r)
            last = r.t
    return picked


def number_frame(clicks: list[Click], previous: list[Click], before: list[Click], reach: float, floor: int = 0) -> None:
    """Give this frame's clicks their numbers, all at once.

    Each man on the previous frame is expected where he was heading (his last position
    plus half his last step, when he was also on the frame `before`: players slow and turn,
    and half did best on real tracking). Clicks and expected
    positions are paired so the total distance is least, and a pair further apart than
    `reach` yards is no pair. A click left over gets a new number above `floor` and above
    everything on these frames, in the order clicked. Clicks marked `fixed` keep theirs.
    """
    from scipy.optimize import linear_sum_assignment

    held = {c.pid for c in clicks if c.fixed}
    was = {c.pid: c.yd for c in before if c.yd is not None}
    expected = {}
    for c in previous:
        if c.yd is None or c.pid in held:
            continue
        step = (c.yd[0] - was[c.pid][0], c.yd[1] - was[c.pid][1]) if c.pid in was else (0.0, 0.0)
        expected[c.pid] = (c.yd[0] + CARRY * step[0], c.yd[1] + CARRY * step[1])
    free = [c for c in clicks if not c.fixed]
    placed = [c for c in free if c.yd is not None]
    given: dict[int, int] = {}
    if placed and expected:
        pids = list(expected)
        cost = np.array([[np.hypot(c.yd[0] - expected[q][0], c.yd[1] - expected[q][1]) for q in pids] for c in placed])
        rows, cols = linear_sum_assignment(np.where(cost <= reach, cost, 1e6))
        given = {id(placed[r]): pids[k] for r, k in zip(rows, cols) if cost[r, k] <= reach}
    top = max([floor, *held, *(c.pid for c in previous), *given.values()], default=0)
    for c in free:
        if id(c) in given:
            c.pid = given[id(c)]
        else:
            top += 1
            c.pid = top


def renumber(clicks: list[Click], index: int, number: int) -> None:
    """Call click `index` number `number`. Whoever on the frame had it takes the old one; both stay put after."""
    old = clicks[index].pid
    for j, c in enumerate(clicks):
        if j != index and c.pid == number:
            c.pid, c.fixed = old, True
    clicks[index].pid, clicks[index].fixed = number, True


def nearest(clicks: list[Click], x: float, y: float, scale: float, within: float = GRAB) -> int | None:
    """The click under the cursor: index of the one nearest screen point (x, y), if within `within` pixels."""
    best = None
    for j, c in enumerate(clicks):
        d = float(np.hypot(c.px[0] * scale - x, c.px[1] * scale - y))
        if d <= within and (best is None or d < best[0]):
            best = (d, j)
    return None if best is None else best[1]


def play_window(csv_path: str | Path, shot: int | None) -> tuple[float, float] | None:
    """(start, stop) seconds for a shot's play, from the play log `nfl run` writes beside its
    CSV: a second before the snap to six after. None when there is no log or no snap in it."""
    log = Path(csv_path).with_suffix(".plays.json")
    if shot is None or not log.exists():
        return None
    for play in json.loads(log.read_text()):
        if play.get("shot") == shot and play.get("snap") is not None:
            return float(play["snap"]) - BEFORE_SNAP, float(play["snap"]) + AFTER_SNAP
    return None


def to_results(frames: list[TruthFrame]) -> list[FrameResult]:
    """Truth frames as FrameResults in the tracking schema (conf 1, no boxes, no teams)."""
    out = []
    for f in frames:
        players = [TrackedPlayer(track_id=c.pid, team=None, box=(c.px[0], c.px[1], c.px[0], c.px[1]), conf=1.0,
                                 foot_px=(float(c.px[0]), float(c.px[1])),
                                 field_yd=None if c.yd is None else (float(c.yd[0]), float(c.yd[1])))
                   for c in f.clicks]
        out.append(FrameResult(frame=f.frame, t=f.t, size=(0, 0), players=players, shot=f.shot, anchored=f.anchored))
    return out


def resume(truth: list[TruthFrame], saved: list[FrameResult]) -> int:
    """Put an earlier session's clicks back on their frames. Returns the first frame (as an
    index into `truth`) that still has none, or the last one when all are done."""
    by_frame = {r.frame: r for r in saved}
    for tf in truth:
        r = by_frame.get(tf.frame)
        if r is not None and not tf.clicks:
            tf.clicks = [Click(tuple(p.foot_px), p.field_yd, p.track_id, fixed=True) for p in r.players]
    return next((k for k, tf in enumerate(truth) if not tf.clicks), len(truth) - 1)


def save(out: Path, truth: list[TruthFrame]) -> tuple[int, int]:
    """Write the frames that have clicks. Returns (frames, positions); nothing is written for none."""
    done = [t for t in truth if t.clicks]
    if not done:
        return 0, 0
    out.parent.mkdir(parents=True, exist_ok=True)
    return len(done), write_csv(out, to_results(done))


def typed(entry: str, key: int) -> tuple[str | None, int | None]:
    """One key while a number is being typed in the window: (what is typed now, the number
    if Enter finished it). Typed text of None means the entry is over."""
    if key in (13, 10):
        return (None, int(entry)) if entry else (None, None)
    if key == 27:
        return None, None
    if key in (8, 127):
        return entry[:-1], None
    if ord("0") <= key <= ord("9") and len(entry) < 3:
        return entry + chr(key), None
    return entry, None


def default_out(video: str | Path, shot: int | None) -> Path:
    stem = Path(video).stem
    return Path("data/truth") / (f"{stem}.shot{shot}.csv" if shot is not None else f"{stem}.csv")


def _draw(frame: np.ndarray, tf: TruthFrame, res: FrameResult, show_boxes: bool, k: int, n: int, scale: float,
          entry: str | None = None, target: int | None = None) -> np.ndarray:
    img = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if show_boxes:
        for p in res.players:
            x1, y1, x2, y2 = (int(v * scale) for v in p.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), 1)
    for j, c in enumerate(tf.clicks):
        x, y = int(c.px[0] * scale), int(c.px[1] * scale)
        cv2.drawMarker(img, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(img, str(c.pid), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, str(c.pid), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        if entry is not None and j == target:
            cv2.circle(img, (x, y), 18, (0, 0, 255), 2, cv2.LINE_AA)
    text = (f"frame {k + 1}/{n}  t={tf.t:.1f}s  shot {tf.shot}  {len(tf.clicks)} players   "
            "click=feet  n=next  p=prev  u=undo  x=delete  i=renumber (cursor on it)  b=boxes  q=quit (saved as you go)")
    colour = (255, 255, 255)
    if entry is not None and target is not None and target < len(tf.clicks):
        text, colour = f"number for the ringed click (was {tf.clicks[target].pid}): {entry}_    Enter = set, Esc = leave it", (0, 255, 255)
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return img


def label_interactive(video, run: list[FrameResult], frames: list[FrameResult], out: Path, reach: float = 5.0,
                      scale: float = 0.5) -> list[TruthFrame]:
    """The clicking loop. Returns what was labelled. `out` is written on every move between
    frames and on any exit, and read first if it is already there."""
    from .capture import VideoSource

    src = VideoSource(str(video))
    maps = mappings_for(run)
    truth = [TruthFrame(r.frame, r.t, r.shot, r.anchored) for r in frames]
    k = 0
    if out.exists():
        k = resume(truth, read_csv(out))
        print(f"picked up {sum(1 for t in truth if t.clicks)} labelled frames from {out}; starting at frame {k + 1}")
    clicked: list[tuple[int, int]] = []
    cursor = [0, 0]

    def on_mouse(event, x, y, flags, param):
        cursor[0], cursor[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    def settle(j: int) -> None:
        """Number frame j's clicks that are not yet fixed, against the two frames before it."""
        others = max((c.pid for i, t in enumerate(truth) if i != j for c in t.clicks), default=0)
        number_frame(truth[j].clicks, truth[j - 1].clicks if j > 0 else [], truth[j - 2].clicks if j > 1 else [], reach, others)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    show_boxes = False
    entry: str | None = None                # the number being typed, in the window
    target: int | None = None               # ...and the click it is for
    image = src.read_frame(frames[k].frame)
    try:
        while True:
            tf, res = truth[k], frames[k]
            cv2.imshow(WINDOW, _draw(image, tf, res, show_boxes, k, len(frames), scale, entry, target))
            key = cv2.waitKey(30) & 0xFF
            if entry is not None:                               # typing a number: the keys are digits, not commands
                clicked.clear()
                if key != 255:
                    entry, number = typed(entry, key)
                    if number is not None and target is not None and target < len(tf.clicks):
                        renumber(tf.clicks, target, number)
                continue
            if clicked:
                x, y = clicked.pop(0)
                px = (x / scale, y / scale)
                m = maps.get(tf.frame)
                yd = None if m is None else tuple(float(v) for v in m.to_field(np.array([px]))[0])
                tf.clicks.append(Click(px, yd, 0))
                settle(k)
            elif key == ord("u") and tf.clicks:
                tf.clicks.pop()
                settle(k)
            elif key == ord("x"):
                j = nearest(tf.clicks, cursor[0], cursor[1], scale)
                if j is not None:
                    tf.clicks.pop(j)
                    settle(k)
            elif key == ord("i") and tf.clicks:
                j = nearest(tf.clicks, cursor[0], cursor[1], scale)
                entry, target = "", len(tf.clicks) - 1 if j is None else j
            elif key == ord("b"):
                show_boxes = not show_boxes
            elif key in (ord("n"), ord("p")):
                for c in tf.clicks:                             # a frame once left keeps its numbers: later frames lean on them
                    c.fixed = True
                save(out, truth)
                k = min(k + 1, len(frames) - 1) if key == ord("n") else max(k - 1, 0)
                image = src.read_frame(frames[k].frame)
            elif key == ord("q"):
                break
    finally:                                                    # Ctrl-C, a closed window, an error: the clicks are kept
        n_frames, n = save(out, truth)
        cv2.destroyAllWindows()
        src.release()
        print(f"{n_frames} frames, {n} player positions -> {out}" if n_frames else "nothing labelled, nothing written")
    return truth


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Click where each player stands on sampled frames of a run; writes truth in the tracking schema.")
    ap.add_argument("video")
    ap.add_argument("csv", help="the run's tracking CSV for this clip (gives the frames and the field mapping)")
    ap.add_argument("--shot", type=int, help="label this shot (play) only")
    ap.add_argument("--start", type=float, help="seconds into the clip")
    ap.add_argument("--stop", type=float)
    ap.add_argument("--every", type=float, default=0.5, help="seconds between labelled frames")
    ap.add_argument("--reach", type=float, default=5.0, help="yards from where a man was heading within which a click keeps his number")
    ap.add_argument("--whole-shot", action="store_true", help="label the whole shot, not just the play (a second before the snap to six after)")
    ap.add_argument("--scale", type=float, default=0.5, help="display scale (4K frames at 0.5 fit a laptop screen)")
    ap.add_argument("--out", help=f"default data/truth/<clip>.shot<N>.csv")
    a = ap.parse_args(argv)
    run = read_csv(a.csv)
    if a.start is None and a.stop is None and not a.whole_shot:
        window = play_window(a.csv, a.shot)
        if window is not None:
            a.start, a.stop = window
            print(f"labelling the play: {a.start:.1f}s to {a.stop:.1f}s (snap at {a.start + BEFORE_SNAP:.1f}s); --whole-shot for all of it")
    frames = sample_frames(run, a.every, a.shot, a.start, a.stop)
    if not frames:
        raise SystemExit("no frames to label: check --shot / --start / --stop against the run")
    out = Path(a.out) if a.out else default_out(a.video, a.shot)
    label_interactive(a.video, run, frames, out, reach=a.reach, scale=a.scale)


if __name__ == "__main__":
    main()
