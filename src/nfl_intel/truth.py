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

Identities: on the first frame clicks are numbered in order. On later frames each click
takes the number of the nearest player on the previous frame within `--reach` yards, else
a new number; the number is drawn by the click so a wrong one can be fixed (`i`, type it in
the window, Enter; Esc leaves it). Keys: `n` next frame, `p` previous, `u` undo the last
click, `b` show or hide the run's boxes (hidden by default so they don't lead the eye), `i`
renumber the last click, `q` quit. The CSV goes to `data/truth/<clip>.shot<N>.csv` unless
`--out` says otherwise. It is written on every move between frames and on any exit, and a
play that already has a file is picked up at its first unlabelled frame.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .field import FieldCalibration
from .schema import FrameResult, TrackedPlayer, read_csv, write_csv

WINDOW = "nfl truth"
MIN_POINTS = 4


@dataclass
class Click:
    px: tuple[float, float]
    yd: tuple[float, float] | None
    pid: int


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


def assign_id(yd: tuple[float, float] | None, previous: list[Click], taken: set[int], reach: float) -> int:
    """The nearest previous player's number within `reach` yards not already used on this frame, else a new one."""
    if yd is not None:
        best = None
        for c in previous:
            if c.yd is None or c.pid in taken:
                continue
            d = float(np.hypot(c.yd[0] - yd[0], c.yd[1] - yd[1]))
            if d <= reach and (best is None or d < best[0]):
                best = (d, c.pid)
        if best is not None:
            return best[1]
    used = taken | {c.pid for c in previous}
    return max(used, default=0) + 1


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
            tf.clicks = [Click(tuple(p.foot_px), p.field_yd, p.track_id) for p in r.players]
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
          entry: str | None = None) -> np.ndarray:
    img = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if show_boxes:
        for p in res.players:
            x1, y1, x2, y2 = (int(v * scale) for v in p.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), 1)
    for c in tf.clicks:
        x, y = int(c.px[0] * scale), int(c.px[1] * scale)
        cv2.drawMarker(img, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(img, str(c.pid), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    text = (f"frame {k + 1}/{n}  t={tf.t:.1f}s  shot {tf.shot}  {len(tf.clicks)} players   "
            "click=feet  n=next  p=prev  u=undo  i=renumber  b=boxes  q=quit (saved as you go)")
    colour = (255, 255, 255)
    if entry is not None and tf.clicks:
        text, colour = f"number for the last click (was {tf.clicks[-1].pid}): {entry}_    Enter = set, Esc = leave it", (0, 255, 255)
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return img


def label_interactive(video, run: list[FrameResult], frames: list[FrameResult], out: Path, reach: float = 3.0,
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

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    show_boxes = False
    entry: str | None = None                # the number being typed for the last click, in the window
    image = src.read_frame(frames[k].frame)
    try:
        while True:
            tf, res = truth[k], frames[k]
            cv2.imshow(WINDOW, _draw(image, tf, res, show_boxes, k, len(frames), scale, entry))
            key = cv2.waitKey(30) & 0xFF
            if entry is not None:                               # typing a number: the keys are digits, not commands
                clicked.clear()
                if key != 255:
                    entry, number = typed(entry, key)
                    if number is not None and tf.clicks:
                        tf.clicks[-1].pid = number
                continue
            if clicked:
                x, y = clicked.pop(0)
                px = (x / scale, y / scale)
                m = maps.get(tf.frame)
                yd = None if m is None else tuple(float(v) for v in m.to_field(np.array([px]))[0])
                previous = truth[k - 1].clicks if k > 0 else []
                tf.clicks.append(Click(px, yd, assign_id(yd, previous, {c.pid for c in tf.clicks}, reach)))
            elif key == ord("u") and tf.clicks:
                tf.clicks.pop()
            elif key == ord("i") and tf.clicks:
                entry = ""
            elif key == ord("b"):
                show_boxes = not show_boxes
            elif key in (ord("n"), ord("p")):
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
    ap.add_argument("--reach", type=float, default=3.0, help="yards within which a click keeps the previous frame's number")
    ap.add_argument("--scale", type=float, default=0.5, help="display scale (4K frames at 0.5 fit a laptop screen)")
    ap.add_argument("--out", help=f"default data/truth/<clip>.shot<N>.csv")
    a = ap.parse_args(argv)
    run = read_csv(a.csv)
    frames = sample_frames(run, a.every, a.shot, a.start, a.stop)
    if not frames:
        raise SystemExit("no frames to label: check --shot / --start / --stop against the run")
    out = Path(a.out) if a.out else default_out(a.video, a.shot)
    label_interactive(a.video, run, frames, out, reach=a.reach, scale=a.scale)


if __name__ == "__main__":
    main()
