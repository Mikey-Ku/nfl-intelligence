"""Build the showcase videos: chosen plays, each with a caption, in two cuts.

  outputs/showcase/sideline.mp4   film + top-down view + panel (for a private viewing: it contains NFL film)
  outputs/showcase/topdown.mp4    top-down view + panel only (no film in it: safe to publish)

Run from the project root:  uv run python scripts/make_showcase.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nfl_intel.capture import VideoSource                      # noqa: E402
from nfl_intel.cli import _build_live, _compose                # noqa: E402
from nfl_intel.panel import qimage_to_bgr, render_panel        # noqa: E402
from nfl_intel.topdown import draw_topdown, view_for           # noqa: E402

# clip, seconds in, seconds out, caption
SEGMENTS = [
    ("data/clips/live-01.mp4", 38.0, 45.0, "ATL at GB, Q4 8:21. Love to Golden, 15 yd TD. The play is read off the screen; the field is anchored by reading the painted numbers."),
    ("data/clips/live-02.mp4", 20.0, 42.0, "CAR at CLE, Q1 14:10. Young to Tremayne, 16 yd. Roles at the snap, then the best throw and its expected yards, live."),
    ("data/clips/live-04.mp4", 203.0, 225.0, "CAR at CLE, Q2 11:56. Young to Evans, 11 yd. The panel compares the play with a typical one from the same stretch of field."),
    ("data/clips/live-04.mp4", 240.0, 258.0, "CAR at CLE, Q2 10:44. Young to Tremble, 9 yd. Names come from NFL Pro's on-field list; D1..D8 are coverage defenders."),
    ("data/clips/live-04.mp4", 410.0, 428.0, "CAR at CLE, Q2 7:10. Watson to Fannin, 8 yd. The nearest rusher is shown beside the value: the model can't see the rush."),
    ("data/clips/live-04.mp4", 548.0, 566.0, "CAR at CLE, Q2 3:36. Judkins run, 11 yd. Runs get the formation, personnel and box count; only passes get a value."),
]

HEIGHT, BAR = 900, 70


def caption_bar(width: int, text: str) -> np.ndarray:
    bar = np.full((BAR, width, 3), 14, dtype=np.uint8)
    size = 0.95
    while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 2)[0][0] > width - 40 and size > 0.5:
        size -= 0.05
    cv2.putText(bar, text, (20, 46), cv2.FONT_HERSHEY_SIMPLEX, size, (235, 235, 235), 2, cv2.LINE_AA)
    return bar


def topdown_only(tracker, snap, caption_width: int) -> np.ndarray:
    a = getattr(tracker, "analysis", None)
    panel = qimage_to_bgr(render_panel(snap, 320, 600, dpr=1.5))
    slot = np.full((HEIGHT, 850, 3), 20, dtype=np.uint8)
    if a is not None and a.anchor is not None and a.players:
        top = draw_topdown(view_for(a, HEIGHT), a)
        k = min(850 / top.shape[1], 1.0)
        if k < 1:
            top = cv2.resize(top, (int(top.shape[1] * k), int(top.shape[0] * k)), interpolation=cv2.INTER_AREA)
        oy, ox = (HEIGHT - top.shape[0]) // 2, (850 - top.shape[1]) // 2
        slot[oy : oy + top.shape[0], ox : ox + top.shape[1]] = top
    else:
        cv2.putText(slot, "waiting for a mapped sideline view", (120, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 190, 180), 2, cv2.LINE_AA)
    gap = np.full((HEIGHT, 16, 3), 20, dtype=np.uint8)
    return np.hstack([slot, gap, panel])


def main() -> None:
    from PySide6.QtGui import QGuiApplication

    app = QGuiApplication.instance() or QGuiApplication([])    # the panel draws with Qt
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/showcase")
    ap.add_argument("--only", type=int, default=None, help="build one segment (index) for a quick look")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fps = 24.0
    writers: dict[str, cv2.VideoWriter | None] = {"sideline": None, "topdown": None}
    model_args = argparse.Namespace(calib=None, weights="yolo11s.pt", device=None, conf=0.3, imgsz=1280, tracker="botsort", tiles=1,
                                    no_analysis=False, no_play_lookup=False, no_hints=False, value_model=None, coverage_model=None)
    segments = SEGMENTS if args.only is None else [SEGMENTS[args.only]]
    t_all = time.perf_counter()
    for clip, start, stop, caption in segments:
        src = VideoSource(clip, start=start, stop=stop)
        tracker, stats = _build_live(model_args, fps=src.fps)
        n = 0
        for idx, t, frame in src:
            res = tracker.process(frame, idx, t)
            snap = stats.update(res, tracker.field.bbox if tracker.field is not None else None)
            side = _compose(frame, res, snap, tracker, fixed=True)
            top = topdown_only(tracker, snap, side.shape[1])
            for name, picture in (("sideline", side), ("topdown", top)):
                picture = np.vstack([caption_bar(picture.shape[1], caption), picture])
                w = 1920 if name == "sideline" else 1280
                picture = cv2.resize(picture, (w, int(picture.shape[0] * w / picture.shape[1]) // 2 * 2), interpolation=cv2.INTER_AREA)
                if writers[name] is None:
                    writers[name] = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps, (picture.shape[1], picture.shape[0]))
                writers[name].write(picture)
            n += 1
        print(f"{clip} {start:.0f}-{stop:.0f}s: {n} frames  ({time.perf_counter() - t_all:.0f}s so far)", flush=True)
    for w in writers.values():
        if w is not None:
            w.release()
    print(f"wrote {out / 'sideline.mp4'} and {out / 'topdown.mp4'}")


if __name__ == "__main__":
    main()
