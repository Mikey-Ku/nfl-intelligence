"""`nfl` command line.

  nfl check                          environment check: device, weights, screen capture
  nfl windows [QUERY]                list capturable windows (find the NFL Pro player)
  nfl overlay --window "NFL Pro"     live overlay on top of the player
  nfl run clip.mp4 [...]             track clip files to CSV (+ optional annotated video)
  nfl replay outputs/clip.csv        render a top-down replay video from a CSV
  nfl calibrate --video clip.mp4     click landmarks to build a homography
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from . import __version__
from .capture import (
    Region,
    ScreenSource,
    VideoSource,
    WindowSource,
    find_window,
    list_windows,
    looks_blank,
    window_by_id,
)
from .detect import DEFAULT_WEIGHTS
from .field import FieldCalibration
from .schema import write_csv


def _add_source_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--window", help="capture the window whose app or title contains this text (by window ID: works even when covered)")
    g.add_argument("--window-id", type=int, help="capture this window ID from `nfl windows`")
    g.add_argument("--region", help="capture 'left,top,width,height' of the screen in points")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help=f"YOLO weights (default {DEFAULT_WEIGHTS})")
    p.add_argument("--device", default=None, help="mps, cuda or cpu (default: auto)")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--calib", help="calibration JSON from `nfl calibrate`")


def _resolve_source(args, fps_cap: float | None = None):
    """--region -> ScreenSource; --window/--window-id -> WindowSource (captures by window ID)."""
    if args.region:
        return ScreenSource(Region.parse(args.region), fps_cap=fps_cap)
    if args.window_id:
        w = window_by_id(args.window_id)
        if w is None:
            sys.exit(f"no window with id {args.window_id}; try `nfl windows`")
    elif args.window:
        w = find_window(args.window)
        if w is None:
            sys.exit(f"no window matches {args.window!r}; try `nfl windows`")
    else:
        sys.exit("give --window TEXT, --window-id N or --region left,top,width,height")
    print(f"capturing {w}")
    return WindowSource(w, fps_cap=fps_cap)


def _build_tracker(args, fps: float):
    from .pipeline import PlayTracker

    calib = FieldCalibration.load(args.calib) if args.calib else None
    return PlayTracker(fps=fps, calibration=calib, weights=args.weights, device=args.device,
                       conf=args.conf, imgsz=args.imgsz)


# ------------------------------------------------------------------ commands

def cmd_check(args) -> None:
    import torch

    print(f"nfl-intelligence {__version__}")
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  mps={torch.backends.mps.is_available()}")
    from .detect import PlayerDetector

    t0 = time.perf_counter()
    det = PlayerDetector(args.weights, device=args.device, imgsz=args.imgsz)
    print(f"weights {args.weights} loaded on {det.device} in {time.perf_counter() - t0:.1f}s")
    import numpy as np

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    det(frame)  # warmup (compiles the MPS graph)
    t0 = time.perf_counter()
    for _ in range(5):
        det(frame)
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"inference on a 1280x720 frame at imgsz={args.imgsz}: {ms:.0f} ms  (~{1000 / ms:.0f} fps)")

    src = ScreenSource(Region(0, 0, 400, 300))
    try:
        frame = src.grab()
    except Exception as e:
        print(f"screen capture FAILED: {e}")
        return
    out = Path(args.out or "outputs/capture_test.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"screen capture: {frame.shape[1]}x{frame.shape[0]} px for a 400x300 pt region (scale {src.scale:.1f}) -> {out}")
    if looks_blank(frame):
        print("  the grab is blank: give this terminal Screen Recording permission in System Settings > Privacy & Security")
    wins = list_windows()
    titled = sum(1 for w in wins if w.title)
    on = sum(1 for w in wins if w.on_screen)
    print(f"windows: {len(wins)} ({on} on screen), {titled} with titles" + ("" if titled else "  (titles need Screen Recording permission)"))


def cmd_windows(args) -> None:
    wins = list_windows()
    if args.query:
        q = args.query.lower()
        wins = [w for w in wins if q in w.owner.lower() or q in w.title.lower()]
    if not wins:
        print("no windows found (titles need Screen Recording permission for this terminal)")
    for w in wins:
        print(w)


def cmd_overlay(args) -> None:
    source = _resolve_source(args, fps_cap=args.fps_cap)
    region = source.region
    tracker = _build_tracker(args, fps=args.fps_cap or 30.0)

    results = []
    writer = None

    def sink(res, frame):
        nonlocal writer
        if args.out:
            results.append(res)
        if args.record:
            from .draw import annotate

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.fps_cap or 30.0, (w, h))
            writer.write(annotate(frame, res))

    try:
        if args.mirror:
            from .draw import annotate

            print("mirror mode: press q in the window to stop")
            for idx, t, frame in source:
                res = tracker.process(frame, idx, t)
                sink(res, frame)
                cv2.imshow("nfl overlay (mirror)", annotate(frame, res))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
        else:
            from .overlay import run_overlay

            print("overlay running: Ctrl-C in this terminal to stop")
            run_overlay(region, tracker, source, sink, show_minimap=not args.no_minimap)
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            print(f"wrote {args.record}")
        if args.out and results:
            n = write_csv(args.out, results)
            print(f"wrote {n} rows to {args.out}")


def cmd_run(args) -> None:
    from .draw import annotate

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = None
    for path in args.videos:
        src = VideoSource(path)
        if tracker is None:
            tracker = _build_tracker(args, fps=src.fps)
        else:
            tracker.reset()
        stem = Path(path).stem
        csv_path = out_dir / f"{stem}.csv"
        writer = None
        if args.save_video:
            writer = cv2.VideoWriter(str(out_dir / f"{stem}.annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), src.fps, src.size)
        t0 = time.perf_counter()
        results = []
        for idx, t, frame in src:
            res = tracker.process(frame, idx, t)
            results.append(res)
            if writer is not None or args.preview:
                img = annotate(frame, res)
                if writer is not None:
                    writer.write(img)
                if args.preview:
                    cv2.imshow("nfl run", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        if writer is not None:
            writer.release()
        src.release()
        n = write_csv(csv_path, results)
        dt = time.perf_counter() - t0
        print(f"{path}: {len(results)} frames, {n} rows -> {csv_path}  ({len(results) / dt:.1f} fps)")
    if args.preview:
        cv2.destroyAllWindows()


def cmd_replay(args) -> None:
    from .replay import render_replay

    out = args.out or str(Path(args.csv).with_suffix(".replay.mp4"))
    render_replay(args.csv, out, fps=args.fps, scale=args.scale)
    print(f"wrote {out}")


def cmd_calibrate(args) -> None:
    from .calibrate import calibrate_interactive

    meta = {}
    if args.video:
        src = VideoSource(args.video)
        frame = src.read_frame(args.frame)
        meta = {"source": args.video, "frame": args.frame}
    elif args.image:
        frame = cv2.imread(args.image)
        meta = {"source": args.image}
    else:
        src = _resolve_source(args)
        frame = src.grab()
        meta = {"region": src.region.__dict__, "scale": src.scale}
    calibrate_interactive(frame, args.out, meta)


# ------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nfl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="environment check")
    _add_model_args(c)
    c.add_argument("--out", help="where to save the test screen grab")
    c.set_defaults(func=cmd_check)

    w = sub.add_parser("windows", help="list capturable windows")
    w.add_argument("query", nargs="?")
    w.set_defaults(func=cmd_windows)

    o = sub.add_parser("overlay", help="live overlay on a playing video")
    _add_source_args(o)
    _add_model_args(o)
    o.add_argument("--mirror", action="store_true", help="show the annotated capture in a normal window instead of a transparent overlay")
    o.add_argument("--no-minimap", action="store_true")
    o.add_argument("--fps-cap", type=float, default=None, help="limit capture rate (default: as fast as inference allows)")
    o.add_argument("--out", help="write tracking CSV when the session ends")
    o.add_argument("--record", help="write the annotated capture to this mp4")
    o.set_defaults(func=cmd_overlay)

    r = sub.add_parser("run", help="track clip files")
    r.add_argument("videos", nargs="+")
    _add_model_args(r)
    r.add_argument("--out-dir", default="outputs")
    r.add_argument("--save-video", action="store_true", help="also write <clip>.annotated.mp4")
    r.add_argument("--preview", action="store_true", help="show frames while tracking (q to skip)")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("replay", help="render a top-down replay from a tracking CSV")
    rp.add_argument("csv")
    rp.add_argument("--out")
    rp.add_argument("--fps", type=float, default=30.0)
    rp.add_argument("--scale", type=float, default=8.0, help="pixels per yard")
    rp.set_defaults(func=cmd_replay)

    cal = sub.add_parser("calibrate", help="click landmarks on a frame to build a homography")
    src = cal.add_mutually_exclusive_group()
    src.add_argument("--video")
    src.add_argument("--image")
    _add_source_args(cal)
    cal.add_argument("--frame", type=int, default=0, help="frame index when using --video")
    cal.add_argument("--out", default="calibrations/sideline.json")
    cal.set_defaults(func=cmd_calibrate)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
