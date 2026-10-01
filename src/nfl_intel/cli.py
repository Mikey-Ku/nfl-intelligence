"""`nfl` command line.

  nfl check                          environment check: device, weights, screen capture
  nfl windows [QUERY]                list capturable windows (find the NFL Pro player)
  nfl overlay --window "NFL Pro"     live overlay on top of the player
  nfl capture --window "NFL Pro"     save the raw capture as a clip to re-run offline
  nfl run clip.mp4 [...]             track clip files to CSV (+ optional annotated video)
  nfl replay outputs/clip.csv        render a top-down replay video from a CSV
  nfl eval outputs/clip.csv gt.csv   position error, recall, ID switches vs hand labels
  nfl calibrate --video clip.mp4     click landmarks to build a homography
  nfl prelabel clips/*.mp4           keyframes + detector boxes as YOLO labels to correct
  nfl train                          fine-tune YOLO on the corrected labels
  nfl value-train                    train the pass value models on Big Data Bowl throws
  nfl coverage-train                 train man/zone and coverage-shell recognition
  nfl value-demo --week 16           replay real plays top-down with the live plus/minus
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

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
from .track import TRACKERS


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
    p.add_argument("--tiles", type=int, default=1, help="split the frame into N x N overlapping tiles to catch small far-sideline players (N² cost)")
    p.add_argument("--tracker", choices=TRACKERS, default="botsort",
                   help="botsort compensates for camera pans (default); bytetrack is faster for a fixed camera")
    p.add_argument("--calib", help="calibration JSON from `nfl calibrate`")
    p.add_argument("--no-analysis", action="store_true",
                   help="track only: no play lookup, roles, real yard lines or pass value")
    p.add_argument("--no-play-lookup", action="store_true",
                   help="don't read NFL Pro's header or look the play up; yard lines come from the painted numbers alone")
    p.add_argument("--no-hints", action="store_true",
                   help="name roles from the formation alone, without the play's charting or the on-field list (for marking them)")
    p.add_argument("--value-model", default=None, help="pass value model (default models/pass_value.pkl)")
    p.add_argument("--coverage-model", default=None, help="coverage model (default models/coverage.pkl)")


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
                       conf=args.conf, imgsz=args.imgsz, tracker=args.tracker, tiles=args.tiles)


def _build_live(args, fps: float):
    """The tracker and the panel's stats. With analysis on (the default) the tracker also reads
    the play, anchors the field, names roles and scores the pass."""
    from .stats import LiveStats, geometry_rows

    tracker = _build_tracker(args, fps)
    if getattr(args, "no_analysis", False):
        return tracker, LiveStats(metrics_fn=geometry_rows)
    from .analysis import AnalyzedTracker, PlayAnalyzer, live_stats, load_models

    value, coverage = load_models(args.value_model, args.coverage_model)
    if value is None:
        print("no pass value model found: the panel shows formations only (train one with `nfl value-train`)")
    analyzer = PlayAnalyzer(value, coverage, score_every=2,       # the value moves slowly; every other frame is plenty
                            read_plays=not getattr(args, "no_play_lookup", False), use_hints=not getattr(args, "no_hints", False))
    return AnalyzedTracker(tracker, analyzer), live_stats(analyzer, value)


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


def _shot(frame, res, snap, tracker, path: Path) -> None:
    """One test image: the annotated field next to the stats panel."""
    from .draw import annotate
    from .fieldmask import expand_bbox
    from .panel import qimage_to_bgr, render_panel

    base = frame
    registrar = getattr(tracker, "registrar", None)
    analysis = getattr(tracker, "analysis", None)
    anchor = analysis.anchor if analysis is not None else None
    if registrar is not None and registrar.locked and registrar.rect is not None:
        from .registration import draw_grid

        base = draw_grid(frame, registrar.H, registrar.rect, label=None if anchor is None else anchor.yard_name)
    img = annotate(base, res)
    field = getattr(tracker, "field", None)
    if field is not None:
        x1, y1, x2, y2 = expand_bbox(field.bbox, frame.shape, margin=0.04)
        img = img[y1:y2, x1:x2]
    height = 900
    img = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height), interpolation=cv2.INTER_AREA)
    panel = qimage_to_bgr(render_panel(snap, 320, 600, dpr=1.5))
    gap = np.full((height, 16, 3), 20, dtype=np.uint8)
    parts = [img, gap]
    if anchor is not None and analysis.players:
        from .topdown import draw_topdown, view_for

        parts += [draw_topdown(view_for(analysis, height), analysis), gap]
    cv2.imwrite(str(path), np.hstack(parts + [panel]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def _build_live_stats(tracker):
    """Fresh stats for the next clip, on the tracker already built."""
    from .stats import LiveStats, geometry_rows

    if not hasattr(tracker, "analyzer"):
        return LiveStats(metrics_fn=geometry_rows)
    from .analysis import live_stats

    return live_stats(tracker.analyzer, tracker.analyzer.value_model)


def _write_plays(tracker, path: Path) -> None:
    """The play log: one entry per play worked out, and a line about each on the terminal."""
    import json

    analyzer = tracker.analyzer
    analyzer.reset()                                   # closes the play still open
    plays, analyzer.plays = analyzer.plays, []
    if not plays:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plays, indent=1))
    print(f"  {len(plays)} play{'s' if len(plays) != 1 else ''} worked out -> {path}")
    for p in plays:
        line = f"    {p.get('play', 'play not read')}: {p.get('offense_look') or 'no look'}"
        if p.get("personnel"):
            line += f" ({p['personnel'].split(':')[0]} v {p['package'].split(',')[0].lower()})"
        if p.get("peak"):
            line += f"; best moment {p['peak']['value']:.1f} yd to {p['peak']['best']} at +{p['peak']['t']:.1f}s"
        if p.get("result"):
            line += f"; {p['result']}"
        if p.get("throw"):
            t = p["throw"]
            line += f"; threw to {t['target']}, option {t['rank']} of {t['of']} at about +{t['thrown_at']:.1f}s" + ("" if t["firm"] else " (not firm)")
        print(line)


def _narrate(tracker, res, told: set) -> None:
    """Print each thing the analysis works out, once per shot, as it happens."""
    a = getattr(tracker, "analysis", None)
    if a is None:
        return

    def say(key: str, text: str) -> None:
        if (res.shot, key) not in told:
            told.add((res.shot, key))
            print(f"  shot {res.shot} {res.t:6.1f}s  {text}")

    if a.record is not None and a.record.header.complete:
        found = f"{a.record.play_type}, {a.record.outcome}" if a.record.matched else "not found in the play-by-play"
        say("play", f"play: {a.record.summary}  ({found})")
        listed = a.record.on_field
        if listed is not None and listed.personnel:
            say("personnel", f"on the field: {listed.offense_team} in {listed.personnel}; {listed.defense_team} in {listed.package}")
    if a.anchor is not None:
        where = a.anchor.yard_name(a.anchor.los_rel) or "goal line"
        say("anchor", f"line of scrimmage on the {where}, {110 - a.anchor.los_abs:.0f} from the goal line, settled by the {a.anchor.source}")
    if a.look[0] and a.look[1] and a.snap_t is None:
        say("look", f"lined up: {a.look[0]} | {a.look[1]}")
    if a.snap_t is not None:
        say("snap", f"snap at {a.snap_t:.1f}s: {a.look[0]} | {a.look[1]}")
    if a.value is not None:
        say("value", f"first value {a.value:.1f} yd, best throw {a.best.label}; receivers " + ", ".join(s.label for s in a.scores))


def cmd_overlay(args) -> None:
    source = _resolve_source(args, fps_cap=args.fps_cap)
    region = source.region
    tracker, stats = _build_live(args, fps=args.fps_cap or 30.0)

    results = []
    writer = None
    shots_dir = Path(args.shots) if args.shots else None
    if shots_dir:
        shots_dir.mkdir(parents=True, exist_ok=True)
    last_shot = [-1e9]

    def sink(res, frame, snap):
        nonlocal writer
        if args.out:
            results.append(res)
        if args.record:
            from .draw import annotate

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.fps_cap or 30.0, (w, h))
            writer.write(annotate(frame, res))
        if shots_dir and snap is not None and res.players and res.t - last_shot[0] >= args.shot_every:
            last_shot[0] = res.t
            _shot(frame, res, snap, tracker, shots_dir / f"shot_{res.t:06.1f}s.jpg")

    try:
        if args.mirror:
            from .draw import annotate
            from .panel import qimage_to_bgr, render_panel
            from PySide6.QtGui import QGuiApplication

            app = QGuiApplication.instance() or QGuiApplication([])
            print("mirror mode: press q in the window to stop")
            t_start = time.monotonic()
            for idx, t, frame in source:
                res = tracker.process(frame, idx, t)
                field = tracker.field
                snap = stats.update(res, field.bbox if field is not None else None)
                sink(res, frame, snap)
                view = annotate(frame, res)
                if not args.no_panel:
                    panel = qimage_to_bgr(render_panel(snap, 300, view.shape[0] // 2, dpr=2.0))
                    view = np.hstack([view, panel])
                cv2.imshow("nfl overlay (mirror)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if args.seconds and time.monotonic() - t_start > args.seconds:
                    break
            cv2.destroyAllWindows()
            del app
        else:
            from .overlay import run_overlay

            print("overlay running: Ctrl-C in this terminal to stop" + (f" (auto-stop after {args.seconds:g}s)" if args.seconds else ""))
            run_overlay(region, tracker, source, sink, show_minimap=not args.no_minimap, seconds=args.seconds,
                        stats=stats, panel_side=None if args.no_panel else args.panel_side)
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            print(f"wrote {args.record}")
        if hasattr(tracker, "finish"):
            tracker.finish()
            if args.out:
                _write_plays(tracker, Path(args.out).with_suffix(".plays.json"))
        if args.out and results:
            n = write_csv(args.out, results)
            print(f"wrote {n} rows to {args.out}")
        if shots_dir:
            print(f"test shots in {shots_dir}")


def cmd_capture(args) -> None:
    """Save the raw (un-annotated) capture so the pipeline can be re-run on it offline."""
    source = _resolve_source(args, fps_cap=args.fps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    times = []
    t_start = time.monotonic()
    try:
        for idx, t, frame in source:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(frame)
            times.append(t)
            if time.monotonic() - t_start > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
    if len(times) > 1:
        real_fps = (len(times) - 1) / (times[-1] - times[0])
        out.with_suffix(".times.csv").write_text("frame,t\n" + "".join(f"{i},{t:.3f}\n" for i, t in enumerate(times)))
        print(f"wrote {len(times)} frames to {out} (captured at {real_fps:.1f} fps, file plays at {args.fps:g})")
    else:
        print("nothing captured")


def cmd_run(args) -> None:
    from PySide6.QtGui import QGuiApplication

    from .draw import annotate

    app = QGuiApplication.instance() or QGuiApplication([])   # the stats panel renders with Qt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = None
    for path in args.videos:
        src = VideoSource(path, start=args.start, stop=args.stop)
        if tracker is not None:
            tracker.reset()
        tracker, stats = _build_live(args, fps=src.fps) if tracker is None else (tracker, _build_live_stats(tracker))
        stem = Path(path).stem
        csv_path = out_dir / f"{stem}.csv"
        writer = None
        if args.save_video:
            writer = cv2.VideoWriter(str(out_dir / f"{stem}.annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), src.fps, src.size)
        topdown = None
        if args.topdown and hasattr(tracker, "analysis"):
            from .topdown import TopDownWriter

            topdown = TopDownWriter(out_dir, stem, src.fps)
        t0 = time.perf_counter()
        results = []
        last_shot = -1e9
        told = set()
        for idx, t, frame in src:
            res = tracker.process(frame, idx, t)
            results.append(res)
            snap = stats.update(res, tracker.field.bbox if tracker.field is not None else None)
            if topdown is not None:
                topdown.add(res, tracker.analysis, snap)
            _narrate(tracker, res, told)
            if args.shots and res.players and t - last_shot >= args.shot_every:
                last_shot = t
                shots = Path(args.shots)
                shots.mkdir(parents=True, exist_ok=True)
                _shot(frame, res, snap, tracker, shots / f"{stem}_{t:06.1f}s.jpg")
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
        if topdown is not None:
            topdown.close()
            for made in topdown.paths:
                print(f"  top-down replay -> {made}")
        src.release()
        if hasattr(tracker, "finish"):
            tracker.finish()
            _write_plays(tracker, out_dir / f"{stem}.plays.json")
        n = write_csv(csv_path, results)
        dt = time.perf_counter() - t0
        ids = {p.track_id for r in results for p in r.players}
        print(f"{path}: {len(results)} frames, {n} rows -> {csv_path}  ({len(results) / dt:.1f} fps, "
              f"{len(ids)} track ids, {snap.reacquired} picked up late)")
    if args.preview:
        cv2.destroyAllWindows()


def cmd_replay(args) -> None:
    from .replay import render_replay

    out = args.out or str(Path(args.csv).with_suffix(".replay.mp4"))
    render_replay(args.csv, out, fps=args.fps, scale=args.scale)
    print(f"wrote {out}")


def cmd_prelabel(args) -> None:
    from .detect import PlayerDetector
    from .train import prelabel, write_dataset_yaml

    det = PlayerDetector(args.weights, device=args.device, conf=args.conf, imgsz=args.imgsz, tiles=args.tiles)
    n = prelabel(args.videos, args.out_dir, det, every=args.every, max_per_clip=args.max_per_clip)
    yaml_path = write_dataset_yaml(Path(args.out_dir))
    print(f"wrote {n} pre-labeled frames under {args.out_dir}; dataset file {yaml_path}")
    print("next: correct the boxes (X-AnyLabeling reads YOLO txt), then `nfl train`")


def cmd_train(args) -> None:
    from .train import fine_tune, write_dataset_yaml

    root = Path(args.data_dir)
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        yaml_path = write_dataset_yaml(root)
    best = fine_tune(yaml_path, weights=args.weights, epochs=args.epochs, imgsz=args.imgsz,
                     batch=args.batch, freeze_epochs=args.freeze_epochs, device=args.device, name=args.name)
    print(f"best weights: {best}\nuse them with: nfl overlay --weights {best} ...")


def cmd_value_baseline(args) -> None:
    from .value_train import rebaseline

    rebaseline(args.model, args.data, args.outcomes)


def cmd_value_train(args) -> None:
    from .value_train import train

    train(root=args.data, outcomes_path=args.outcomes, out=args.out, noise=args.noise)


def cmd_coverage_train(args) -> None:
    from .coverage import train

    train(root=args.data, outcomes_path=args.outcomes, out=args.out)


def cmd_value_demo(args) -> None:
    from PySide6.QtGui import QGuiApplication

    from .bdb import iter_plays, load_outcomes
    from .demo import render, score_plays
    from .value import PassValueModel

    app = QGuiApplication.instance() or QGuiApplication([])   # the panel renders with Qt
    model = PassValueModel.load(args.model)
    coverage = None
    if Path(args.coverage_model).exists():
        from .coverage import CoverageModel

        coverage = CoverageModel.load(args.coverage_model)
    outcomes = load_outcomes(args.outcomes)
    plays = []
    for play in iter_plays(args.week, args.data, outcomes):
        if args.play is not None and play.play_id != args.play:
            continue
        if args.game is not None and play.game_id != args.game:
            continue
        if play.qb < 0 or play.target < 0 or len(play.xy) < 20 or not play.receivers:
            continue
        plays.append(play)
    if not plays:
        sys.exit("no matching plays")
    scored = score_plays(plays, model)
    if args.play is None:
        scored.sort(key=lambda sp: -sp.swing)          # the plays where the value moved most
    out_dir = Path(args.out_dir)
    for sp in scored[: args.count]:
        name = f"value_w{sp.play.week:02d}_{sp.play.game_id}_{sp.play.play_id}"
        path = render(sp, out_dir / f"{name}.mp4", shots_dir=out_dir / "shots", model=model, coverage=coverage)
        print(f"{path}  value {sp.values[0]:.1f} -> {sp.values[-1]:.1f} yd (range {sp.swing:.1f})  {sp.play.desc[:90]}")
    del app


def cmd_data(args) -> None:
    from .nflverse import describe, refresh

    for kind, size in refresh(args.season, args.root).items():
        print(f"  {kind:14s} " + ("not published for this season yet" if size is None else f"{size / 1e6:.1f} MB"))
    print(describe(args.season, args.root))


def cmd_roles_check(args) -> None:
    from .roles_check import check

    lo, _, hi = args.weeks.partition("-")
    check(list(range(int(lo), int(hi or lo) + 1)), noise=args.noise, root=args.data)


def cmd_plays_check(args) -> None:
    from .plays_check import report

    report(args.logs)


def cmd_eval(args) -> None:
    from .evaluate import evaluate_files

    print(evaluate_files(args.pred, args.truth, args.threshold))


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
    o.add_argument("--seconds", type=float, default=None, help="stop automatically after this many seconds")
    o.add_argument("--no-panel", action="store_true", help="hide the live stats panel")
    o.add_argument("--panel-side", choices=["left", "right"], default="left", help="which side of the video the stats panel docks to")
    o.add_argument("--shots", help="save a test image (annotated field + stats panel) into this folder every few seconds")
    o.add_argument("--shot-every", type=float, default=3.0, help="seconds between test images")
    o.set_defaults(func=cmd_overlay)

    cp = sub.add_parser("capture", help="save the raw window capture to a clip for offline re-runs")
    _add_source_args(cp)
    cp.add_argument("--out", default="data/clips/capture.mp4")
    cp.add_argument("--seconds", type=float, default=20.0)
    cp.add_argument("--fps", type=float, default=15.0, help="capture rate cap and playback rate of the file")
    cp.set_defaults(func=cmd_capture)

    r = sub.add_parser("run", help="track clip files")
    r.add_argument("videos", nargs="+")
    _add_model_args(r)
    r.add_argument("--out-dir", default="outputs")
    r.add_argument("--save-video", action="store_true", help="also write <clip>.annotated.mp4")
    r.add_argument("--preview", action="store_true", help="show frames while tracking (q to skip)")
    r.add_argument("--shots", help="save test images (annotated field + stats panel) into this folder")
    r.add_argument("--shot-every", type=float, default=2.0, help="seconds between test images")
    r.add_argument("--topdown", action="store_true", help="also write a top-down replay with the stats panel for every shot that anchors")
    r.add_argument("--start", type=float, default=0.0, help="begin this many seconds into the clip")
    r.add_argument("--stop", type=float, default=None, help="end this many seconds into the clip")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("replay", help="render a top-down replay from a tracking CSV")
    rp.add_argument("csv")
    rp.add_argument("--out")
    rp.add_argument("--fps", type=float, default=30.0)
    rp.add_argument("--scale", type=float, default=8.0, help="pixels per yard")
    rp.set_defaults(func=cmd_replay)

    pl = sub.add_parser("prelabel", help="sample keyframes from clips with the detector's boxes as YOLO labels to correct")
    pl.add_argument("videos", nargs="+")
    _add_model_args(pl)
    pl.add_argument("--out-dir", default="data/labels")
    pl.add_argument("--every", type=int, default=30, help="keep every Nth frame")
    pl.add_argument("--max-per-clip", type=int, default=None)
    pl.set_defaults(func=cmd_prelabel)

    tr = sub.add_parser("train", help="fine-tune YOLO on the corrected labels")
    tr.add_argument("--data-dir", default="data/labels")
    tr.add_argument("--weights", default="yolo11s.pt")
    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--freeze-epochs", type=int, default=10, help="stage-1 epochs with the backbone frozen (0 to skip)")
    tr.add_argument("--imgsz", type=int, default=1280)
    tr.add_argument("--batch", type=int, default=4)
    tr.add_argument("--device", default=None)
    tr.add_argument("--name", default="nfl")
    tr.set_defaults(func=cmd_train)

    vt = sub.add_parser("value-train", help="train the pass value models on Big Data Bowl throws")
    vt.add_argument("--data", default="data/bdb/2026-prediction")
    vt.add_argument("--outcomes", default="data/nflverse/play_by_play_2023.parquet")
    vt.add_argument("--out", default="models/pass_value.pkl")
    vt.add_argument("--noise", type=float, default=0.3, help="position noise (yards) added to a copy of the training data")
    vt.set_defaults(func=cmd_value_train)

    vb = sub.add_parser("value-baseline", help="recompute the typical-play curves for the trained model (no retraining)")
    vb.add_argument("--data", default="data/bdb/2026-prediction")
    vb.add_argument("--outcomes", default="data/nflverse/play_by_play_2023.parquet")
    vb.add_argument("--model", default="models/pass_value.pkl")
    vb.set_defaults(func=cmd_value_baseline)

    ct = sub.add_parser("coverage-train", help="train coverage recognition (man/zone and shell) on Big Data Bowl plays")
    ct.add_argument("--data", default="data/bdb/2026-prediction")
    ct.add_argument("--outcomes", default="data/nflverse/play_by_play_2023.parquet")
    ct.add_argument("--out", default="models/coverage.pkl")
    ct.set_defaults(func=cmd_coverage_train)

    vd = sub.add_parser("value-demo", help="replay Big Data Bowl plays top-down with the live value panel")
    vd.add_argument("--week", type=int, default=16, help="2023 week (15-18 were held out of training)")
    vd.add_argument("--game", type=int, default=None)
    vd.add_argument("--play", type=int, default=None)
    vd.add_argument("--count", type=int, default=2, help="how many plays to render (largest value swings first)")
    vd.add_argument("--data", default="data/bdb/2026-prediction")
    vd.add_argument("--outcomes", default="data/nflverse/play_by_play_2023.parquet")
    vd.add_argument("--model", default="models/pass_value.pkl")
    vd.add_argument("--coverage-model", default="models/coverage.pkl")
    vd.add_argument("--out-dir", default="outputs/value_demo")
    vd.set_defaults(func=cmd_value_demo)

    da = sub.add_parser("data", help="fetch this season's public play-by-play and charting (nflverse)")
    da.add_argument("--season", type=int, default=2026)
    da.add_argument("--root", default="data/nflverse")
    da.set_defaults(func=cmd_data)

    rc = sub.add_parser("roles-check", help="mark role assignment against real snaps (Big Data Bowl)")
    rc.add_argument("--weeks", default="1-3", help="2023 weeks, e.g. 1-3 or 16")
    rc.add_argument("--noise", type=float, default=0.4, help="position noise in yards, standing in for video")
    rc.add_argument("--data", default="data/bdb/2026-prediction")
    rc.set_defaults(func=cmd_roles_check)

    pc = sub.add_parser("plays-check", help="mark play logs against the play-by-play and the official charting")
    pc.add_argument("logs", nargs="+", help="<clip>.plays.json files from `nfl run`")
    pc.set_defaults(func=cmd_plays_check)

    ev = sub.add_parser("eval", help="score a tracking CSV against a ground-truth CSV in the same schema")
    ev.add_argument("pred")
    ev.add_argument("truth")
    ev.add_argument("--threshold", type=float, default=None, help="match distance (default 3 yd calibrated, 60 px otherwise)")
    ev.set_defaults(func=cmd_eval)

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
