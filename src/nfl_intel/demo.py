"""Replay a Big Data Bowl pass play top-down with the live value panel beside it.

This is the plus/minus running on the NFL's own tracking, from the snap to the throw. The
same value code will run on video-derived tracking once positions are in yards; until then
this shows what the panel will say and lets the model be checked against real plays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .bdb import HZ, Play
from .field import FIELD_WIDTH
from .schema import FrameResult, TrackedPlayer
from .stats import LiveStats
from .value import PassValueModel, ReceiverScore, play_value, velocities

OFFENSE = (255, 170, 80)      # BGR
DEFENSE = (60, 200, 255)
BEST = (120, 230, 120)
LOS = (255, 200, 80)
WHITE = (245, 245, 245)


@dataclass
class ScoredPlay:
    play: Play
    scores: list[list[ReceiverScore]]      # per frame, per receiver
    values: np.ndarray                     # per frame: expected yards of the best throw
    best: list[ReceiverScore | None]

    @property
    def swing(self) -> float:
        return float(self.values.max() - self.values.min())


SUFFIXES = {"jr.", "jr", "sr.", "sr", "ii", "iii", "iv", "v"}


def short_name(name: str) -> str:
    parts = [p for p in name.split() if p.lower() not in SUFFIXES]
    return parts[-1] if parts else name


def play_features(play: Play) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Features for every receiver at every frame of a play, and the (frame, receiver) of each row."""
    from .value import receiver_features

    v = velocities(play.xy, HZ)
    rows, keys = [], []
    for t in range(len(play.xy)):
        for r in play.receivers:
            rows.append(receiver_features(play.xy[t], v[t], r, play.qb, play.defenders, play.los, t / HZ))
            keys.append((t, r))
    return np.array(rows), keys


def score_plays(plays: list[Play], model: PassValueModel) -> list[ScoredPlay]:
    """Score many plays with one model call (gradient boosting is slow called row by row)."""
    feats = [play_features(p) for p in plays]
    X = np.vstack([f[0] for f in feats if len(f[0])])
    p_all, y_all = model.score(X)
    out, k = [], 0
    for play, (Xp, keys) in zip(plays, feats):
        labels = [short_name(n) for n in play.names]
        scores: list[list[ReceiverScore]] = [[] for _ in range(len(play.xy))]
        for i, (t, r) in enumerate(keys):
            pc, yd = float(p_all[k + i]), float(y_all[k + i])
            scores[t].append(ReceiverScore(r, pc, yd, pc * yd, float(Xp[i, 0]), labels[r]))
        k += len(keys)
        values, best = [], []
        for s in scores:
            val, b = play_value(s)
            values.append(val)
            best.append(b)
        out.append(ScoredPlay(play, scores, np.array(values), best))
    return out


def score_play(play: Play, model: PassValueModel) -> ScoredPlay:
    return score_plays([play], model)[0]


def explain(sp: ScoredPlay, t0: float, t1: float) -> str:
    """Name the receiver whose expected yards moved most between two moments, and what changed."""
    i0 = min(len(sp.scores) - 1, max(0, int(round(t0 * HZ))))
    i1 = min(len(sp.scores) - 1, max(0, int(round(t1 * HZ))))
    a = {s.index: s for s in sp.scores[i0]}
    b = {s.index: s for s in sp.scores[i1]}
    rising = sp.values[i1] >= sp.values[i0]
    pick, change = None, 0.0
    for idx, s1 in b.items():
        if idx not in a:
            continue
        d = s1.expected_yards - a[idx].expected_yards
        if (rising and d > change) or (not rising and d < change):
            pick, change = idx, d
    if pick is None:
        return ""
    s0, s1 = a[pick], b[pick]
    d_sep, d_depth = s1.separation - s0.separation, s1.depth - s0.depth
    if rising:
        if d_sep >= 0.5:
            return f"{s1.label} opens up, {s0.separation:.1f} to {s1.separation:.1f} yd of space"
        if d_depth >= 1.0:
            return f"{s1.label} gets deeper, {s0.depth:.0f} to {s1.depth:.0f} yd downfield"
        return f"{s1.label} becomes the better throw"
    if d_sep <= -0.5:
        return f"{s1.label} gets covered, {s0.separation:.1f} to {s1.separation:.1f} yd of space"
    return f"{s1.label} defender closing, catch chance {s0.p_catch:.0%} to {s1.p_catch:.0%}"


def decision(sp: ScoredPlay) -> tuple[str, str] | None:
    """How the real throw ranked among the options at the moment of the throw."""
    scores = sorted(sp.scores[-1], key=lambda s: -s.expected_yards)
    for rank, s in enumerate(scores, 1):
        if s.index == sp.play.target:
            return ("Threw to", f"{s.label}  option {rank} of {len(scores)}, {s.expected_yards:.1f} yd")
    return None


def to_results(sp: ScoredPlay) -> list[FrameResult]:
    play = sp.play
    out = []
    for t in range(len(play.xy)):
        players = []
        for i in range(play.xy.shape[1]):
            x, y = play.xy[t, i]
            players.append(TrackedPlayer(
                track_id=i, team=1 if play.offense[i] else 0, box=(x, y, x, y), conf=1.0,
                foot_px=(float(x), float(y)), field_yd=(float(x), float(y)),
                role=play.roles[i], label=short_name(play.names[i]),
            ))
        out.append(FrameResult(frame=t, t=t / HZ, size=(0, 0), players=players))
    return out


def make_stats(sp: ScoredPlay, model: PassValueModel | None = None) -> LiveStats:
    def value_fn(result: FrameResult) -> float:
        return float(sp.values[result.frame])

    def metrics_fn(result: FrameResult) -> list[tuple[str, str]]:
        scores = sp.scores[result.frame]
        if not scores:
            return []
        best = sp.best[result.frame]
        most_open = max(scores, key=lambda s: s.separation)
        tightest = min(scores, key=lambda s: s.separation)
        rows = [
            ("Best throw", f"{best.label}  {best.p_catch:.0%} caught, {best.yards_if_caught:.0f} yd"),
            ("Most open", f"{most_open.label}  {most_open.separation:.1f} yd"),
            ("Tightest coverage", f"{tightest.label}  {tightest.separation:.1f} yd"),
        ]
        if result.frame == len(sp.scores) - 1 and decision(sp) is not None:
            rows.append(decision(sp))
        return rows

    return LiveStats(value_fn=value_fn, metrics_fn=metrics_fn, value_unit="yd",
                     explain_fn=lambda t0, t1: explain(sp, t0, t1),
                     team_names=("Coverage defenders", "QB + receivers", "Unassigned"),
                     on_field_note="in this dataset (no linemen)",
                     baseline_fn=model.typical if model is not None and model.meta.get("baseline") else None)


# ---------------------------------------------------------------- drawing

def _field_view(play: Play, height: int) -> tuple[np.ndarray, float, float]:
    """A slice of the field around the play: grass, yard lines, numbers, line of scrimmage."""
    x0 = max(0.0, play.los - 14.0)
    x1 = min(120.0, max(play.los + 36.0, float(play.xy[..., 0].max()) + 6.0, play.ball_land[0] + 4.0))
    scale = height / FIELD_WIDTH
    width = int((x1 - x0) * scale)
    img = np.full((height, width, 3), (62, 118, 44), dtype=np.uint8)
    for yard in range(0, 121):
        x = int((yard - x0) * scale)
        if not 0 <= x < width:
            continue
        if yard % 5 == 0 and 10 <= yard <= 110:
            cv2.line(img, (x, 0), (x, height - 1), (215, 225, 215), 2 if yard in (10, 110) else 1)
            if yard % 10 == 0 and 20 <= yard <= 100:
                n = yard - 10 if yard <= 60 else 110 - yard
                for yy in (int(height * 0.13), int(height * 0.90)):
                    cv2.putText(img, str(n), (x - 16, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 215, 200), 2, cv2.LINE_AA)
        elif 10 < yard < 110:
            for hy in (70.75 / 3, FIELD_WIDTH - 70.75 / 3):
                cv2.line(img, (x, int(hy * scale) - 4), (x, int(hy * scale) + 4), (200, 215, 200), 1)
    if x0 < 10:
        cv2.rectangle(img, (0, 0), (int((10 - x0) * scale), height - 1), (48, 92, 34), -1)
    if x1 > 110:
        cv2.rectangle(img, (int((110 - x0) * scale), 0), (width - 1, height - 1), (48, 92, 34), -1)
    cv2.line(img, (int((play.los - x0) * scale), 0), (int((play.los - x0) * scale), height - 1), LOS, 2)
    return img, x0, scale


def _text(img, text, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 40, 20), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_frame(sp: ScoredPlay, t: int, base: np.ndarray, x0: float, scale: float, thrown: bool = False) -> np.ndarray:
    play = sp.play
    img = base.copy()

    def P(p):
        # y is flipped so the offense's left is at the top of the picture
        return int((p[0] - x0) * scale), int((FIELD_WIDTH - p[1]) * scale)

    best = sp.best[t]
    for k in range(max(0, t - 8), t):                     # short trails
        for i in range(play.xy.shape[1]):
            cv2.line(img, P(play.xy[k, i]), P(play.xy[k + 1, i]), OFFENSE if play.offense[i] else DEFENSE, 1, cv2.LINE_AA)
    if best is not None and play.qb >= 0:
        q, r = P(play.xy[t, play.qb]), P(play.xy[t, best.index])
        cv2.line(img, q, r, BEST, 2, cv2.LINE_AA)
    if thrown:
        bx, by = P(play.ball_land)
        cv2.drawMarker(img, (bx, by), WHITE, cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        if play.target >= 0 and play.qb >= 0:
            cv2.line(img, P(play.xy[t, play.qb]), (bx, by), WHITE, 1, cv2.LINE_AA)
    by_index = {s.index: s for s in sp.scores[t]}
    for i in range(play.xy.shape[1]):
        c = P(play.xy[t, i])
        color = OFFENSE if play.offense[i] else DEFENSE
        cv2.circle(img, c, 9, (20, 40, 20), -1, cv2.LINE_AA)
        cv2.circle(img, c, 7, color, -1, cv2.LINE_AA)
        if best is not None and i == best.index:
            cv2.circle(img, c, 14, BEST, 2, cv2.LINE_AA)
        if i == play.qb:
            _text(img, "QB", (c[0] - 10, c[1] - 14), 0.45)
        elif i in by_index:
            s = by_index[i]
            _text(img, f"{s.label} {s.expected_yards:.1f}", (c[0] + 12, c[1] + 4), 0.45,
                  BEST if best is not None and i == best.index else WHITE)
        else:
            _text(img, play.positions[i], (c[0] + 11, c[1] + 4), 0.38, (225, 235, 245))
    return img


def render(sp: ScoredPlay, out_path: str | Path, height: int = 900, hold: float = 2.0, shots_dir: str | Path | None = None,
           model: PassValueModel | None = None) -> Path:
    """Write the replay video (30 fps, real time) and optionally a few still shots."""
    from .panel import qimage_to_bgr, render_panel

    play = sp.play
    base, x0, scale = _field_view(play, height)
    stats = make_stats(sp, model)
    results = to_results(sp)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_w = 640
    title_h = 56
    size = (base.shape[1] + 16 + panel_w, height + title_h)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    shots = Path(shots_dir) if shots_dir else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    n = len(results)
    shot_frames = {n // 3, (2 * n) // 3, n - 1}
    last = None
    for t, res in enumerate(results):
        snap = stats.update(res)
        thrown = t == n - 1
        frame = np.full((size[1], size[0], 3), 20, dtype=np.uint8)
        frame[title_h:, : base.shape[1]] = draw_frame(sp, t, base, x0, scale, thrown=thrown)
        panel = qimage_to_bgr(render_panel(snap, int(panel_w / 1.6), int(height / 1.6), dpr=1.6))
        frame[title_h : title_h + panel.shape[0], base.shape[1] + 16 : base.shape[1] + 16 + panel.shape[1]] = panel[:height, :panel_w]
        title = f"Week {play.week}  play {play.play_id}   {t / HZ:.1f}s after the snap"
        if thrown and play.target >= 0:
            result = "caught" if play.complete else "not caught"
            yards = f" for {play.yards:.0f} yd" if play.complete and play.yards is not None else ""
            title = f"Thrown to {short_name(play.names[play.target])}: {result}{yards}"
        cv2.putText(frame, title, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
        for _ in range(3):
            writer.write(frame)
        if shots and t in shot_frames:
            cv2.imwrite(str(shots / f"{out_path.stem}_{t / HZ:04.1f}s.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        last = frame
    for _ in range(int(hold * 30)):
        writer.write(last)
    writer.release()
    return out_path
