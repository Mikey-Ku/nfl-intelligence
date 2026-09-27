"""Render a tracking CSV as a top-down replay video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .draw import team_color
from .field import draw_field, field_to_image
from .schema import FrameResult, read_csv


def render_replay(
    csv_path: str | Path,
    out_path: str | Path,
    fps: float = 30.0,
    scale: float = 8.0,
    trail: int = 8,
) -> Path:
    """Calibrated plays draw on the field in yards; uncalibrated ones draw in pixel space."""
    results = read_csv(csv_path)
    if not results:
        raise ValueError(f"{csv_path} has no rows")
    calibrated = any(p.field_yd is not None for r in results for p in r.players)

    if calibrated:
        base = draw_field(scale)
        def pos(p):
            return None if p.field_yd is None else field_to_image(p.field_yd, scale)[0]
    else:
        max_x = max(p.foot_px[0] for r in results for p in r.players) + 20
        max_y = max(p.foot_px[1] for r in results for p in r.players) + 20
        base = np.full((int(max_y), int(max_x), 3), (40, 40, 40), dtype=np.uint8)
        def pos(p):
            return np.array(p.foot_px)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = base.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    history: dict[int, list[np.ndarray]] = {}
    radius = max(3, int(scale * 0.6))
    for res in results:
        img = base.copy()
        seen = set()
        for p in res.players:
            xy = pos(p)
            if xy is None:
                continue
            seen.add(p.track_id)
            hist = history.setdefault(p.track_id, [])
            hist.append(xy)
            del hist[:-trail]
            color = team_color(p.team)
            for a, b in zip(hist, hist[1:]):
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 1, cv2.LINE_AA)
            cv2.circle(img, (int(xy[0]), int(xy[1])), radius, color, -1, cv2.LINE_AA)
            cv2.putText(img, str(p.track_id), (int(xy[0]) + radius, int(xy[1]) - radius),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        for tid in list(history):
            if tid not in seen:
                del history[tid]
        cv2.putText(img, f"t={res.t:5.2f}s  frame {res.frame}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    return out_path


def render_topdown(results: list[FrameResult], out_path: str | Path, fps: float = 30.0, height: int = 720,
                   trail: int = 12, title: str = "") -> Path | None:
    """Top-down replay of frames that are mapped to yards, on a slice of field around the
    players. Yard numbers are relative to the registration's own zero line."""
    from .field import FIELD_WIDTH, HASH_INSET

    frames = [r for r in results if any(p.field_yd is not None for p in r.players)]
    if len(frames) < 2:
        return None
    xy = np.array([p.field_yd for r in frames for p in r.players if p.field_yd is not None])
    ok = (xy[:, 1] > -3) & (xy[:, 1] < FIELD_WIDTH + 3)
    x0 = 5 * np.floor((np.percentile(xy[ok, 0], 1) - 4) / 5)
    x1 = 5 * np.ceil((np.percentile(xy[ok, 0], 99) + 4) / 5)
    scale = height / (FIELD_WIDTH + 4.0)
    width = int((x1 - x0) * scale)

    def P(pt):
        return int((pt[0] - x0) * scale), int((FIELD_WIDTH + 2.0 - pt[1]) * scale)

    base = np.full((height, width, 3), (62, 118, 44), dtype=np.uint8)
    cv2.rectangle(base, P((x0, FIELD_WIDTH)), P((x1, 0.0)), (235, 235, 235), 2)
    for x in np.arange(x0, x1 + 1, 5.0):
        cv2.line(base, P((x, 0.0)), P((x, FIELD_WIDTH)), (215, 225, 215), 1)
        cv2.putText(base, f"{x:.0f}", (P((x, 3.0))[0] + 4, P((x, 3.0))[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 215, 200), 1, cv2.LINE_AA)
    for x in np.arange(x0, x1 + 0.5, 1.0):
        for row in (HASH_INSET, FIELD_WIDTH - HASH_INSET):
            cv2.line(base, P((x, row - 0.33)), P((x, row + 0.33)), (200, 215, 200), 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    history: dict[int, list] = {}
    for res in frames:
        img = base.copy()
        seen = set()
        for p in res.players:
            if p.field_yd is None or not (-3 < p.field_yd[1] < FIELD_WIDTH + 3):
                continue
            seen.add(p.track_id)
            hist = history.setdefault(p.track_id, [])
            hist.append(p.field_yd)
            del hist[:-trail]
            color = team_color(p.team)
            for a, b in zip(hist, hist[1:]):
                cv2.line(img, P(a), P(b), color, 1, cv2.LINE_AA)
            cv2.circle(img, P(p.field_yd), 8, (20, 40, 20), -1, cv2.LINE_AA)
            cv2.circle(img, P(p.field_yd), 6, color, -1, cv2.LINE_AA)
        for tid in [k for k in history if k not in seen]:
            del history[tid]
        cv2.putText(img, f"{title}  t={res.t:5.1f}s  {len(seen)} players".strip(), (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    return out_path
