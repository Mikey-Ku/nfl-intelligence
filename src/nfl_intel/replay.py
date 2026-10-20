"""`nfl replay`: a top-down replay video from a saved tracking CSV."""

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
