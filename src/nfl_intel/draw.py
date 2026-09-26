"""OpenCV drawing for annotated frames and the top-down minimap."""

from __future__ import annotations

import cv2
import numpy as np

from .field import draw_field, field_to_image
from .schema import FrameResult

TEAM_COLORS = {0: (0, 200, 255), 1: (255, 120, 60), None: (200, 200, 200)}  # BGR


def team_color(team: int | None) -> tuple[int, int, int]:
    return TEAM_COLORS.get(team, TEAM_COLORS[None])


def annotate(frame: np.ndarray, result: FrameResult, thickness: int = 2) -> np.ndarray:
    out = frame.copy()
    for p in result.players:
        x1, y1, x2, y2 = (int(round(v)) for v in p.box)
        color = team_color(p.team)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{p.track_id}"
        if p.field_yd is not None:
            label += f" ({p.field_yd[0]:.0f},{p.field_yd[1]:.0f})"
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.circle(out, (int(p.foot_px[0]), int(p.foot_px[1])), 3, color, -1)
    cv2.putText(out, f"frame {result.frame}  players {len(result.players)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def minimap(result: FrameResult, scale: float = 6.0, base: np.ndarray | None = None) -> np.ndarray | None:
    """Top-down field with a dot per calibrated player, or None if nothing is calibrated."""
    pts = [(p.field_yd, p.team, p.track_id) for p in result.players if p.field_yd is not None]
    if not pts:
        return None
    img = (base if base is not None else draw_field(scale)).copy()
    for yd, team, tid in pts:
        x, y = field_to_image(yd, scale)[0]
        cv2.circle(img, (int(x), int(y)), max(3, int(scale * 0.6)), team_color(team), -1)
        cv2.putText(img, str(tid), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale / 6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img
