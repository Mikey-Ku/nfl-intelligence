"""OpenCV drawing for annotated frames and the top-down minimap."""

from __future__ import annotations

import cv2
import numpy as np

from .field import draw_field, field_to_image
from .schema import FrameResult

TEAM_COLORS = {0: (0, 200, 255), 1: (255, 120, 60), None: (200, 200, 200)}  # BGR
LINE_COLORS = {"scrimmage": (255, 200, 80), "to gain": (60, 220, 255)}       # as on television: blue and yellow


def team_color(team: int | None) -> tuple[int, int, int]:
    return TEAM_COLORS.get(team, TEAM_COLORS[None])


def player_label(p) -> str:
    """The most useful thing known about a player: his name in the play, else his role, else his track."""
    if p.label:
        return p.label
    if p.role and p.role != "OFF":
        return p.role
    if p.role == "OFF":
        return ""
    label = f"{p.track_id}"
    if p.field_yd is not None:
        label += f" ({p.field_yd[0]:.0f},{p.field_yd[1]:.0f})"
    return label


def annotate(frame: np.ndarray, result: FrameResult, thickness: int = 2) -> np.ndarray:
    out = frame.copy()
    for name, pts in result.lines:
        if len(pts) >= 2:
            cv2.polylines(out, [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)], False, LINE_COLORS.get(name, (255, 255, 255)),
                          max(2, frame.shape[0] // 400), cv2.LINE_AA)
    for p in result.players:
        x1, y1, x2, y2 = (int(round(v)) for v in p.box)
        color = team_color(p.team)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = player_label(p)
        size = 0.5 if p.label is None and p.role is None else max(0.6, frame.shape[0] / 1500)
        org = (x1, max(12, y1 - 4))
        if size > 0.5:
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, size, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1 if size <= 0.5 else 2, cv2.LINE_AA)
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
    names = {p.track_id: player_label(p) for p in result.players if p.label or p.role}
    for yd, team, tid in pts:
        x, y = field_to_image(yd, scale)[0]
        cv2.circle(img, (int(x), int(y)), max(3, int(scale * 0.6)), team_color(team), -1)
        cv2.putText(img, names.get(tid, str(tid)), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale / 6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img
