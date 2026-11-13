"""OpenCV drawing: boxes and labels on a frame, and the small top-down minimap."""

from __future__ import annotations

import cv2
import numpy as np

from .field import draw_field, field_to_image
from .schema import FrameResult

TEAM_COLORS = {0: (0, 200, 255), 1: (255, 120, 60), None: (200, 200, 200)}  # BGR
LINE_COLORS = {"scrimmage": (255, 200, 80), "to gain": (60, 220, 255)}       # as on television: blue and yellow


PILE = (235, 235, 235)        # the colour a pile is outlined and shaded in
PILE_SPAN = 2.5               # yards: piled players this close to the next are drawn as one pile


def pile_groups(players) -> list[list]:
    """The players marked as in a pile, gathered into the piles they make."""
    from .fieldtrack import chains

    piled = [p for p in players if p.pile and p.field_yd is not None]
    if not piled:
        return []
    group = chains(np.array([p.field_yd for p in piled]), PILE_SPAN)
    groups = [[p for p, g in zip(piled, group) if g == k] for k in np.unique(group)]
    return [g for g in groups if len(g) >= 2]


def pile_outlines(players, pad: float = 4.0) -> list[tuple[tuple[float, float, float, float], list]]:
    """One rectangle in the picture round each pile, with the players in it. Piles whose
    rectangles run into each other are drawn as one."""
    found = []
    for pile in pile_groups(players):
        with_box = [p for p in pile if p.box[2] > p.box[0]]
        if with_box:
            b = np.array([p.box for p in with_box])
            found.append([b[:, 0].min() - pad, b[:, 1].min() - pad, b[:, 2].max() + pad, b[:, 3].max() + pad, with_box])
    merged = True
    while merged:
        merged = False
        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                a, b = found[i], found[j]
                if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                    found[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] + b[4]]
                    del found[j]
                    merged = True
                    break
            if merged:
                break
    return [((f[0], f[1], f[2], f[3]), f[4]) for f in found]


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
    for rect, pile in pile_outlines(result.players):    # one outline round a pile, and nobody in it named
        x1, y1, x2, y2 = (int(round(v)) for v in rect)
        cv2.rectangle(out, (x1, y1), (x2, y2), PILE, thickness)
        size = max(0.6, frame.shape[0] / 1500)
        cv2.putText(out, "pile", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, size, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(out, "pile", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, size, PILE, 2, cv2.LINE_AA)
        for p in pile:
            cv2.circle(out, (int(p.foot_px[0]), int(p.foot_px[1])), 4, team_color(p.team), -1)
    for p in list(result.players) + list(getattr(result, "loose", [])):
        if p.role == "OFF" or p.pile:                   # officials and strays get no box: the boxes are the players
            continue
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
