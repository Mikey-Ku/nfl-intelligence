"""Top-down view of an anchored play: a slice of the real field, players by side, the best throw.

Same picture as the Big Data Bowl demo, drawn from film. The offense always moves left to
right, and its left is at the top.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .draw import PILE, PILE_SPAN, team_color
from .field import FIELD_WIDTH, HASH_INSET
from .fieldtrack import chains

GRASS = (62, 118, 44)
END_ZONE = (48, 92, 34)
PAINT = (215, 225, 215)
BEST = (120, 230, 120)
LOS = (255, 200, 80)
FIRST_DOWN = (60, 220, 255)
WHITE = (245, 245, 245)
OFFENSE_ROLES = ("QB", "REC", "OL")
DEFENSE_ROLES = ("DEF", "DL")


def _text(img, text, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 40, 20), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class FieldSlice:
    """The part of the field a play happens on, drawn once."""

    def __init__(self, los: float, height: int = 720, behind: float = 15.0, ahead: float = 35.0, to_go: float | None = None):
        self.los = los
        self.x0 = max(0.0, los - behind)
        self.x1 = min(120.0, max(los + ahead, self.x0 + 40.0))
        self.x0 = min(self.x0, self.x1 - 40.0)
        self.scale = height / FIELD_WIDTH
        self.height = height
        self.width = int((self.x1 - self.x0) * self.scale)
        img = np.full((height, self.width, 3), GRASS, dtype=np.uint8)
        if self.x0 < 10:
            cv2.rectangle(img, (0, 0), (self.px(10.0), height - 1), END_ZONE, -1)
        if self.x1 > 110:
            cv2.rectangle(img, (self.px(110.0), 0), (self.width - 1, height - 1), END_ZONE, -1)
        for yard in range(int(np.ceil(self.x0)), int(self.x1) + 1):
            x = self.px(yard)
            if yard % 5 == 0 and 10 <= yard <= 110:
                cv2.line(img, (x, 0), (x, height - 1), PAINT, 2 if yard in (10, 110) else 1)
                if yard % 10 == 0 and 20 <= yard <= 100:
                    n = yard - 10 if yard <= 60 else 110 - yard
                    for yy in (int(height * 0.13), int(height * 0.90)):
                        cv2.putText(img, str(n), (x - 16, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 215, 200), 2, cv2.LINE_AA)
            elif 10 < yard < 110:
                for hy in (HASH_INSET, FIELD_WIDTH - HASH_INSET):
                    cv2.line(img, (x, int(hy * self.scale) - 4), (x, int(hy * self.scale) + 4), (200, 215, 200), 1)
        if to_go is not None and los + to_go < 110:
            cv2.line(img, (self.px(los + to_go), 0), (self.px(los + to_go), height - 1), FIRST_DOWN, 2)
        cv2.line(img, (self.px(los), 0), (self.px(los), height - 1), LOS, 2)
        self.base = img

    def px(self, x: float) -> int:
        return int(round((x - self.x0) * self.scale))

    def P(self, pt) -> tuple[int, int]:
        # y is flipped so the offense's left is at the top of the picture
        return self.px(pt[0]), int(round((FIELD_WIDTH - pt[1]) * self.scale))


def side_of(p, offense_team: int | None, los: float) -> int | None:
    """The jersey group a player is drawn as: his role decides, then where he stands, then his shirt."""
    if offense_team is None:
        return p.team
    if p.role in OFFENSE_ROLES:
        return offense_team
    if p.role in DEFENSE_ROLES:
        return 1 - offense_team
    return p.team


def draw_topdown(view: FieldSlice, analysis, trails: dict[int, list] | None = None, trail: int = 10) -> np.ndarray:
    img = view.base.copy()
    anchor = analysis.anchor
    unseen = {id(p) for p in analysis.unseen}
    players = [p for p in list(analysis.players) + list(getattr(analysis, "loose", [])) + list(analysis.unseen)
               if p.field_yd is not None and -2 < p.field_yd[1] < FIELD_WIDTH + 2 and view.x0 - 1 < p.field_yd[0] < view.x1 + 1]
    ids = {id(p): sid for p, sid in zip(analysis.players, analysis.slot_ids)}
    k = view.height / 720.0                     # everything is sized for a 720-line picture
    piled = [p for p in players if p.pile]
    if piled:                                   # a pile is one area: where it is, how many are in it, and no more than that
        where = np.array([p.field_yd for p in piled])
        group = chains(where, PILE_SPAN)
        shade = img.copy()
        rings = []
        for g in np.unique(group):
            pts = where[group == g]
            if len(pts) < 2:                    # one man still counted in a pile that has moved off him: a mark, no area
                continue
            centre = pts.mean(axis=0)
            radius = int(round((float(np.linalg.norm(pts - centre, axis=1).max()) + 1.1) * view.scale))
            rings.append((view.P(centre), radius, int((group == g).sum())))
            cv2.circle(shade, view.P(centre), radius, PILE, -1, cv2.LINE_AA)
        img = cv2.addWeighted(shade, 0.22, img, 0.78, 0.0)
        for c, radius, n in rings:
            cv2.circle(img, c, radius, PILE, 1, cv2.LINE_AA)
            _text(img, f"pile, {n}", (c[0] - int(22 * k), c[1] - radius - int(6 * k)), 0.42 * k, (225, 235, 245))
    if trails is not None:
        seen = set()
        for p in players:
            if p.pile:
                continue
            sid = ids.get(id(p), p.track_id)
            seen.add(sid)
            hist = trails.setdefault(sid, [])
            hist.append(p.field_yd)
            del hist[:-trail]
            color = team_color(side_of(p, anchor.offense_team, anchor.los_abs))
            for a, b in zip(hist, hist[1:]):
                cv2.line(img, view.P(a), view.P(b), color, 1, cv2.LINE_AA)
        for sid in [k for k in trails if k not in seen]:
            del trails[sid]
    best = analysis.best
    by_label = {s.label: s for s in analysis.scores}
    qb = next((p for p in players if p.role == "QB"), None)
    target = next((p for p in players if best is not None and p.label == best.label), None)
    if qb is not None and target is not None:
        cv2.line(img, view.P(qb.field_yd), view.P(target.field_yd), BEST, 2, cv2.LINE_AA)
    for p in players:
        if p.role == "OFF":
            continue
        c = view.P(p.field_yd)
        color = team_color(side_of(p, anchor.offense_team, anchor.los_abs))
        if p.pile:                              # somewhere in there: a small mark in his side's colour, and no name
            cv2.circle(img, c, int(round(3 * k)), color, -1 if id(p) not in unseen else 1, cv2.LINE_AA)
            continue
        r = int(round((5 if p.role in ("OL", "DL") else 7) * k))
        cv2.circle(img, c, r + 2, (20, 40, 20), -1, cv2.LINE_AA)
        if id(p) in unseen:                     # lost by the detector: a ring where he should be by now
            cv2.circle(img, c, r, color, 2, cv2.LINE_AA)
        else:
            cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
        if p is target:
            cv2.circle(img, c, r * 2, BEST, 2, cv2.LINE_AA)
        if p.role == "QB":
            _text(img, "QB", (c[0] - int(10 * k), c[1] - r - 6), 0.5 * k)
        elif p.label in by_label:
            s = by_label[p.label]
            _text(img, f"{s.label} {s.expected_yards:.1f}", (c[0] + r + 5, c[1] + 4), 0.5 * k, BEST if p is target else WHITE)
        elif p.label or p.role:
            _text(img, p.label or p.role, (c[0] + r + 4, c[1] + 4), 0.42 * k, (225, 235, 245))
    return img


def title_for(analysis) -> str:
    parts = []
    if analysis.record is not None and analysis.record.header.complete:
        parts.append(analysis.record.summary)
    if analysis.since_snap is not None:
        parts.append(f"{analysis.since_snap:+.1f} s from the snap")
    elif analysis.anchor is not None:
        parts.append("before the snap")
    return "   ".join(parts)


def view_for(analysis, height: int = 720) -> FieldSlice:
    to_go = analysis.record.header.distance if analysis.record is not None else None
    return FieldSlice(analysis.anchor.los_abs, height=height, to_go=to_go)


class TopDownWriter:
    """Writes one top-down video per anchored shot, with the stats panel beside it."""

    def __init__(self, out_dir: str | Path, stem: str, fps: float, height: int = 720, panel_width: int = 450):
        self.out_dir, self.stem, self.fps = Path(out_dir), stem, fps
        self.height, self.panel_width = height, panel_width
        self.title_h = 48
        self.paths: list[Path] = []
        self._writer = None
        self._key = None
        self._view: FieldSlice | None = None
        self._trails: dict[int, list] = {}
        self._last = None

    def add(self, result, analysis, snapshot) -> np.ndarray | None:
        from .panel import qimage_to_bgr, render_panel

        if analysis.anchor is None or not analysis.players:
            return None
        key = (result.shot, round(analysis.anchor.los_abs, 1), analysis.anchor.direction)
        if key != self._key:
            self.close()
            self._key, self._trails = key, {}
            self._view = view_for(analysis, self.height)
            size = (self._view.width + 16 + self.panel_width, self.height + self.title_h)
            path = self.out_dir / f"{self.stem}.shot{result.shot}.topdown.mp4"
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
            self.paths.append(path)
        view = self._view
        frame = np.full((self.height + self.title_h, view.width + 16 + self.panel_width, 3), 20, dtype=np.uint8)
        frame[self.title_h:, : view.width] = draw_topdown(view, analysis, self._trails)
        if snapshot is not None:
            dpr = 1.4
            panel = qimage_to_bgr(render_panel(snapshot, int(self.panel_width / dpr), int(self.height / dpr), dpr=dpr))
            ph, pw = min(panel.shape[0], self.height), min(panel.shape[1], self.panel_width)
            frame[self.title_h : self.title_h + ph, view.width + 16 : view.width + 16 + pw] = panel[:ph, :pw]
        cv2.putText(frame, title_for(analysis), (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
        self._writer.write(frame)
        self._last = frame
        return frame

    def close(self, hold: float = 1.5) -> None:
        if self._writer is not None:
            if self._last is not None:
                for _ in range(int(hold * self.fps)):
                    self._writer.write(self._last)
            self._writer.release()
        self._writer, self._key, self._last = None, None, None
