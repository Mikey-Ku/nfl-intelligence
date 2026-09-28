"""Tracking output schema: one row per player per frame.

Mirrors the Big Data Bowl layout (frame, id, team, x/y in yards) so anything built on that
data can run on ours. Pixel coordinates and boxes are kept alongside for debugging.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

CSV_FIELDS = [
    "frame", "t", "shot", "track_id", "team", "conf",
    "x1", "y1", "x2", "y2",
    "foot_x_px", "foot_y_px",
    "x_yd", "y_yd",
    "role", "label",
    "anchored",
]


@dataclass
class TrackedPlayer:
    track_id: int
    team: int | None                       # 0 / 1 after clustering, None before warmup
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in source pixels
    conf: float
    foot_px: tuple[float, float]           # bottom-centre of the box: where the player stands
    field_yd: tuple[float, float] | None   # (x, y) in yards once calibrated
    role: str | None = None                # QB / REC / DEF / OL / DL / OFFICIAL once roles are assigned
    label: str | None = None               # jersey number or name when known


@dataclass
class FrameResult:
    frame: int
    t: float
    size: tuple[int, int]                  # source width, height in pixels
    players: list[TrackedPlayer]
    shot: int = 0                          # camera shot number; goes up at every cut
    anchored: bool = False                 # yards are on the real field, offense moving +x (else relative to the mapping)
    lines: list = field(default_factory=list)   # (name, [(x, y), ...] in source pixels): "scrimmage", "to gain"


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.2f}"


def write_csv(path: str | Path, results: Iterable[FrameResult]) -> int:
    """Write results as CSV. Returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for res in results:
            for p in res.players:
                fx, fy = p.field_yd if p.field_yd is not None else (None, None)
                w.writerow([
                    res.frame, f"{res.t:.3f}", res.shot, p.track_id,
                    "" if p.team is None else p.team, f"{p.conf:.3f}",
                    *(f"{v:.1f}" for v in p.box),
                    _fmt(p.foot_px[0]), _fmt(p.foot_px[1]),
                    _fmt(fx), _fmt(fy),
                    p.role or "", p.label or "",
                    int(res.anchored),
                ])
                n += 1
    return n


def read_csv(path: str | Path) -> list[FrameResult]:
    """Read a tracking CSV back into FrameResults (size is unknown, reported as 0x0)."""
    frames: dict[int, FrameResult] = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["frame"])
            res = frames.get(idx)
            if res is None:
                res = frames[idx] = FrameResult(idx, float(row["t"]), (0, 0), [], shot=int(row.get("shot") or 0),
                                                anchored=bool(int(row.get("anchored") or 0)))
            field = None
            if row["x_yd"] != "":
                field = (float(row["x_yd"]), float(row["y_yd"]))
            res.players.append(TrackedPlayer(
                track_id=int(row["track_id"]),
                team=None if row["team"] == "" else int(row["team"]),
                box=(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                conf=float(row["conf"]),
                foot_px=(float(row["foot_x_px"]), float(row["foot_y_px"])),
                field_yd=field,
                role=row.get("role") or None,
                label=row.get("label") or None,
            ))
    return [frames[k] for k in sorted(frames)]
