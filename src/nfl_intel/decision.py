"""Grade the throw: who the ball went to, when it left, and what else was there.

The film doesn't show the ball well enough to track, but the play-by-play says who was
thrown to, how far downfield the ball was caught or fell, and to which side. Put next to
the tracked receivers that is enough to pick out the target on most plays, and from where he
was when the ball arrived, to work back to the moment it was thrown. Every receiver is then
scored at that moment, which is the question a quarterback is judged on: of the throws he
had, which did he take?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schema import FrameResult
from .value import PassValueModel, receiver_features

# Fitted to 3,351 throws in the Big Data Bowl set (passer to landing spot, frames in the air):
# flight = 0.09 s + distance / 20.8 yd/s, off by about a tenth of a second for the typical throw.
BALL_SPEED = 20.8
RELEASE = 0.09
EARLIEST, LATEST = 0.6, 7.0   # seconds after the snap a pass can be in the air
# How long quarterbacks held the ball in the same set, by how far downfield the ball went
# (medians over 3,300 throws). Where a receiver was at the ball's depth for a while, the
# moment that best fits this is taken. Against the truth that puts the throw within 0.3 s on
# 41% of throws and within 0.6 s on 69%; without tracking the ball it does no better, which
# is why a grade is only called firm when it holds half a second either side.
HOLD = ((0.0, 2.5), (5.0, 2.2), (10.0, 2.3), (15.0, 2.7), (20.0, 3.0), (30.0, 2.8), (1e9, 3.2))
DEPTH_SLACK = 1.5             # yards: "at the ball's depth" means within this of his closest approach
UNSURE = 0.5                  # seconds either side of the estimate that the grade is re-run at


def typical_hold(air_yards: float) -> float:
    return next(hold for upto, hold in HOLD if air_yards < upto)


@dataclass
class Option:
    label: str
    expected: float
    p_catch: float
    yards: float
    separation: float
    depth: float


@dataclass
class ThrowGrade:
    target: str                      # the label we track him under
    thrown_at: float                 # seconds after the snap, estimated
    arrived_at: float
    how: str                         # "by name" or "by where the ball went"
    rank: int                        # 1 = the best option on the board
    options: list[Option] = field(default_factory=list)
    ranks: tuple[int, int] = (0, 0)  # best and worst rank if the throw came half a second earlier or later
    firm: bool = False               # the rank is the same across that half second

    @property
    def chosen(self) -> Option:
        return next(o for o in self.options if o.label == self.target)

    @property
    def best(self) -> Option:
        return max(self.options, key=lambda o: o.expected)

    @property
    def left_on_the_field(self) -> float:
        """Expected yards given up by not taking the best option."""
        return self.best.expected - self.chosen.expected

    def sentence(self) -> str:
        c = self.chosen
        text = f"{self.target}, option {self.rank} of {len(self.options)} at about +{self.thrown_at:.1f}s: {c.p_catch:.0%} caught, {c.expected:.1f} yd expected"
        if self.rank > 1:
            text += f"; {self.best.label} was {self.best.expected:.1f}"
        if not self.firm:
            text += f" (option {self.ranks[0]} to {self.ranks[1]} depending on the moment)"
        return text

    def as_dict(self) -> dict:
        return {"target": self.target, "thrown_at": round(self.thrown_at, 2), "arrived_at": round(self.arrived_at, 2), "how": self.how,
                "rank": self.rank, "of": len(self.options), "firm": self.firm, "ranks": list(self.ranks),
                "left_on_the_field": round(self.left_on_the_field, 2),
                "options": [{"label": o.label, "expected": round(o.expected, 2), "catch": round(o.p_catch, 2),
                             "separation": round(o.separation, 1), "depth": round(o.depth, 1)} for o in self.options]}


def tracks(frames: list[FrameResult], snap: float) -> dict[str, dict]:
    """Every named player's path after the snap: label -> {role, t (since snap), xy}."""
    out: dict[str, dict] = {}
    for f in frames:
        for p in list(f.players) + list(getattr(f, "loose", [])):
            if p.field_yd is None or p.role in (None, "OFF"):
                continue
            key = p.label or f"{p.role}{p.track_id}"
            tr = out.setdefault(key, {"role": p.role, "t": [], "xy": []})
            tr["role"] = p.role
            tr["t"].append(f.t - snap)
            tr["xy"].append(p.field_yd)
    for tr in out.values():
        order = np.argsort(tr["t"])
        tr["t"] = np.array(tr["t"])[order]
        tr["xy"] = np.array(tr["xy"], dtype=np.float64)[order]
    return out


def where(track: dict, t: float, gap: float = 0.6) -> np.ndarray | None:
    """Position at time t, between the sightings either side; None if he wasn't seen near then."""
    ts = track["t"]
    if len(ts) == 0 or t < ts[0] - gap or t > ts[-1] + gap:
        return None
    k = int(np.argmin(np.abs(ts - t)))
    if abs(ts[k] - t) > gap:
        return None
    return np.array([np.interp(t, ts, track["xy"][:, 0]), np.interp(t, ts, track["xy"][:, 1])])


def velocity(track: dict, t: float, span: float = 0.25) -> np.ndarray:
    a, b = where(track, t - span), where(track, t + span)
    if a is None or b is None:
        a, b = where(track, t - 2 * span), where(track, t)
        if a is None or b is None:
            return np.zeros(2)
    return (b - a) / (2 * span)


def find_target(paths: dict[str, dict], los: float, ball_y: float, air_yards: float | None, pass_location: str,
                name: str = "", gained: float | None = None, complete: bool = False) -> tuple[str, float, str] | None:
    """(label, seconds after the snap the ball arrived, how we know) or None when it can't be told.

    A named receiver is taken at his word. Otherwise the ball's landing spot picks him: the
    receiver on that side who was that far downfield, and, on a completion, who then ended
    the play about where the gain says the ball carrier did.
    """
    if air_yards is None:
        return None
    receivers = {k: v for k, v in paths.items() if v["role"] == "REC" and len(v["t"]) >= 5}
    if not receivers:
        return None

    def arrival(tr: dict) -> tuple[float, float] | None:
        """When he was at the ball's depth on the ball's side: (time, how far off he was).
        If he was there for a while, the moment that fits a typical hold-and-throw is taken."""
        ok = (tr["t"] >= EARLIEST) & (tr["t"] <= LATEST)
        if not ok.any():
            return None
        t, xy = tr["t"][ok], tr["xy"][ok]
        off = np.abs(xy[:, 0] - los - air_yards)
        side = xy[:, 1] - ball_y
        if pass_location == "left":
            off = off + np.clip(2.0 - side, 0, None)
        elif pass_location == "right":
            off = off + np.clip(2.0 + side, 0, None)
        elif pass_location == "middle":
            off = off + np.clip(np.abs(side) - 8.0, 0, None)
        there = off <= off.min() + DEPTH_SLACK
        hold = typical_hold(air_yards)
        flight = RELEASE + np.hypot(xy[:, 0] - (los - 5.0), xy[:, 1] - ball_y) / BALL_SPEED     # the passer is about five yards deep
        k = int(np.argmin(np.where(there, np.abs(t - flight - hold), np.inf)))
        return float(t[k]), float(off.min())

    if name and name in receivers:
        got = arrival(receivers[name])
        if got is not None and got[1] <= 3.0:        # named, but never seen where the ball went: his track was lost
            return name, got[0], "by name"
        return None
    scored = []
    for label, tr in receivers.items():
        got = arrival(tr)
        if got is None:
            continue
        cost = got[1]
        if complete and gained is not None:
            cost += 0.5 * abs(float(tr["xy"][-1, 0]) - los - gained) if tr["t"][-1] > got[0] + 0.5 else 1.5
        scored.append((cost, label, got[0]))
    if not scored:
        return None
    scored.sort()
    if scored[0][0] > 3.0 or (len(scored) > 1 and scored[1][0] - scored[0][0] < 1.0):
        return None                                  # nobody fits, or two fit about as well
    return scored[0][1], scored[0][2], "by where the ball went"


def grade_throw(frames: list[FrameResult], snap: float, los: float, model: PassValueModel, air_yards: float | None,
                pass_location: str = "", name: str = "", gained: float | None = None, complete: bool = False) -> ThrowGrade | None:
    """Grade one pass from its anchored frames (positions on the real field, roles filled in)."""
    paths = tracks(frames, snap)
    qb = next((v for v in paths.values() if v["role"] == "QB"), None)
    if qb is None or len(qb["t"]) < 3:
        return None
    ball_y = float(where(qb, 0.0, gap=1.5)[1]) if where(qb, 0.0, gap=1.5) is not None else float(qb["xy"][0, 1])
    found = find_target(paths, los, ball_y, air_yards, pass_location, name, gained, complete)
    if found is None:
        return None
    target, arrived, how = found
    spot = where(paths[target], arrived)
    thrown = arrived
    for _ in range(3):                               # the ball left a flight time before it arrived
        passer = where(qb, thrown, gap=2.0)
        if passer is None or spot is None:
            break
        thrown = max(0.3, arrived - RELEASE - float(np.linalg.norm(spot - passer)) / BALL_SPEED)
    passer = where(qb, thrown, gap=2.0)
    if passer is None:
        return None
    def board(at: float) -> list[Option] | None:
        """Every receiver scored at one moment, best first."""
        here_qb = where(qb, at, gap=2.0)
        if here_qb is None:
            return None
        labels, pos, vel, roles = ["QB"], [here_qb], [velocity(qb, at)], ["QB"]
        for label, tr in paths.items():
            if tr["role"] in ("REC", "DEF"):
                here = where(tr, at)
                if here is not None:
                    labels.append(label); pos.append(here); vel.append(velocity(tr, at)); roles.append(tr["role"])
        receivers = [i for i, r in enumerate(roles) if r == "REC"]
        defenders = [i for i, r in enumerate(roles) if r == "DEF"]
        if target not in labels or len(defenders) < 2:
            return None
        xy, v = np.array(pos), np.array(vel)
        X = np.stack([receiver_features(xy, v, r, 0, defenders, los, at) for r in receivers])
        p, y = model.score(X)
        options = [Option(labels[r], float(p[i] * y[i]), float(p[i]), float(y[i]), float(X[i, 0]), float(X[i, 4])) for i, r in enumerate(receivers)]
        return sorted(options, key=lambda o: -o.expected)

    def rank_in(options: list[Option]) -> int:
        return 1 + next(i for i, o in enumerate(options) if o.label == target)

    options = board(thrown)
    if options is None:
        return None
    rank = rank_in(options)
    around = [rank] + [rank_in(b) for b in (board(max(0.3, thrown - UNSURE)), board(thrown + UNSURE)) if b is not None]
    return ThrowGrade(target, float(thrown), float(arrived), how, rank, options, (min(around), max(around)), len(around) == 3 and min(around) == max(around))
