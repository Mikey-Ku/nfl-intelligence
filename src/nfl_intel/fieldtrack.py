"""Track players on the field, in yards, instead of in the picture.

In the picture a player's box jumps around as the camera pans and as bodies overlap, and the
image tracker hands out a new number every time it loses someone. On the field, positions
are smooth and nobody moves faster than about 11 yards a second, so matching each frame's
detections to the players we already know is a small assignment problem.

Each player is a slot: a smoothed position, a velocity, a team, and (after the snap) a role.
Slots coast on their velocity for a short while when their player isn't seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_SPEED = 12.0            # yards per second; nobody on a football field is faster


@dataclass
class Slot:
    id: int
    xy: np.ndarray                          # smoothed position, yards
    v: np.ndarray                           # velocity, yards per second
    team: int | None
    t: float                                # time of the last update
    seen: float                             # time the player was last actually detected
    born: float
    hits: int = 1
    role: str | None = None
    label: str | None = None
    team_votes: list = field(default_factory=lambda: [0, 0])
    track_id: int | None = None             # the image tracker's id, for drawing

    def predict(self, t: float) -> np.ndarray:
        return self.xy + self.v * (t - self.t)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.v))


class FieldTracker:
    def __init__(self, gate: float = 2.5, coast: float = 1.0, smooth: float = 0.45, velocity_window: float = 0.4,
                 team_penalty: float = 2.0, min_hits: int = 3):
        self.gate = gate                      # a detection further than this from every slot starts a new one
        self.coast = coast                    # seconds a slot survives unseen
        self.smooth = smooth                  # weight of the new measurement in the position
        self.velocity_window = velocity_window
        self.team_penalty = team_penalty      # yards added to the cost of matching across teams
        self.min_hits = min_hits
        self.slots: list[Slot] = []
        self._next = 0
        self._history: dict[int, list] = {}

    def reset(self) -> None:
        self.slots, self._next, self._history = [], 0, {}

    def update(self, t: float, xy: np.ndarray, teams: list, track_ids: list | None = None) -> list[Slot | None]:
        """Match this frame's detections to slots. Returns, for each detection, its slot."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        assigned: list[Slot | None] = [None] * len(xy)
        live = [s for s in self.slots if t - s.seen <= self.coast]
        if live and len(xy):
            pred = np.array([s.predict(t) for s in live])
            cost = np.linalg.norm(pred[:, None, :] - xy[None, :, :], axis=2)
            for i, s in enumerate(live):
                allowed = self.gate + MAX_SPEED * 0.25 * min(1.0, t - s.seen)      # unseen slots may have drifted
                for j in range(len(xy)):
                    if cost[i, j] > allowed:
                        cost[i, j] = 1e6
                    elif s.team is not None and teams[j] is not None and teams[j] != s.team:
                        cost[i, j] += self.team_penalty
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e5:
                    continue
                self._absorb(live[i], t, xy[j], teams[j], None if track_ids is None else track_ids[j])
                assigned[j] = live[i]
        for j in range(len(xy)):
            if assigned[j] is None:
                s = Slot(self._next, xy[j].copy(), np.zeros(2), teams[j], t, t, t,
                         track_id=None if track_ids is None else track_ids[j])
                if teams[j] in (0, 1):
                    s.team_votes[teams[j]] += 1
                self._next += 1
                self._history[s.id] = [(t, xy[j].copy())]
                self.slots.append(s)
                assigned[j] = s
        for s in self.slots:                                   # unseen slots coast, slowing down
            if s.t < t and t - s.seen <= self.coast:
                s.xy = s.predict(t)
                s.v = s.v * 0.9
                s.t = t
        self.slots = [s for s in self.slots if t - s.seen <= self.coast]
        for sid in [k for k in self._history if all(s.id != k for s in self.slots)]:
            del self._history[sid]
        return assigned

    def _absorb(self, s: Slot, t: float, xy: np.ndarray, team, track_id) -> None:
        pred = s.predict(t)
        new = self.smooth * xy + (1.0 - self.smooth) * pred
        hist = self._history.setdefault(s.id, [])
        hist.append((t, new.copy()))
        while len(hist) > 2 and t - hist[1][0] >= self.velocity_window:
            hist.pop(0)
        t0, p0 = hist[0]
        if t - t0 >= 0.15:
            v = (new - p0) / (t - t0)
            speed = np.linalg.norm(v)
            if speed > MAX_SPEED:
                v = v * (MAX_SPEED / speed)
            s.v = v
        s.xy, s.t, s.seen = new, t, t
        s.hits += 1
        s.track_id = track_id
        if team in (0, 1):
            s.team_votes[team] += 1
            s.team = int(np.argmax(s.team_votes))

    @property
    def players(self) -> list[Slot]:
        """Slots seen often enough to be a person rather than a flicker."""
        return [s for s in self.slots if s.hits >= self.min_hits]


class SnapDetector:
    """The snap is when a still formation starts moving: most players' speed jumps at once.

    By the time most of them are clearly moving the snap is a few tenths of a second old, so
    once it is sure the detector walks back to where the typical speed first left its resting
    level, and calls that the snap.
    """

    def __init__(self, speed: float = 1.6, share: float = 0.4, hold: float = 0.12, min_players: int = 8, settle: float = 0.6,
                 rise: float = 0.2, look_back: float = 1.2):
        self.speed, self.share, self.hold = speed, share, hold
        self.min_players, self.settle = min_players, settle
        self.rise, self.look_back = rise, look_back
        self.reset()

    def reset(self) -> None:
        self.snap_t: float | None = None
        self.seen_t: float | None = None          # when the detector became sure
        self._since: float | None = None
        self._start: float | None = None
        self._still_since: float | None = None
        self._typical: list[tuple[float, float]] = []      # (time, median speed)

    def _walk_back(self, t: float) -> float:
        recent = [(a, m) for a, m in self._typical if t - a <= self.look_back + 1.5]
        resting = [m for a, m in recent if t - a > self.look_back]
        level = (float(np.median(resting)) if resting else 0.3) + self.rise
        snap = self._since
        for a, m in reversed(recent):
            if a > self._since:
                continue
            if t - a > self.look_back or m <= level:
                break
            snap = a
        return snap

    def update(self, t: float, slots: list[Slot]) -> float | None:
        """Returns the snap time once found (and keeps returning it)."""
        if self.snap_t is not None:
            return self.snap_t
        if self._start is None:
            self._start = t
        speeds = np.array([s.speed for s in slots if t - s.seen < 0.3 and s.hits >= 5])
        if len(speeds) < self.min_players:
            self._since = None
            return None
        self._typical.append((t, float(np.median(speeds))))
        while t - self._typical[0][0] > self.look_back + 1.5:
            self._typical.pop(0)
        moving = float((speeds > self.speed).mean())
        if moving < 0.2:
            if self._still_since is None:
                self._still_since = t
        if moving >= self.share and self._still_since is not None and t - self._still_since >= self.settle:
            if self._since is None:
                self._since = t
            elif t - self._since >= self.hold:
                self.seen_t = t
                self.snap_t = self._walk_back(t)
        else:
            self._since = None
        return self.snap_t
