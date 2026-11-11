"""Track players on the field, in yards, instead of in the picture.

In the picture a player's box jumps around as the camera pans and as bodies overlap, and the
image tracker hands out a new number every time it loses someone. On the field, positions
are smooth and nobody moves faster than about 11 yards a second, so matching each frame's
detections to the players we already know is a small assignment problem.

Each player is a slot: a smoothed position, a velocity, a team, and (after the snap) a role.
Slots coast on their velocity for a short while when their player isn't seen.

Tight coverage is the hard case: a receiver and the man on him overlap in the picture and
the detector returns one box for the two of them. The one that went missing is then carried
along with the one still seen, at the offset he had, until the detector can tell them apart
again. Without that the defender would vanish and his receiver would look wide open.
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
    partner: int | None = None              # the slot this one is hidden behind, while the detector sees one box for both
    twin: int | None = None                 # the slot this one was born on top of: a second box on the same man, most likely
    offset: np.ndarray | None = None        # where this one was relative to the partner when last seen

    def predict(self, t: float) -> np.ndarray:
        return self.xy + self.v * (t - self.t)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.v))


class FieldTracker:
    def __init__(self, gate: float = 2.5, coast: float = 1.0, smooth: float = 0.45, velocity_window: float = 0.4,
                 team_penalty: float = 2.0, min_hits: int = 3, merge_within: float = 2.0, hidden_for: float = 3.0,
                 same_track: float = 2.0):
        self.double_box = 1.0                 # an unconfirmed detection this close to a confirmed one is the same man
        self.clear_of = 1.5                   # ...and one this far from every known player may start a new one
        self.twin_for = 1.5                   # seconds a slot born on top of another is watched as its double
        self.twin_gone = 0.0                  # the older one unseen this long while the newer is seen: one man, the newer slot
        self.same_track = same_track          # yards taken off the cost of keeping a detection with the slot that
                                              # had its image track last frame: an unbroken track is the same man
        self.gate = gate                      # a detection further than this from every slot starts a new one
        self.coast = coast                    # seconds a slot survives unseen
        self.merge_within = merge_within      # two opponents closer than this can share one box
        self.hidden_for = hidden_for          # seconds a slot survives hidden behind its partner
        self.partner_gap = 0.6                # a partner unseen this long is no longer a guide
        self.smooth = smooth                  # weight of the new measurement in the position
        self.velocity_window = velocity_window
        self.team_penalty = team_penalty      # yards added to the cost of matching across teams
        self.min_hits = min_hits
        self.slots: list[Slot] = []
        self._next = 0
        self._history: dict[int, list] = {}

    def reset(self) -> None:
        self.slots, self._next, self._history = [], 0, {}

    def update(self, t: float, xy: np.ndarray, teams: list, track_ids: list | None = None,
               confirmed: list[bool] | None = None) -> list[Slot | None]:
        """Match this frame's detections to slots. Returns, for each detection, its slot.

        `confirmed` says which detections the image tracker stands behind. The others (a box
        it has seen once or twice) can show where a known player is, and can start a new one
        only well clear of everybody already known: on a goal-line pile the image tracker
        confirms few boxes and these are real men. One that sits on a confirmed detection is
        dropped, and one beside a known player starts nothing: nearly all of those are a
        second box on the same man, and a slot made from one steals his number later. Their
        slot comes back as None.
        """
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        assigned: list[Slot | None] = [None] * len(xy)
        sure = np.ones(len(xy), bool) if confirmed is None else np.asarray(confirmed, bool)
        use = sure.copy()
        if (~sure).any() and sure.any():
            apart = np.linalg.norm(xy[~sure][:, None, :] - xy[sure][None, :, :], axis=2).min(axis=1)
            use[~sure] = apart > self.double_box
        elif (~sure).any():
            use[:] = True
        live = [s for s in self.slots if t - s.seen <= self._lifetime(s)]
        if live and len(xy):
            pred = np.array([s.predict(t) for s in live])
            cost = np.linalg.norm(pred[:, None, :] - xy[None, :, :], axis=2)
            cost[:, ~use] = 1e6
            for i, s in enumerate(live):
                allowed = self.gate + MAX_SPEED * 0.25 * min(1.0, t - s.seen)      # unseen slots may have drifted
                for j in range(len(xy)):
                    if cost[i, j] > allowed or not use[j]:
                        cost[i, j] = 1e6
                    elif s.team is not None and teams[j] is not None and teams[j] != s.team:
                        cost[i, j] += self.team_penalty
            if track_ids is not None and self.same_track:
                held = {s.track_id: i for i, s in enumerate(live) if s.track_id is not None and t - s.seen <= 0.5}
                for j, tid in enumerate(track_ids):
                    i = held.get(tid) if tid is not None else None
                    if i is not None and cost[i, j] < 1e5:
                        cost[i, j] -= self.same_track
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e5 - 100:
                    continue
                self._absorb(live[i], t, xy[j], teams[j], None if track_ids is None else track_ids[j])
                assigned[j] = live[i]
        taken = [s for s in assigned if s is not None]
        for j in range(len(xy)):
            alone = use[j] and all(float(np.linalg.norm(o.xy - xy[j])) > self.clear_of for o in live)
            if assigned[j] is None and (sure[j] or alone):
                s = Slot(self._next, xy[j].copy(), np.zeros(2), teams[j], t, t, t,
                         track_id=None if track_ids is None else track_ids[j])
                on = min(taken, key=lambda o: float(np.linalg.norm(o.xy - xy[j])), default=None)
                if on is not None and float(np.linalg.norm(on.xy - xy[j])) <= self.double_box:
                    s.twin = on.id
                if teams[j] in (0, 1):
                    s.team_votes[teams[j]] += 1
                self._next += 1
                self._history[s.id] = [(t, xy[j].copy())]
                self.slots.append(s)
                assigned[j] = s
        self._settle_twins(t)
        by_id = {s.id: s for s in self.slots}
        for s in self.slots:                                   # unseen slots: hidden behind a partner, or coasting
            if s.t >= t or t - s.seen > self._lifetime(s):
                continue
            if s.partner is None and s.hits >= self.min_hits:
                s.partner, s.offset = self._hidden_behind(s, t)
            partner = by_id.get(s.partner) if s.partner is not None else None
            if partner is not None and t - partner.seen <= self.partner_gap:
                # the partner may miss a frame or two himself; where he is thought to be is still the best guide
                where = partner.xy if partner.t >= t else partner.predict(t)
                s.xy, s.v, s.t = where + s.offset, partner.v.copy(), t
                continue
            if partner is not None:
                s.partner, s.offset = None, None               # the partner is gone too: nothing to follow
            s.xy = s.predict(t)
            s.v = s.v * 0.9
            s.t = t
        self.slots = [s for s in self.slots if t - s.seen <= self._lifetime(s)]
        for sid in [k for k in self._history if all(s.id != k for s in self.slots)]:
            del self._history[sid]
        return assigned

    def _settle_twins(self, t: float) -> None:
        """A slot born on top of another is usually the image tracker starting a second track
        on the same man. If the older track then stops, the man is the new slot: it takes the
        old slot's number, and with it his role and name. If both are still seen after a
        second and a half, or have moved apart, they are two men after all."""
        by_id = {s.id: s for s in self.slots}
        for s in self.slots:
            if s.twin is None:
                continue
            old = by_id.get(s.twin)
            if old is None or t - s.born > self.twin_for or float(np.linalg.norm(old.xy - s.xy)) > 2.0 * self.double_box:
                s.twin = None
            elif s.seen == t and old.seen < t and t - old.seen >= self.twin_gone:
                s.id, old.id = old.id, s.id
                s.role, s.label, old.role, old.label = old.role, old.label, None, None
                s.hits, s.team_votes, s.team = s.hits + old.hits, old.team_votes, old.team if old.team is not None else s.team
                self._history[s.id], self._history[old.id] = self._history.get(old.id, []), self._history.get(s.id, [])
                old.seen = t - 10.0 * self.hidden_for            # the stand-in is done
                s.twin = None

    def _lifetime(self, s: Slot) -> float:
        return self.hidden_for if s.partner is not None else self.coast

    def _hidden_behind(self, s: Slot, t: float) -> tuple[int | None, np.ndarray | None]:
        """The player this slot has just disappeared behind, if there is one close enough.
        Usually the man covering him or the man he covers, but a jersey read wrongly must not
        stop it, so a teammate will do."""
        best, near = None, self.merge_within
        for o in self.slots:
            if o is s or o.seen != t or o.hits < self.min_hits:
                continue
            d = float(np.linalg.norm(o.xy - s.xy))
            if d < near:
                best, near = o, d
        return (None, None) if best is None else (best.id, s.xy - best.xy)

    def _absorb(self, s: Slot, t: float, xy: np.ndarray, team, track_id) -> None:
        s.partner, s.offset = None, None
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
        if track_id is not None:
            for o in self.slots:                               # an image track belongs to one slot at a time
                if o is not s and o.track_id == track_id:
                    o.track_id = None
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
                 rise: float = 0.2, look_back: float = 1.2, lead: float = 0.12):
        self.lead = lead          # checked by eye on film: the walk-back still lands this long after the ball moves
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
                self.snap_t = self._walk_back(t) - self.lead
        else:
            self._since = None
        return self.snap_t
