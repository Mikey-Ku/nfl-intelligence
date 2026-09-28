"""Pass value: how good is the best throw available right now?

For every eligible receiver, at every frame, two models estimate what would happen if the
ball were thrown to him now: the chance it is caught and the yards gained if it is. Their
product is that receiver's expected yards; the play value is the best receiver's. The
plus/minus is how much that value has moved since the snap.

The models are trained on Big Data Bowl throws (features at the moment of the real throw to
the real target, outcome from play-by-play) and then applied to every receiver at every
frame. That is a counterfactual: quarterbacks throw to open receivers, so covered throws are
rarer in training than on film. Treat the numbers as comparisons within a play, not as
exact probabilities.

Everything is computed from positions alone (velocity is a difference of positions), so the
same code runs on sensor data and on video-derived tracking once it is in yards.
"""

from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .field import FIELD_WIDTH

FEATURES = [
    "sep1", "sep2", "n_def_3", "n_def_6", "depth", "dist_qb", "lateral", "sideline", "to_goal",
    "speed", "vx", "closing", "leverage", "def_depth_diff", "t", "qb_speed", "qb_depth",
]
VELOCITY_LAG = 0.4      # seconds of position history used for velocity
DEFAULT_MODEL = Path("models/pass_value.pkl")


def velocities(xy: np.ndarray, hz: float, lag: float = VELOCITY_LAG) -> np.ndarray:
    """(frames, players, 2) positions -> velocities in yd/s from a lagged difference."""
    k = max(1, int(round(lag * hz)))
    v = np.zeros_like(xy, dtype=np.float64)
    for t in range(1, len(xy)):
        j = max(0, t - k)
        v[t] = (xy[t] - xy[j]) / ((t - j) / hz)
    return v


def receiver_features(xy: np.ndarray, v: np.ndarray, rec: int, qb: int, defenders: list[int], los: float, t: float) -> np.ndarray:
    """Feature vector for one receiver at one frame. xy, v are (players, 2)."""
    p, pv = xy[rec], v[rec]
    q, qv = xy[qb], v[qb]
    if defenders:
        d = xy[defenders]
        dist = np.linalg.norm(d - p, axis=1)
        order = np.argsort(dist)
        sep1 = float(dist[order[0]])
        sep2 = float(dist[order[1]]) if len(order) > 1 else 30.0
        near = defenders[int(order[0])]
        rel_p = xy[near] - p
        rel_v = v[near] - pv
        closing = float(-(rel_p @ rel_v) / max(sep1, 0.5))
        to_qb = q - p
        leverage = float((rel_p @ to_qb) / (max(sep1, 0.5) * max(np.linalg.norm(to_qb), 0.5)))
        def_depth_diff = float(xy[near][0] - p[0])
        n3, n6 = int((dist < 3).sum()), int((dist < 6).sum())
    else:
        sep1 = sep2 = 30.0
        closing = leverage = def_depth_diff = 0.0
        n3 = n6 = 0
    return np.array([
        sep1, sep2, n3, n6,
        p[0] - los,
        float(np.linalg.norm(p - q)),
        abs(p[1] - q[1]),
        min(p[1], FIELD_WIDTH - p[1]),
        110.0 - p[0],
        float(np.linalg.norm(pv)),
        pv[0],
        closing, leverage, def_depth_diff,
        t,
        float(np.linalg.norm(qv)),
        los - q[0],
    ], dtype=np.float64)


@dataclass
class ReceiverScore:
    index: int                # player index (or track id at run time)
    p_catch: float
    yards_if_caught: float
    expected_yards: float
    separation: float
    label: str = ""
    depth: float = 0.0        # yards past the line of scrimmage


class PassValueModel:
    def __init__(self, catch, yards, meta: dict | None = None):
        self.catch = catch          # classifier: P(complete)
        self.yards = yards          # regressor: yards gained given complete
        self.meta = meta or {}

    def save(self, path: str | Path = DEFAULT_MODEL) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"catch": self.catch, "yards": self.yards, "meta": self.meta, "features": FEATURES}, f)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL) -> PassValueModel:
        with Path(path).open("rb") as f:
            d = pickle.load(f)
        if d.get("features") != FEATURES:
            raise ValueError("model was trained with a different feature list; retrain with `nfl value-train`")
        return cls(d["catch"], d["yards"], d.get("meta"))

    def typical(self, clock: float, hz: float = 10.0, to_goal: float | None = None) -> float:
        """What a typical play is worth this long after the snap (0 if the model has no curve).

        With `to_goal`, the yards from the line of scrimmage to the goal line, the comparison
        is with plays from the same stretch of field."""
        curve = self.meta.get("baseline") or []
        by_goal = self.meta.get("baseline_by_goal")
        if to_goal is not None and by_goal:
            edges = by_goal["edges"]
            curve = by_goal["curves"][sum(1 for e in edges if to_goal >= e)]
        if not curve:
            return 0.0
        return float(curve[min(len(curve) - 1, max(0, int(round(clock * hz))))])

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(X)
        p = self.catch.predict_proba(X)[:, 1]
        y = np.clip(self.yards.predict(X), -5.0, 99.0)
        return p, y

    def score_frame(self, xy, v, receivers, qb, defenders, los, t, labels=None) -> list[ReceiverScore]:
        if qb < 0 or not receivers:
            return []
        X = np.stack([receiver_features(xy, v, r, qb, defenders, los, t) for r in receivers])
        p, y = self.score(X)
        return [
            ReceiverScore(int(r), float(p[i]), float(y[i]), float(p[i] * y[i]), float(X[i, 0]),
                          labels[r] if labels else "", float(X[i, 4]))
            for i, r in enumerate(receivers)
        ]


def play_value(scores: list[ReceiverScore]) -> tuple[float, ReceiverScore | None]:
    """Expected yards of the best throw available, and which receiver that is."""
    if not scores:
        return 0.0, None
    best = max(scores, key=lambda s: s.expected_yards)
    return max(0.0, best.expected_yards), best


class Kinematics:
    """Velocities for live tracks: smoothed positions, differenced over VELOCITY_LAG seconds."""

    def __init__(self, lag: float = VELOCITY_LAG, smooth: float = 0.5):
        self.lag = lag
        self.smooth = smooth            # weight of the newest position in the running average
        self._hist: dict[int, deque] = {}

    def update(self, track_id: int, t: float, xy) -> tuple[np.ndarray, np.ndarray]:
        """Returns (smoothed position, velocity) for this track."""
        h = self._hist.setdefault(track_id, deque(maxlen=64))
        p = np.asarray(xy, dtype=np.float64)
        if h:
            p = self.smooth * p + (1 - self.smooth) * h[-1][1]
        h.append((t, p))
        while len(h) > 2 and t - h[1][0] >= self.lag:
            h.popleft()
        t0, p0 = h[0]
        v = (p - p0) / (t - t0) if t > t0 else np.zeros(2)
        return p, v

    def forget(self, keep: set[int]) -> None:
        for tid in [k for k in self._hist if k not in keep]:
            del self._hist[tid]
