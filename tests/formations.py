"""Synthetic plays for the tests, in the absolute frame: offense moving +x, line of scrimmage at LOS."""

import numpy as np

from nfl_intel.fieldtrack import Slot
from nfl_intel.schema import FrameResult, TrackedPlayer

LOS = 95.0
BALL = 26.0

# (role, x from the line, y). Offense is team 0, defense team 1.
SHOTGUN = [
    ("QB", -4.5, BALL), ("RB", -4.5, BALL - 2.0),
    ("OL", -1.4, BALL - 3.2), ("OL", -1.4, BALL - 1.6), ("OL", -1.3, BALL), ("OL", -1.4, BALL + 1.6), ("OL", -1.4, BALL + 3.2),
    ("TE", -1.5, BALL + 5.0), ("WR", -1.0, 46.0), ("WR", -1.5, 40.0), ("WR", -1.0, 8.0),
    ("DL", 0.8, BALL - 3.5), ("DL", 0.8, BALL - 1.0), ("DL", 0.8, BALL + 1.5), ("DL", 0.9, BALL + 5.5),
    ("LB", 4.0, BALL - 2.0), ("LB", 4.5, BALL + 3.0),
    ("CB", 5.0, 46.0), ("CB", 3.0, 39.5), ("CB", 6.0, 8.5), ("S", 12.0, 20.0), ("S", 12.0, 33.0),
]
OFFENSE = {"QB", "RB", "OL", "TE", "WR"}


def slots(formation=SHOTGUN, drop=(), teams=None):
    """Slots and absolute positions for a formation, leaving out the players in `drop`."""
    out, xy = [], []
    for i, (role, dx, y) in enumerate(formation):
        if i in drop:
            continue
        team = 0 if role in OFFENSE else 1
        if teams and i in teams:
            team = teams[i]
        out.append(Slot(i, np.array([LOS + dx, y]), np.zeros(2), team, 0.0, 0.0, 0.0, hits=10))
        xy.append((LOS + dx, y))
    return out, np.array(xy)


def film(direction=-1, los_rel=40.0, fps=20.0, still=1.6, run=2.4, drop=(3, 4, 5), seed=0, jitter=0.05, size=(1920, 1080)):
    """Frames of a pass play as the pipeline would hand them over: positions in the mapping's
    own yards (offense moving `direction`), a still formation and then the snap."""
    rng = np.random.default_rng(seed)
    frames = []
    n_still, n_run = int(still * fps), int(run * fps)
    for f in range(n_still + n_run):
        t = f / fps
        s = max(0.0, t - still)
        players = []
        for i, (role, dx, y) in enumerate(SHOTGUN):
            if i in drop:
                continue
            x = LOS + dx
            if role == "WR":
                x += 6.5 * s
            elif role == "TE":
                x += 5.0 * s
            elif role == "RB":
                y -= 4.0 * s
            elif role == "QB":
                x -= 2.5 * min(s, 1.2)
            elif role == "DL":
                x -= 2.0 * s
            elif role in ("CB", "S"):
                x += 4.5 * s
            elif role == "LB":
                x += 2.0 * s
            x_rel = los_rel + direction * (x - LOS)
            y_rel = y if direction > 0 else 160.0 / 3.0 - y
            p = np.array([x_rel, y_rel]) + rng.normal(0, jitter, 2)
            players.append(TrackedPlayer(i, 0 if role in OFFENSE else 1, (10.0 * i, 0.0, 10.0 * i + 8, 20.0), 0.9,
                                         (10.0 * i + 4, 20.0), (float(p[0]), float(p[1]))))
        frames.append(FrameResult(f, t, size, players))
    return frames, still
