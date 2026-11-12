"""Which team a box shows, taught by where the players lined up.

The jersey split in `teams.py` clusters torso colours into two groups with no labels. That is
enough to tell the sides apart before the snap. After it the split is wrong about a third of
the time on film where one team wears dark jerseys with large white numbers, and it carries a
vote per image track, which stays with the track when the track slides from a blocker to the
man he is blocking.

Before the snap the two teams stand on opposite sides of the ball, so every box is a labelled
picture of one team's kit: helmet, jersey and pants. A small classifier fitted on those boxes
at the snap then reads any later box, whatever the image tracker calls it. On a player who
stays clear of the pile it agrees with his side on practically every frame; where it
disagrees for a stretch, the box has moved to a man of the other team.
"""

from __future__ import annotations

import cv2
import numpy as np

from .teams import DEFAULT_GRASS_LAB

SIZE = (24, 48)                                           # width, height the box is shrunk to
BANDS = ((0.03, 0.22), (0.20, 0.50), (0.50, 0.85))        # helmet, jersey, pants, as shares of the box's height
BINS = 6


def kit_feature(frame: np.ndarray, box, grass_lab: np.ndarray | None = None, tolerance: float = 28.0) -> np.ndarray | None:
    """What the player in the box wears: for each band, how the pixels that aren't grass are
    spread over colour (Lab a and b together) and over lightness. None for a box too small."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 8:
        return None
    H, W = frame.shape[:2]
    a, b = int(round(max(0.0, x1 - 0.1 * w))), int(round(min(W, x2 + 0.1 * w)))
    c, d = int(round(max(0.0, y1 - 0.05 * h))), int(round(min(H, y2 + 0.05 * h)))
    if b - a < 4 or d - c < 4:
        return None
    lab = cv2.cvtColor(cv2.resize(frame[c:d, a:b], SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float64)
    grass = DEFAULT_GRASS_LAB if grass_lab is None else np.asarray(grass_lab, dtype=np.float64)
    not_grass = np.linalg.norm(lab - grass[None, None], axis=2) > tolerance
    out = []
    for top, bottom in BANDS:
        rows = slice(int(top * SIZE[1]), int(bottom * SIZE[1]))
        band, keep = lab[rows, 4:-4].reshape(-1, 3), not_grass[rows, 4:-4].reshape(-1)       # the middle: less of the neighbours
        pixels = band[keep] if keep.sum() >= 8 else band
        colour, _, _ = np.histogram2d(pixels[:, 1], pixels[:, 2], bins=BINS, range=[[88, 178], [88, 178]])
        light, _ = np.histogram(pixels[:, 0], bins=BINS, range=(0, 255))
        v = np.concatenate([colour.ravel(), light]).astype(np.float64)
        out.append(np.sqrt(v / max(1.0, v.sum())))
    return np.concatenate(out)


class KitReader:
    """Fitted once a play, on the boxes of the players lined up before the snap."""

    def __init__(self, sure: float = 0.9, min_each: int = 40):
        self.sure = sure                  # a reading less certain than this is no reading
        self.min_each = min_each          # boxes needed from each side before it will fit
        self.model = None

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def fit(self, features: list[np.ndarray], sides: list[int]) -> bool:
        """`sides` is 0 for a man on the offense's side of the ball and 1 for the defense's."""
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(sides, dtype=int)
        if min(int((y == 0).sum()), int((y == 1).sum())) < self.min_each:
            return False
        self.model = LogisticRegression(C=1.0, max_iter=500).fit(np.asarray(features), y)
        return True

    def read(self, features: list[np.ndarray | None]) -> list[int | None]:
        """The side each box's kit belongs to, None where there is no box to read or it isn't clear."""
        out: list[int | None] = [None] * len(features)
        have = [i for i, f in enumerate(features) if f is not None]
        if self.model is None or not have:
            return out
        p = self.model.predict_proba(np.asarray([features[i] for i in have]))[:, 1]
        for i, q in zip(have, p):
            if q >= self.sure:
                out[i] = 1
            elif q <= 1.0 - self.sure:
                out[i] = 0
        return out
