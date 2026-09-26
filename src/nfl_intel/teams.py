"""Split players into two teams by jersey colour.

Each box gives a torso crop; grass pixels are masked out and the mean Lab colour is the
feature. Features from the first few frames are clustered with k-means (k=2) and later
frames are assigned to the nearest centre. Votes are accumulated per track so a single
bad frame can't flip a player's team.
"""

from __future__ import annotations

import cv2
import numpy as np


def torso_crop(frame: np.ndarray, box) -> np.ndarray:
    """Upper-middle part of the box: jersey, not helmet, pants or grass."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cx2 = int(x1 + 0.25 * w), int(x1 + 0.75 * w)
    cy1, cy2 = int(y1 + 0.15 * h), int(y1 + 0.50 * h)
    H, W = frame.shape[:2]
    cx1, cx2 = max(0, cx1), min(W, cx2)
    cy1, cy2 = max(0, cy1), min(H, cy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return frame[0:0, 0:0]
    return frame[cy1:cy2, cx1:cx2]


def jersey_feature(crop: np.ndarray) -> np.ndarray | None:
    """Mean Lab colour of the non-grass pixels, or None for an empty crop."""
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 50)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    keep = ~grass.reshape(-1)
    pixels = lab[keep] if keep.sum() >= 10 else lab
    return pixels.mean(axis=0)


def kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain k-means with k-means++ seeding. Returns (centres, labels)."""
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(X)
    centres = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centres.append(X[rng.choice(n, p=probs)])
    C = np.array(centres)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        labels = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C, labels


class TeamClassifier:
    def __init__(self, k: int = 2, seed: int = 0):
        self.k = k
        self.seed = seed
        self.centres: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.centres is not None

    def fit(self, feats: np.ndarray) -> None:
        C, _ = kmeans(np.asarray(feats), self.k, seed=self.seed)
        # deterministic label order: team 0 has the darker jersey
        self.centres = C[np.argsort(C[:, 0])]

    def predict(self, feats: np.ndarray) -> np.ndarray:
        if self.centres is None:
            raise RuntimeError("TeamClassifier is not fitted")
        X = np.asarray(feats, dtype=np.float64).reshape(-1, self.centres.shape[1])
        return np.argmin(((X[:, None, :] - self.centres[None]) ** 2).sum(-1), axis=1)


class TeamMemory:
    """Per-track vote counts so a team label sticks once it's been seen a few times."""

    def __init__(self, k: int = 2):
        self.k = k
        self.votes: dict[int, np.ndarray] = {}

    def update(self, track_id: int, label: int) -> int:
        v = self.votes.setdefault(track_id, np.zeros(self.k))
        v[label] += 1
        return int(np.argmax(v))

    def current(self, track_id: int) -> int | None:
        v = self.votes.get(track_id)
        return None if v is None or v.sum() == 0 else int(np.argmax(v))


class TeamAssigner:
    """Collects jersey features during a warmup, fits the clusters, then labels tracks."""

    def __init__(self, warmup_frames: int = 15, min_samples: int = 20, k: int = 2):
        self.warmup_frames = warmup_frames
        self.min_samples = min_samples
        self.classifier = TeamClassifier(k)
        self.memory = TeamMemory(k)
        self._buffer: list[np.ndarray] = []
        self._frames = 0

    def reset(self) -> None:
        self.__init__(self.warmup_frames, self.min_samples, self.classifier.k)

    def assign(self, frame: np.ndarray, boxes, track_ids) -> list[int | None]:
        feats = [jersey_feature(torso_crop(frame, b)) for b in boxes]
        if not self.classifier.fitted:
            self._buffer.extend(f for f in feats if f is not None)
            self._frames += 1
            if self._frames >= self.warmup_frames and len(self._buffer) >= self.min_samples:
                self.classifier.fit(np.array(self._buffer))
                self._buffer.clear()
            else:
                return [None] * len(feats)
        out: list[int | None] = []
        for f, tid in zip(feats, track_ids):
            if f is None:
                out.append(self.memory.current(int(tid)))
                continue
            label = int(self.classifier.predict(f[None])[0])
            out.append(self.memory.update(int(tid), label))
        return out
