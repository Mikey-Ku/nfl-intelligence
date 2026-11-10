"""Player looks: knowing a player on a later play by what he looks like.

A jersey number is read on a few frames of a few plays. Once it has been, the player's look
(build, skin, sleeves, gloves, socks, shoes, visor) is worth remembering: on the next play
the same man can be named from his look alone, number or no number. Looks are kept per team
and number for the whole game in a `LookBook`, and only ever compared against the players
the roster says are on the field, on his side, in his kind of role: a handful of candidates,
not 53.

The look is the ImageNet ResNet-18's view of the player's body in three horizontal parts
(head and shoulders, torso and arms, legs and feet), fine-tuned on our own film
by `looks_train.py` (`models/looks.pt`; without it the pretrained features are used and
hardly ever commit). Marked on plays it had not seen: the pretrained features picked the
right man among his on-field teammates 60% of the time; fine-tuned on the other clip of the
same game, 89%, and 98% of the ones it was sure enough to name (42 of 55). A number read
always wins over a look.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .jersey import MEAN, STD

BODY = (128, 256)               # width, height of the body crop the network sees
DEFAULT_MODEL = Path("models/looks.pt")
DIM = 512 * 3


def body_crop(frame: np.ndarray, box, size: tuple[int, int] = BODY) -> np.ndarray | None:
    """The whole player, a little room around him, resized to `size` (BGR)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 8:
        return None
    H, W = frame.shape[:2]
    a, b = int(round(max(0.0, x1 - 0.1 * w))), int(round(min(W, x2 + 0.1 * w)))
    c, d = int(round(max(0.0, y1 - 0.05 * h))), int(round(min(H, y2 + 0.05 * h)))
    if b - a < 4 or d - c < 4:
        return None
    return cv2.resize(frame[c:d, a:b], size, interpolation=cv2.INTER_AREA)


def normalised(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


class LookEmbedder:
    """Turns body crops into unit vectors that are close for the same man."""

    def __init__(self, weights: str | Path | None = DEFAULT_MODEL, device: str | None = None):
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        pretrained = weights is None or not Path(weights).exists()
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc = torch.nn.Identity()
        if not pretrained:
            self.net.load_state_dict(torch.load(str(weights), map_location="cpu")["net"])
        self.fine_tuned = not pretrained
        self.net.to(self.device).eval()

    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, DIM))
        torch = self.torch
        x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(((x - MEAN) / STD).transpose(0, 3, 1, 2))).to(self.device)
        m = self.net
        with torch.no_grad():
            h = m.layer4(m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x))))))))     # (N, 512, 8, 4)
            f = torch.cat([h[:, :, :3].mean((2, 3)), h[:, :, 3:6].mean((2, 3)), h[:, :, 6:].mean((2, 3))], 1)
        return normalised(f.cpu().numpy())

    def embed(self, frame: np.ndarray, boxes) -> tuple[np.ndarray, np.ndarray]:
        """(kept, looks): the boxes big enough to crop, and their unit vectors."""
        crops, kept = [], []
        for i, box in enumerate(boxes):
            c = body_crop(frame, box)
            if c is not None:
                crops.append(c)
                kept.append(i)
        return np.array(kept, dtype=int), self.embed_crops(crops)


@dataclass
class LookBook:
    """What each player has looked like so far, by team and number: a running mean."""

    looks: dict = field(default_factory=dict)        # (team, number) -> [sum vector, count]

    def add(self, team: str, number: int, looks: np.ndarray) -> None:
        looks = np.atleast_2d(np.asarray(looks, dtype=np.float64))
        if len(looks) == 0:
            return
        entry = self.looks.setdefault((team, int(number)), [np.zeros(looks.shape[1]), 0])
        entry[0] += looks.sum(0)
        entry[1] += len(looks)

    def known(self, team: str) -> set[int]:
        return {n for t, n in self.looks if t == team}

    def match(self, team: str, numbers, look: np.ndarray) -> list[tuple[int, float]]:
        """Candidates on `numbers` this book knows, most alike first: (number, similarity)."""
        look = normalised(look)
        out = []
        for n in numbers:
            e = self.looks.get((team, int(n)))
            if e is not None and e[1] > 0:
                out.append((int(n), float(normalised(e[0]) @ look)))
        return sorted(out, key=lambda c: -c[1])

    def recognise(self, team: str, numbers, look: np.ndarray, min_sim: float = 0.85, min_gap: float = 0.10) -> int | None:
        """The one candidate this look is clearly, or None: he must look like the man
        (`min_sim`) and be `min_gap` ahead of the runner-up. With fewer than two candidates
        known there is no runner-up to beat, so nobody is named. The defaults come from a
        fine-tuned embedding, where the same man scores about 0.93 and his closest teammate
        about 0.79; the pretrained one puts everybody above 0.9, and these rules keep it quiet."""
        ranked = self.match(team, numbers, look)
        if len(ranked) < 2 or ranked[0][1] < min_sim or ranked[0][1] - ranked[1][1] < min_gap:
            return None
        return ranked[0][0]

    def recognise_many(self, team: str, numbers, looks: np.ndarray, min_sim: float = 0.88, min_gap: float = 0.12,
                       min_frames: int = 8, min_share: float = 0.7) -> int | None:
        """`recognise` frame by frame, then a vote: the man most frames were sure of, if that is
        at least `min_frames` frames and `min_share` of the frames that were sure of anyone.
        A track that drifted onto another player for a while is outvoted rather than averaged in."""
        looks = np.atleast_2d(np.asarray(looks, dtype=np.float64))
        votes: dict[int, int] = {}
        for look in looks:
            n = self.recognise(team, numbers, look, min_sim, min_gap)
            if n is not None:
                votes[n] = votes.get(n, 0) + 1
        if not votes:
            return None
        best = max(votes, key=votes.get)
        if votes[best] < min_frames or votes[best] < min_share * sum(votes.values()):
            return None
        return best

    def save(self, path: str | Path) -> None:
        keys = sorted(self.looks)
        np.savez(str(path), teams=np.array([k[0] for k in keys]), numbers=np.array([k[1] for k in keys], dtype=np.int64),
                 sums=np.stack([self.looks[k][0] for k in keys]) if keys else np.zeros((0, DIM)),
                 counts=np.array([self.looks[k][1] for k in keys], dtype=np.int64))

    @classmethod
    def load(cls, path: str | Path) -> LookBook:
        z = np.load(str(path), allow_pickle=False)
        book = cls()
        for t, n, s, c in zip(z["teams"], z["numbers"], z["sums"], z["counts"]):
            book.looks[(str(t), int(n))] = [np.asarray(s, dtype=np.float64), int(c)]
        return book


class ShotLooks:
    """Each tracked player's looks through one shot: the running mean, and the last `keep`
    frames one by one for the vote."""

    def __init__(self, embedder, every: int = 3, keep: int = 60):
        self.embedder = embedder
        self.every = max(1, every)
        self.keep = keep
        self.sums: dict = {}
        self.counts: dict = {}
        self.recent: dict = {}
        self._frames = 0

    def update(self, frame: np.ndarray, boxes, ids) -> None:
        self._frames += 1
        if (self._frames - 1) % self.every or len(boxes) == 0:
            return
        kept, looks = self.embedder.embed(frame, boxes)
        for i, e in zip(kept, looks):
            tid = ids[i]
            if tid in self.sums:
                self.sums[tid] += e
                self.counts[tid] += 1
            else:
                self.sums[tid] = e.astype(np.float64).copy()
                self.counts[tid] = 1
                self.recent[tid] = deque(maxlen=self.keep)
            self.recent[tid].append(e.astype(np.float32))

    def look(self, tid, min_frames: int = 3) -> np.ndarray | None:
        if self.counts.get(tid, 0) < min_frames:
            return None
        return normalised(self.sums[tid])

    def looks(self, tid, min_frames: int = 3) -> np.ndarray | None:
        """The recent looks one by one, (n, DIM), or None with too few."""
        if self.counts.get(tid, 0) < min_frames:
            return None
        return np.stack(self.recent[tid])

    def move(self, old, new) -> None:
        """The tracker decided `old` and `new` are the same man."""
        if old in self.sums:
            if new in self.sums:
                self.sums[new] += self.sums.pop(old)
                self.counts[new] += self.counts.pop(old)
                for e in self.recent.pop(old):
                    self.recent[new].append(e)
            else:
                self.sums[new], self.counts[new] = self.sums.pop(old), self.counts.pop(old)
                self.recent[new] = self.recent.pop(old)
