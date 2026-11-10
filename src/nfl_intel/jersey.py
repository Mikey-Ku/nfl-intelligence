"""Jersey numbers: which player a track is, read off his shirt.

NFL Pro's "On Field Players" list says which 22 numbers are on the field, so this is never
"read any number from 0 to 99". It is "which of these eleven is he", decided over a whole
shot: most frames show a player side-on and say nothing, a few show his chest or his back.

How it works:
  1. Cut the torso out of each tracked box (`torso_crop`).
  2. A small network (`NumberNet`: a pretrained ResNet-18 with two heads, tens digit and units
     digit) gives each crop a probability for every number, plus "nothing to read".
  3. `NumberVotes` adds up the confident frames per track.
  4. `assign` gives each roster number to at most one track, and only when the evidence is
     clear. A track nobody could read keeps its formation label instead of a guess.

General text readers were tried first (Apple Vision, PARSeq, TrOCR, PP-OCR). None read a
dark jersey, and the best named about two players a play; see `jersey_train.py` for how the
network is trained on our own film.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SIZE = 96                       # the network's input, pixels square
TORSO_TOP, TORSO_BOTTOM = 0.05, 0.62      # of the box height: under the helmet to the waist
TORSO_SIDE = 0.10               # extra width each side, of the box width
NONE = 10                       # "no tens digit" in the tens head; "nothing to read" in the units head
DEFAULT_MODEL = Path("models/jersey.pt")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def torso_crop(frame: np.ndarray, box, size: int = SIZE) -> np.ndarray | None:
    """The part of a player's box where the number is, resized to `size` square (BGR)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 8:
        return None
    H, W = frame.shape[:2]
    a, b = int(round(max(0.0, x1 - TORSO_SIDE * w))), int(round(min(W, x2 + TORSO_SIDE * w)))
    c, d = int(round(max(0.0, y1 + TORSO_TOP * h))), int(round(min(H, y1 + TORSO_BOTTOM * h)))
    if b - a < 4 or d - c < 4:
        return None
    return cv2.resize(frame[c:d, a:b], (size, size), interpolation=cv2.INTER_CUBIC)


def digits_of(number: int) -> tuple[int, int]:
    """(tens class, units class) for a jersey number. 7 is (NONE, 7); 70 is (7, 0)."""
    if not 0 <= number <= 99:
        raise ValueError(f"not a jersey number: {number}")
    return (NONE, number) if number < 10 else (number // 10, number % 10)


def number_probs(tens: np.ndarray, units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Head probabilities (N, 11) each -> (N, 100) for numbers 0-99 and (N,) for "nothing to read"."""
    tens, units = np.asarray(tens, dtype=np.float64), np.asarray(units, dtype=np.float64)
    out = np.zeros((len(tens), 100))
    out[:, :10] = tens[:, NONE:NONE + 1] * units[:, :10]
    for t in range(1, 10):
        out[:, 10 * t:10 * t + 10] = tens[:, t:t + 1] * units[:, :10]
    return out, units[:, NONE]


def to_input(crops: list[np.ndarray]) -> np.ndarray:
    """BGR crops -> the float array the network takes (N, 3, SIZE, SIZE)."""
    x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32) / 255.0
    return np.ascontiguousarray(((x - MEAN) / STD).transpose(0, 3, 1, 2))


def build_net(pretrained: bool = False):
    """ResNet-18 with a tens head and a units head. `pretrained` starts from ImageNet weights."""
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    class NumberNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            width = self.body.fc.in_features
            self.body.fc = nn.Identity()
            self.drop = nn.Dropout(0.2)
            self.tens = nn.Linear(width, 11)
            self.units = nn.Linear(width, 11)

        def forward(self, x):
            f = self.drop(self.body(x))
            return self.tens(f), self.units(f)

    return NumberNet()


class NumberReader:
    """Loads the trained network and reads crops."""

    def __init__(self, weights: str | Path = DEFAULT_MODEL, device: str | None = None):
        import torch

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.net = build_net()
        self.net.load_state_dict(torch.load(str(weights), map_location="cpu")["net"])
        self.net.to(self.device).eval()

    def read_crops(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """(N, 100) number probabilities and (N,) "nothing to read"."""
        if not crops:
            return np.zeros((0, 100)), np.zeros(0)
        torch = self.torch
        with torch.no_grad():
            t, u = self.net(torch.from_numpy(to_input(crops)).to(self.device))
            t, u = torch.softmax(t, 1).cpu().numpy(), torch.softmax(u, 1).cpu().numpy()
        return number_probs(t, u)

    def read(self, frame: np.ndarray, boxes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reads every box in a frame. Returns (kept, probs, blank): `kept` indexes the boxes
        big enough to cut a torso from."""
        crops, kept = [], []
        for i, box in enumerate(boxes):
            c = torso_crop(frame, box)
            if c is not None:
                crops.append(c)
                kept.append(i)
        probs, blank = self.read_crops(crops)
        return np.array(kept, dtype=int), probs, blank


@dataclass
class NumberVotes:
    """Adds up what the reader says about each track over a shot.

    A frame counts only when the reader is sure (`min_conf`), and counts for what it is sure
    of. Side-on frames, where it isn't, add nothing."""

    min_conf: float = 0.6
    score: dict = field(default_factory=lambda: defaultdict(lambda: np.zeros(100)))
    frames: dict = field(default_factory=lambda: defaultdict(int))

    def add(self, track_ids, probs: np.ndarray) -> None:
        for tid, p in zip(track_ids, probs):
            self.frames[tid] += 1
            if float(p.max()) >= self.min_conf:
                self.score[tid] += p

    def best(self, tid, among=None) -> tuple[int, float, float]:
        """(number, its score, the runner-up's score) for one track, optionally among a roster."""
        s = self.score.get(tid)
        if s is None:
            return -1, 0.0, 0.0
        idx = np.arange(100) if among is None else np.array(sorted(set(among)), dtype=int)
        if len(idx) == 0:
            return -1, 0.0, 0.0
        order = idx[np.argsort(s[idx])[::-1]]
        return int(order[0]), float(s[order[0]]), float(s[order[1]]) if len(order) > 1 else 0.0


def assign(votes: NumberVotes, sides: dict, rosters: dict, min_score: float = 2.5, min_ratio: float = 2.0) -> dict:
    """Roster numbers for tracks: {track id: number}.

    `sides`: {track id: "offense" | "defense"}; `rosters`: {"offense": [numbers], "defense": [...]}.
    Within a side each number goes to at most one track (the pairing with the most evidence
    overall), and only where the track's score for it is at least `min_score` (about three
    sure frames) and `min_ratio` times its score for any other number on that roster."""
    from scipy.optimize import linear_sum_assignment

    out: dict = {}
    for side, numbers in rosters.items():
        numbers = sorted(set(int(n) for n in numbers))
        tids = [t for t, s in sides.items() if s == side and t in votes.score]
        if not numbers or not tids:
            continue
        S = np.array([[votes.score[t][n] for n in numbers] for t in tids])
        rows, cols = linear_sum_assignment(-S)
        for r, c in zip(rows, cols):
            others = np.delete(S[r], c)
            runner_up = float(others.max()) if len(others) else 0.0
            if S[r, c] >= min_score and S[r, c] >= min_ratio * runner_up:
                out[tids[r]] = numbers[c]
    return out


class ShotNumbers:
    """Reads numbers through one shot and says who is who. The piece the live pipeline calls.

    `update` every frame (or every few) with the tracked boxes; `names` whenever names are
    wanted. Ids are whatever the caller tracks players by. Start a new one at each cut."""

    def __init__(self, reader, every: int = 3, min_conf: float = 0.6):
        self.reader = reader
        self.every = max(1, every)
        self.votes = NumberVotes(min_conf=min_conf)
        self._frames = 0

    def update(self, frame: np.ndarray, boxes, ids) -> None:
        self._frames += 1
        if (self._frames - 1) % self.every or len(boxes) == 0:
            return
        kept, probs, _ = self.reader.read(frame, boxes)
        self.votes.add([ids[i] for i in kept], probs)

    def names(self, sides: dict, on_field) -> dict:
        """{id: the roster entry} for every player whose number was read clearly. `on_field`
        is personnel.OnField (anything with .offense and .defense lists of .number)."""
        rosters = {"offense": [p.number for p in on_field.offense], "defense": [p.number for p in on_field.defense]}
        listed = {("offense", p.number): p for p in on_field.offense} | {("defense", p.number): p for p in on_field.defense}
        return {tid: listed[(sides[tid], n)] for tid, n in assign(self.votes, sides, rosters).items()}
