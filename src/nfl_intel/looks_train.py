"""Fine-tunes the look embedding (looks.py) to tell teammates apart, on our own film.

The labels are free: a run of `nfl run` (or `nfl overlay --out`) has already named players
by their jersey numbers, so every box on a track that was named by number is a picture of a
known man. Those pictures, by team and number, are the classes; ImageNet's ResNet-18 learns
to put the same man's pictures together and his teammates' apart. Names a look gave are not
used (`plays.json` keeps "numbers" and "looks" separate): they would only teach the network
its own mistakes.

Fine-tuned on one clip of a game and marked on another clip of the same game, the embedding
told a player from his on-field teammates 89% of the time against 60% for the pretrained
features, and 98% of the ones it was sure enough to name. On a new game, run this again on
that game's first clip; the model learns the cues that matter (socks, sleeves, gloves,
build) and should carry over partly, but that has not been measured yet.

    uv run python -m nfl_intel.looks_train data/clips/live-04.mp4 outputs/live-04.csv --out models/looks.pt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .jersey import MEAN, STD
from .looks import BODY, DEFAULT_MODEL, body_crop
from .schema import read_csv


def gather(csv_path: str | Path, plays_path: str | Path | None = None, per_track: int = 40) -> list[dict]:
    """Boxes of players named by number: {frame, box, team, number, shot}, up to `per_track`
    evenly spaced per track. The play log (`<csv>.plays.json` by default) says which names
    came from numbers and which team each side was."""
    csv_path = Path(csv_path)
    plays_path = Path(plays_path) if plays_path is not None else csv_path.with_suffix("").with_suffix(".plays.json")
    plays = {p["shot"]: p for p in json.loads(Path(plays_path).read_text())}
    out: list[dict] = []
    by_track: dict = {}
    for f in read_csv(csv_path):
        play = plays.get(f.shot)
        if play is None or not play.get("numbers"):
            continue
        for p in f.players:
            if p.label in play["numbers"] and p.role in ("QB", "REC", "OL", "DEF", "DL"):
                team = play["offense"] if p.role in ("QB", "REC", "OL") else play["defense"]
                by_track.setdefault((f.shot, p.track_id), []).append(
                    {"frame": f.frame, "box": p.box, "team": team, "number": int(play["numbers"][p.label]), "shot": f.shot})
    for rows in by_track.values():
        for i in sorted(set(np.linspace(0, len(rows) - 1, min(len(rows), per_track)).astype(int))):
            out.append(rows[i])
    return out


def crops_for(video: str | Path, rows: list[dict]) -> list[np.ndarray]:
    """The body crops for gathered rows, reading each frame of the clip once."""
    cap = cv2.VideoCapture(str(video))
    want: dict = {}
    for i, r in enumerate(rows):
        want.setdefault(r["frame"], []).append(i)
    crops: list = [None] * len(rows)
    pos = -1
    for frame_idx in sorted(want):
        if frame_idx - pos > 120 or frame_idx <= pos:          # far away: seek; near: step through, it is cheaper
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            pos = frame_idx - 1
        while pos < frame_idx - 1:
            cap.grab()
            pos += 1
        ok, frame = cap.read()
        pos = frame_idx
        if not ok:
            continue
        for i in want[frame_idx]:
            crops[i] = body_crop(frame, rows[i]["box"])
    return crops


def _augment(c: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w, h = BODY
    b = c.astype(np.float32)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.normal(0, 4)), float(rng.uniform(0.9, 1.1)))
    M[:, 2] += rng.normal(0, 5, 2)
    b = cv2.warpAffine(b, M, (w, h), borderMode=cv2.BORDER_REFLECT) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    if rng.random() < 0.3:
        b = cv2.GaussianBlur(b, (0, 0), float(rng.uniform(0.5, 1.5)))
    if rng.random() < 0.3:                                 # a patch hidden: another player in the way
        x0, y0 = int(rng.integers(0, w - 28)), int(rng.integers(0, h - 56))
        b[y0:y0 + 56, x0:x0 + 28] = rng.uniform(0, 255)
    return np.clip(b, 0, 255).astype(np.uint8)


def train(video: str | Path, csv_path: str | Path, out: str | Path = DEFAULT_MODEL, plays_path=None, epochs: int = 8,
          batch: int = 64, lr: float = 2e-4, start: str | Path | None = None, seed: int = 0, log=print) -> Path:
    """Fine-tunes ResNet-18 as a cosine classifier over the (team, number) identities in the
    run and saves the backbone. `start` continues from an earlier `looks.pt`."""
    import torch
    import torch.nn.functional as F
    from torchvision.models import ResNet18_Weights, resnet18

    rows = gather(csv_path, plays_path)
    crops = crops_for(video, rows)
    keep = [i for i, c in enumerate(crops) if c is not None]
    rows, crops = [rows[i] for i in keep], [crops[i] for i in keep]
    ids = sorted({(r["team"], r["number"]) for r in rows})
    if len(ids) < 2:
        raise ValueError("fewer than two players named by number in this run: nothing to tell apart")
    y = np.array([ids.index((r["team"], r["number"])) for r in rows])
    log(f"{len(crops)} crops of {len(ids)} players: " + ", ".join(f"{t} {n}" for t, n in ids))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = resnet18(weights=None if start else ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    if start:
        net.load_state_dict(torch.load(str(start), map_location="cpu")["net"])
    net.to(dev)
    head = torch.nn.Linear(512 * 3, len(ids)).to(dev)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    steps = max(1, len(crops) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.2)

    def feats(x):
        m = net
        h = m.layer4(m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x))))))))
        return torch.cat([h[:, :, :3].mean((2, 3)), h[:, :, 3:6].mean((2, 3)), h[:, :, 6:].mean((2, 3))], 1)

    for epoch in range(epochs):
        net.train()
        total = 0.0
        for _ in range(steps):
            pick = rng.integers(0, len(crops), batch)
            x = np.stack([cv2.cvtColor(_augment(crops[i], rng), cv2.COLOR_BGR2RGB) for i in pick]).astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(((x - MEAN) / STD).transpose(0, 3, 1, 2))).to(dev)
            f = F.normalize(feats(x), dim=1)
            loss = F.cross_entropy(head(f) * 16.0, torch.from_numpy(y[pick]).to(dev), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
        log(f"epoch {epoch + 1}/{epochs}: loss {total / steps:.3f}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict(), "meta": {"players": [f"{t} {n}" for t, n in ids], "crops": len(crops), "epochs": epochs}}, str(out))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the look embedding on a run's players named by number.")
    ap.add_argument("video")
    ap.add_argument("csv", help="the run's tracking CSV; its .plays.json sits beside it")
    ap.add_argument("--plays", help="the play log, if not beside the CSV")
    ap.add_argument("--out", default=str(DEFAULT_MODEL))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--start", help="continue from an earlier looks.pt")
    a = ap.parse_args(argv)
    print(f"saved {train(a.video, a.csv, a.out, a.plays, epochs=a.epochs, start=a.start)}")


if __name__ == "__main__":
    main()
