"""Trains the jersey-number reader (jersey.py) on torso crops from our own film.

The labels are weak on purpose. Nobody boxes digits; a whole track is given its number once
("this player is 21"), and every crop of that track inherits it. Most of those crops show
him side-on with nothing to read. Training on them as if they all showed a 21 teaches the
network that "a brown shirt bent over" is a number (the first version did exactly that, and
called every Browns lineman 58). So the loss asks for less:

  - each film crop may be its track's number or "nothing to read", whichever the picture
    supports. Side-on crops look alike across tracks with different numbers, so the only
    answer that fits them all is "nothing to read";
  - each track's handful of crops in a batch (a bag) must show the number at least once, or
    everything would be called unreadable. The bags that fit worst are skipped: sometimes
    none of the handful is readable;
  - "sure" crops, where a text reader already read the right number (free, but only on
    light jerseys) or a person marked the number readable (`jersey_mark`), are trained on
    as plain examples; crops a person marked as nothing to read are trained as blank;
  - half of every batch is made-up torsos (`synth`) with every number from 0 to 99 in many
    fonts and colours, and some with no number at all, so the digits are learned as digits
    and not as "the Browns' cornerback".

Data layout: `<dir>/labels.csv` with columns file, number, sure, group, clip, and the crops
(96x96 BGR PNGs from `jersey.torso_crop`) in `<dir>/crops/`. It is film: never committed.

    uv run python -m nfl_intel.jersey_train data/jersey --out models/jersey.pt
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .jersey import DEFAULT_MODEL, NONE, SIZE, NumberReader, build_net, digits_of, to_input

FONT_DIRS = ["/System/Library/Fonts/Supplemental", "/System/Library/Fonts", "/Library/Fonts"]
FONT_NAMES = ["Impact.ttf", "Arial Black.ttf", "Arial Bold.ttf", "Arial Narrow Bold.ttf", "DIN Condensed Bold.ttf",
              "DIN Alternate Bold.ttf", "Tahoma Bold.ttf", "Trebuchet MS Bold.ttf", "Georgia Bold.ttf", "Arial Rounded Bold.ttf",
              "Courier New Bold.ttf", "Verdana Bold.ttf"]
_fonts: list[str] | None = None


def fonts() -> list[str]:
    global _fonts
    if _fonts is None:
        _fonts = [str(Path(d) / n) for d in FONT_DIRS for n in FONT_NAMES if (Path(d) / n).exists()]
    return _fonts


def _contrasting(rng, base: np.ndarray) -> np.ndarray:
    """A colour that stands out from `base`, the way a number does from its shirt."""
    lum = float(base @ [0.114, 0.587, 0.299])
    for _ in range(20):
        c = rng.integers(0, 256, 3)
        if rng.random() < 0.6:
            c = np.full(3, 255 if lum < 128 else 0) + rng.integers(-25, 26, 3)
        c = np.clip(c, 0, 255)
        if abs(float(c @ [0.114, 0.587, 0.299]) - lum) > 90:
            return c
    return np.full(3, 255 if lum < 128 else 0)


def _draw_number(rng, S: int, text: str, ink, outline, outline_w: int) -> np.ndarray:
    """The number on a transparent layer (BGRA), slanted the way a shirt is."""
    px = int(S * rng.uniform(0.34, 0.62))               # numbers fill a good part of the chest
    names = fonts()
    layer = np.zeros((S, S, 4), np.uint8)
    if names:
        from PIL import Image, ImageDraw, ImageFont

        im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        font = ImageFont.truetype(names[int(rng.integers(len(names)))], px)
        kw = {"stroke_width": outline_w, "stroke_fill": tuple(int(v) for v in outline[::-1]) + (255,)} if outline_w else {}
        ImageDraw.Draw(im).text((S / 2, S / 2), text, font=font, fill=tuple(int(v) for v in ink[::-1]) + (255,), anchor="mm", **kw)
        layer = cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGRA)
    else:                                              # no system fonts (not a Mac): OpenCV's own
        scale = px / 22.0
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, max(2, px // 8))
        org = (int(S / 2 - tw / 2), int(S / 2 + th / 2))
        if outline_w:
            cv2.putText(layer, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (*map(int, outline), 255), max(2, px // 8) + 2 * outline_w, cv2.LINE_AA)
        cv2.putText(layer, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (*map(int, ink), 255), max(2, px // 8), cv2.LINE_AA)
    angle, shear = float(rng.normal(0, 16)), float(rng.normal(0, 0.18))
    squeeze = float(rng.uniform(0.55, 1.0))            # a shirt turned away from the camera
    M = cv2.getRotationMatrix2D((S / 2, S / 2), angle, 1.0)
    A = np.array([[squeeze, shear, (1 - squeeze) * S / 2 - shear * S / 2], [0, 1, 0], [0, 0, 1.0]])
    M = (A @ np.vstack([M, [0, 0, 1]]))[:2]
    M[:, 2] += [rng.normal(0, S * 0.06), rng.normal(0, S * 0.07)]
    return cv2.warpAffine(layer, M, (S, S), flags=cv2.INTER_LINEAR)


def synth(rng: np.random.Generator, size: int = SIZE) -> tuple[np.ndarray, int | None]:
    """One made-up torso (BGR, `size` square) and its number; None means there is nothing to
    read on it (about one in eight)."""
    S = size * 2
    grass = np.array([rng.integers(30, 95), rng.integers(95, 170), rng.integers(40, 105)], dtype=np.float64)
    img = np.tile(grass, (S, S, 1)) + rng.normal(0, 6, (S, S, 1))
    jersey = rng.integers(0, 256, 3).astype(np.float64)
    r = rng.random()
    if r < 0.35:
        jersey = 215 + rng.integers(0, 41, 3).astype(np.float64)
    elif r < 0.6:
        jersey = rng.integers(0, 70, 3).astype(np.float64)
    cx, half = S / 2 + rng.normal(0, S * 0.05), S * rng.uniform(0.24, 0.42)
    top = S * rng.uniform(0.0, 0.16)
    body = np.zeros((S, S), np.uint8)
    lean = rng.normal(0, S * 0.08)
    pts = np.array([[cx - half, top], [cx + half, top], [cx + half * 0.8 + lean, S], [cx - half * 0.8 + lean, S]], dtype=np.int32)
    cv2.fillConvexPoly(body, pts, 255)
    cv2.ellipse(body, (int(cx), int(top + S * 0.08)), (int(half * 1.1), int(S * 0.12)), 0, 0, 360, 255, -1)
    img[body > 0] = jersey + rng.normal(0, 5, (int((body > 0).sum()), 3))
    if rng.random() < 0.5:                             # trousers showing at the bottom
        y = int(S * rng.uniform(0.78, 0.95))
        img[y:][body[y:] > 0] = rng.integers(0, 256, 3)
    if rng.random() < 0.6:                             # the helmet above
        cv2.circle(img, (int(cx + rng.normal(0, S * 0.08)), int(top - S * rng.uniform(-0.02, 0.08))), int(S * rng.uniform(0.09, 0.15)),
                   tuple(float(v) for v in rng.integers(0, 256, 3)), -1)
    number = None if rng.random() < 0.125 else int(rng.integers(0, 100))
    if number is not None:
        ink = _contrasting(rng, jersey)
        ow = int(rng.integers(1, 4)) if rng.random() < 0.5 else 0
        layer = _draw_number(rng, S, str(number), ink, rng.integers(0, 256, 3), ow)
        alpha = (layer[..., 3:4].astype(np.float64) / 255.0) * (body[..., None] > 0)
        img = img * (1 - alpha) + layer[..., :3] * alpha
    if rng.random() < 0.35:                            # an arm, a strap, another player's hand
        p, q = rng.integers(0, S, 2), rng.integers(0, S, 2)
        cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), tuple(float(v) for v in rng.integers(0, 256, 3)), int(S * rng.uniform(0.03, 0.09)))
    img *= np.linspace(rng.uniform(0.75, 1.1), rng.uniform(0.75, 1.1), S)[None, :, None]     # light across the body
    img = np.clip(img, 0, 255).astype(np.uint8)
    small = int(rng.integers(30, 62))                  # film gives a torso about this many pixels
    img = cv2.resize(img, (small, small), interpolation=cv2.INTER_AREA)
    img = np.clip(img + rng.normal(0, rng.uniform(1, 6), img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 75))])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC), number


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A film crop, nudged: small turn, shift and zoom, lighting, sometimes grey or inverted.
    Never mirrored: a mirrored 2 is not a 2."""
    S = img.shape[0]
    M = cv2.getRotationMatrix2D((S / 2, S / 2), float(rng.normal(0, 7)), float(rng.uniform(0.9, 1.15)))
    M[:, 2] += rng.normal(0, S * 0.04, 2)
    out = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    out = out * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    out = np.clip(out, 0, 255).astype(np.uint8)
    r = rng.random()
    if r < 0.15:
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif r < 0.30:
        out = 255 - out                                # a dark shirt for a light one, and back
    elif r < 0.55:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int32) + int(rng.integers(0, 180))) % 180
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.5, 1.3)))
    return out


def load_labels(data_dir: str | Path, with_marks: bool = True) -> list[dict]:
    """Rows of labels.csv as dicts, with `number`, `sure` and `blank` as ints.

    `marks.csv` (from `jersey_mark`), when present, makes a crop marked readable `sure` and
    a crop marked as nothing to read `blank`; the rest keep the weak track label."""
    data_dir = Path(data_dir)
    with open(data_dir / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    marks: dict[str, int] = {}
    if with_marks:
        from .jersey_mark import load_marks

        marks = load_marks(data_dir)
    for r in rows:
        r["number"], r["sure"], r["blank"] = int(r["number"]), int(r["sure"]), 0
        r["path"] = str(data_dir / "crops" / r["file"])
        if r["file"] in marks:
            r["sure"], r["blank"] = (1, 0) if marks[r["file"]] else (0, 1)
    return rows


def film_loss(lp_number, lp_blank, sure, blank, blank_pull: float):
    """Per film crop: a sure crop is its number, a blank one is nothing to read, any other is
    its number or nothing, with a small pull towards nothing."""
    import torch

    either = -torch.logaddexp(lp_number, lp_blank) - blank_pull * lp_blank
    return torch.where(sure, -lp_number, torch.where(blank, -lp_blank, either))


def _targets(numbers: list[int | None]) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([NONE if n is None else digits_of(n)[0] for n in numbers], dtype=np.int64)
    u = np.array([NONE if n is None else digits_of(n)[1] for n in numbers], dtype=np.int64)
    return t, u


def train(data_dir: str | Path, out: str | Path = DEFAULT_MODEL, epochs: int = 14, bags: int = 8, bag: int = 6, n_synth: int = 48,
          keep: float = 0.75, warm: int = 2, blank_pull: float = 0.05, lr: float = 4e-4, seed: int = 0,
          hold_out: tuple[str, ...] = (), log=print) -> Path:
    """Fine-tunes the network and saves it. Each step sees `bags` tracks with `bag` crops
    each, plus `n_synth` made-up torsos. `hold_out` names clips to leave out (for marking)."""
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    rows = [r for r in load_labels(data_dir) if r["clip"] not in hold_out]
    images = [cv2.imread(r["path"]) for r in rows]
    keep_rows = [i for i, im in enumerate(images) if im is not None]
    rows, images = [rows[i] for i in keep_rows], [images[i] for i in keep_rows]
    if not rows:
        raise ValueError(f"no training crops in {data_dir}")
    groups: dict = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["group"]].append(i)
    keys = sorted(groups)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = build_net(pretrained=True).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(rows) // (bags * bag))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.15)
    log(f"{len(rows)} film crops ({sum(r['sure'] for r in rows)} sure, {sum(r['blank'] for r in rows)} marked blank) "
        f"from {len(keys)} tracks; {steps} steps an epoch on {dev}")
    n_film = bags * bag
    for epoch in range(epochs):
        net.train()
        total, used, seen = 0.0, 0, 0
        for _ in range(steps):
            pick: list[int] = []
            for g in rng.choice(len(keys), bags, replace=len(keys) < bags):
                members = np.array(groups[keys[g]])
                w = np.array([3.0 if rows[i]["sure"] else 1.0 for i in members])
                pick.extend(rng.choice(members, bag, replace=len(members) < bag, p=w / w.sum()).tolist())
            crops = [augment(images[i], rng) for i in pick]
            numbers: list[int | None] = [rows[i]["number"] for i in pick]
            sure = np.array([bool(rows[i]["sure"]) for i in pick])
            blank = np.array([bool(rows[i]["blank"]) for i in pick])
            for _ in range(n_synth):
                im, n = synth(rng)
                crops.append(augment(im, rng) if rng.random() < 0.5 else im)
                numbers.append(n)
            t, u = (torch.from_numpy(v).to(dev) for v in _targets(numbers))
            lt, lu = net(torch.from_numpy(to_input(crops)).to(dev))
            lpt, lpu = F.log_softmax(lt, 1), F.log_softmax(lu, 1)
            idx = torch.arange(len(crops), device=dev)
            lp_number = lpt[idx, t] + lpu[idx, u]                   # log P(the label)
            lp_blank = lpt[:, NONE] + lpu[:, NONE]                 # log P(nothing to read)
            film = film_loss(lp_number[:n_film], lp_blank[:n_film], torch.from_numpy(sure).to(dev),
                             torch.from_numpy(blank).to(dev), blank_pull)
            # each bag must show its number at least once
            miss = torch.log1p(-lp_number[:n_film].exp().clamp(max=1 - 1e-6)).view(bags, bag).sum(1)
            bag_loss = -torch.log1p(-miss.exp().clamp(max=1 - 1e-6))
            use = torch.ones(bags, dtype=torch.bool, device=dev)
            if epoch >= warm and bags > 1:                           # a handful with nothing readable in it
                cut = torch.quantile(bag_loss.detach().float().cpu(), keep).to(dev)
                use = bag_loss.detach() <= cut
            step_loss = film.mean() + bag_loss[use].mean() + (-lp_number[n_film:]).mean()
            opt.zero_grad()
            step_loss.backward()
            opt.step()
            sched.step()
            total += float(step_loss.detach())
            used += int(use.sum())
            seen += bags
        log(f"epoch {epoch + 1}/{epochs}: loss {total / steps:.3f}, {used}/{seen} bags used")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict(), "meta": {"crops": len(rows), "tracks": len(keys), "epochs": epochs, "hold_out": list(hold_out)}}, str(out))
    return out


def mark(reader: NumberReader, rows: list[dict], min_conf: float = 0.6, min_votes: float = 2.5) -> dict:
    """Marks the reader against labelled tracks.

    Per crop: how many it was sure of, and how many of those were right. Per track: named
    right, named wrong, or left alone, with a number called only on `min_votes` of evidence."""
    images = [cv2.imread(r["path"]) for r in rows]
    probs = np.zeros((len(rows), 100))
    for i in range(0, len(rows), 256):
        probs[i:i + 256], _ = reader.read_crops(images[i:i + 256])
    conf, said = probs.max(1), probs.argmax(1)
    score: dict = defaultdict(lambda: np.zeros(100))
    truth: dict = {}
    out: Counter = Counter()
    for r, p, c, n in zip(rows, probs, conf, said):
        truth[r["group"]] = r["number"]
        out["crops"] += 1
        if c >= min_conf:
            out["sure"] += 1
            out["sure_right"] += int(n == r["number"])
            score[r["group"]] += p
    for g, n in truth.items():
        s = score.get(g)
        if s is None or s.max() < min_votes:
            out["tracks_none"] += 1
        elif int(s.argmax()) == n:
            out["tracks_right"] += 1
        else:
            out["tracks_wrong"] += 1
    return dict(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train the jersey-number reader on labelled torso crops.")
    ap.add_argument("data", help="folder with labels.csv and crops/")
    ap.add_argument("--out", default=str(DEFAULT_MODEL))
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--hold-out", nargs="*", default=[], help="clips to leave out of training and mark on")
    a = ap.parse_args(argv)
    path = train(a.data, a.out, epochs=a.epochs, hold_out=tuple(a.hold_out))
    print(f"saved {path}")
    if a.hold_out:
        rows = [r for r in load_labels(a.data) if r["clip"] in a.hold_out]
        print("held-out clips:", mark(NumberReader(path), rows))


if __name__ == "__main__":
    main()
