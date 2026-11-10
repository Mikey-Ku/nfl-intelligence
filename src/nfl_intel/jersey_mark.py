"""Mark torso crops readable or not, a page at a time, so the number reader can learn dark jerseys.

The reader's film labels are per track ("this player is 21"), and the loss lets each crop be
its number or nothing to read. That was enough for light jerseys, where a text reader had
already marked hundreds of crops as surely showing their number. Dark jerseys got no such
crops, and the reader learned almost nothing about them. This is the quickest way to make
them: a page of crops from one track at a time, click the ones where the number is plainly
readable (green) and, with a second click, the ones where there is nothing to read (red).
A third click clears. Unmarked crops stay as they were.

    uv run python -m nfl_intel.jersey_mark data/jersey --kind dark

Keys: `n` next page, `p` previous, `a` mark every unmarked crop on the page readable,
`z` mark them all unreadable, `q` save and quit. Progress is saved to `<dir>/marks.csv`
(file, readable) on every page turn, and picked up again next time. `jersey_train` reads
it: a readable crop is trained as its number, an unreadable one as "nothing to read".
Each page shows the track's number in the corner, so a wrong track label shows up too:
leave that track unmarked and fix labels.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

WINDOW = "nfl jersey marks"
READABLE, UNREADABLE = 1, 0
COLOUR = {READABLE: (60, 200, 60), UNREADABLE: (50, 50, 220)}


def load_marks(data_dir: str | Path) -> dict[str, int]:
    path = Path(data_dir) / "marks.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {r["file"]: int(r["readable"]) for r in csv.DictReader(f)}


def save_marks(data_dir: str | Path, marks: dict[str, int]) -> Path:
    path = Path(data_dir) / "marks.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "readable"])
        for file in sorted(marks):
            w.writerow([file, marks[file]])
    return path


def pages(rows: list[dict], kind: str | None, per_page: int) -> list[list[dict]]:
    """Crops of one kind, grouped by track, each track cut into pages of `per_page`."""
    chosen = sorted((r for r in rows if kind is None or r.get("kind") == kind), key=lambda r: (r["group"], r["file"]))
    out: list[list[dict]] = []
    track: list[dict] = []
    for r in chosen:
        if track and r["group"] != track[0]["group"]:
            out.extend(track[i:i + per_page] for i in range(0, len(track), per_page))
            track = []
        track.append(r)
    if track:
        out.extend(track[i:i + per_page] for i in range(0, len(track), per_page))
    return out


def toggle(state: int | None) -> int | None:
    """Unmarked -> readable -> unreadable -> unmarked."""
    return {None: READABLE, READABLE: UNREADABLE, UNREADABLE: None}[state]


def cell_at(x: int, y: int, cols: int, cell: int, n: int, top: int) -> int | None:
    if y < top:
        return None
    i = (y - top) // cell * cols + x // cell
    return int(i) if 0 <= x // cell < cols and 0 <= i < n else None


def grid(images: list[np.ndarray], states: list[int | None], cols: int, cell: int, header: str, top: int = 36) -> np.ndarray:
    rows = max(1, (len(images) + cols - 1) // cols)
    canvas = np.full((top + rows * cell, cols * cell, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, header, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    for i, (im, st) in enumerate(zip(images, states)):
        r, c = divmod(i, cols)
        y, x = top + r * cell, c * cell
        tile = cv2.resize(im, (cell - 6, cell - 6), interpolation=cv2.INTER_CUBIC)
        canvas[y + 3:y + cell - 3, x + 3:x + cell - 3] = tile
        if st is not None:
            cv2.rectangle(canvas, (x + 1, y + 1), (x + cell - 2, y + cell - 2), COLOUR[st], 3)
    return canvas


def mark_interactive(data_dir: str | Path, kind: str | None = "dark", cols: int = 8, per_page: int = 40, cell: int = 128) -> dict[str, int]:
    from .jersey_train import load_labels

    data_dir = Path(data_dir)
    rows = load_labels(data_dir, with_marks=False)
    marks = load_marks(data_dir)
    book = pages(rows, kind, per_page)
    if not book:
        raise SystemExit(f"no crops of kind {kind!r} in {data_dir}")
    k = 0
    while k < len(book) and all(r["file"] in marks for r in book[k]):          # resume after the last finished page
        k += 1
    k = min(k, len(book) - 1)
    clicked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    images = [cv2.imread(r["path"]) for r in book[k]]
    while True:
        page = book[k]
        states = [marks.get(r["file"]) for r in page]
        done = sum(1 for r in rows if (kind is None or r.get("kind") == kind) and r["file"] in marks)
        header = (f"page {k + 1}/{len(book)}  track {page[0]['group']}  number {page[0]['number']}   "
                  f"{done} marked   click: green=readable, red=nothing, again=clear   n/p pages  a=all readable  z=all nothing  q=save")
        cv2.imshow(WINDOW, grid(images, states, cols, cell, header))
        key = cv2.waitKey(30) & 0xFF
        if clicked:
            x, y = clicked.pop()
            i = cell_at(x, y, cols, cell, len(page), 36)
            if i is not None:
                st = toggle(marks.get(page[i]["file"]))
                if st is None:
                    marks.pop(page[i]["file"], None)
                else:
                    marks[page[i]["file"]] = st
        elif key in (ord("a"), ord("z")):
            for r in page:
                marks.setdefault(r["file"], READABLE if key == ord("a") else UNREADABLE)
        elif key in (ord("n"), ord("p")):
            save_marks(data_dir, marks)
            k = min(k + 1, len(book) - 1) if key == ord("n") else max(k - 1, 0)
            images = [cv2.imread(r["path"]) for r in book[k]]
        elif key == ord("q"):
            break
    cv2.destroyWindow(WINDOW)
    path = save_marks(data_dir, marks)
    n_read = sum(1 for v in marks.values() if v == READABLE)
    print(f"{len(marks)} crops marked ({n_read} readable, {len(marks) - n_read} nothing to read) -> {path}")
    return marks


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Mark torso crops readable or not, a track at a time.")
    ap.add_argument("data", help="folder with labels.csv and crops/")
    ap.add_argument("--kind", default="dark", help="which crops: dark, white, or all")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--per-page", type=int, default=40)
    ap.add_argument("--cell", type=int, default=128, help="pixels per crop on screen")
    a = ap.parse_args(argv)
    mark_interactive(a.data, None if a.kind == "all" else a.kind, a.cols, a.per_page, a.cell)


if __name__ == "__main__":
    main()
