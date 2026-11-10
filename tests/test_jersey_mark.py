"""Marking crops readable or not: pages by track, the click cycle, marks into the training labels and loss."""

import csv

import cv2
import numpy as np

from nfl_intel.jersey_mark import READABLE, UNREADABLE, cell_at, grid, load_marks, pages, save_marks, toggle
from nfl_intel.jersey_train import film_loss, load_labels


def _data(tmp_path):
    (tmp_path / "crops").mkdir()
    rows = []
    for g, number, kind, n in (("a:1:1", 21, "dark", 3), ("b:2:2", 7, "white", 2), ("c:3:3", 58, "dark", 2)):
        for i in range(n):
            file = f"{g.replace(':', '_')}_{i}.png"
            cv2.imwrite(str(tmp_path / "crops" / file), np.full((96, 96, 3), 90, dtype=np.uint8))
            rows.append({"file": file, "number": number, "sure": 0, "group": g, "clip": g[0], "kind": kind})
    with (tmp_path / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def test_pages_and_toggle(tmp_path):
    rows = load_labels(_data(tmp_path))
    book = pages(rows, "dark", per_page=2)
    assert [[r["file"][:1] for r in page] for page in book] == [["a", "a"], ["a"], ["c", "c"]]
    assert pages(rows, None, 10)[0][0]["group"] == "a:1:1" and len(pages(rows, None, 10)) == 3
    assert toggle(None) == READABLE and toggle(READABLE) == UNREADABLE and toggle(UNREADABLE) is None


def test_grid_and_cell_lookup():
    images = [np.zeros((96, 96, 3), dtype=np.uint8)] * 5
    canvas = grid(images, [None, READABLE, UNREADABLE, None, None], cols=4, cell=50, header="x", top=20)
    assert canvas.shape == (20 + 2 * 50, 200, 3)
    assert cell_at(10, 25, 4, 50, 5, 20) == 0 and cell_at(60, 25, 4, 50, 5, 20) == 1
    assert cell_at(10, 80, 4, 50, 5, 20) == 4 and cell_at(60, 80, 4, 50, 5, 20) is None     # only five crops
    assert cell_at(10, 5, 4, 50, 5, 20) is None                                              # the header
    assert tuple(canvas[22, 52]) == (60, 200, 60) and tuple(canvas[22, 102]) == (50, 50, 220)   # the borders


def test_marks_round_trip_into_the_labels(tmp_path):
    d = _data(tmp_path)
    save_marks(d, {"a_1_1_0.png": READABLE, "a_1_1_1.png": UNREADABLE})
    assert load_marks(d) == {"a_1_1_0.png": 1, "a_1_1_1.png": 0}
    by_file = {r["file"]: r for r in load_labels(d)}
    assert (by_file["a_1_1_0.png"]["sure"], by_file["a_1_1_0.png"]["blank"]) == (1, 0)
    assert (by_file["a_1_1_1.png"]["sure"], by_file["a_1_1_1.png"]["blank"]) == (0, 1)
    assert (by_file["a_1_1_2.png"]["sure"], by_file["a_1_1_2.png"]["blank"]) == (0, 0)
    assert all(r["blank"] == 0 for r in load_labels(d, with_marks=False))


def test_film_loss_uses_the_marks():
    import torch

    lp_number = torch.log(torch.tensor([0.2, 0.2, 0.2]))
    lp_blank = torch.log(torch.tensor([0.5, 0.5, 0.5]))
    sure = torch.tensor([True, False, False])
    blank = torch.tensor([False, True, False])
    loss = film_loss(lp_number, lp_blank, sure, blank, blank_pull=0.0)
    assert torch.allclose(loss, -torch.log(torch.tensor([0.2, 0.5, 0.7])), atol=1e-6)
    pulled = film_loss(lp_number, lp_blank, sure, blank, blank_pull=0.1)
    assert pulled[2] > loss[2] and torch.allclose(pulled[:2], loss[:2])
