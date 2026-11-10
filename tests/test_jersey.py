"""Jersey numbers: the crop, the digit heads, the votes over a shot, and who gets which number.
Synthetic pictures only."""

import csv

import cv2
import numpy as np
import pytest

from nfl_intel.jersey import NONE, SIZE, NumberVotes, ShotNumbers, assign, build_net, digits_of, number_probs, to_input, torso_crop
from nfl_intel.jersey_train import _targets, augment, load_labels, mark, synth


def one_hot(number: int, strength: float = 0.9) -> np.ndarray:
    """What a reader that is `strength` sure of `number` returns for one crop."""
    p = np.full(100, (1 - strength) / 99)
    p[number] = strength
    return p


def test_digits_of_splits_tens_and_units():
    assert digits_of(7) == (NONE, 7)
    assert digits_of(0) == (NONE, 0)
    assert digits_of(70) == (7, 0)
    assert digits_of(99) == (9, 9)
    with pytest.raises(ValueError):
        digits_of(100)


def test_number_probs_puts_the_heads_together():
    tens, units = np.zeros((3, 11)), np.zeros((3, 11))
    tens[0, 2] = units[0, 1] = 1.0                     # 21
    tens[1, NONE] = units[1, 5] = 1.0                  # 5
    tens[2, NONE] = units[2, NONE] = 1.0               # nothing to read
    probs, blank = number_probs(tens, units)
    assert probs.shape == (3, 100)
    assert probs[0].argmax() == 21 and probs[1].argmax() == 5
    assert probs[2].max() == 0.0 and blank[2] == 1.0
    assert np.all(probs.sum(1) <= 1.0 + 1e-9)


def test_torso_crop_takes_the_chest_not_the_legs():
    frame = np.zeros((400, 400, 3), np.uint8)
    frame[100:162] = (0, 0, 255)                       # chest rows of the box below: red
    frame[162:200] = (255, 0, 0)                       # legs: blue
    crop = torso_crop(frame, (180, 100, 220, 200))
    assert crop.shape == (SIZE, SIZE, 3)
    assert crop[..., 2].mean() > 200 and crop[..., 0].mean() < 30
    assert torso_crop(frame, (10, 10, 12, 14)) is None


def test_to_input_is_what_the_network_takes():
    x = to_input([np.zeros((SIZE, SIZE, 3), np.uint8)] * 2)
    assert x.shape == (2, 3, SIZE, SIZE) and x.dtype == np.float32


def test_network_has_a_tens_and_a_units_head():
    import torch

    net = build_net().eval()
    with torch.no_grad():
        tens, units = net(torch.zeros(2, 3, SIZE, SIZE))
    assert tens.shape == (2, 11) and units.shape == (2, 11)


def test_votes_ignore_frames_the_reader_is_unsure_of():
    votes = NumberVotes(min_conf=0.6)
    votes.add(["a"] * 4, np.array([one_hot(21)] * 4))
    votes.add(["a"] * 30, np.array([one_hot(26, 0.3)] * 30))       # thirty side-on frames leaning the wrong way
    number, score, runner_up = votes.best("a")
    assert number == 21 and score > 3.0 and runner_up < 0.1
    assert votes.best("nobody") == (-1, 0.0, 0.0)


def test_best_among_a_roster_skips_numbers_not_on_the_field():
    votes = NumberVotes()
    votes.add(["a"] * 3, np.array([one_hot(28)] * 3))              # an 82 read back to front
    votes.add(["a"] * 2, np.array([one_hot(82)] * 2))
    assert votes.best("a")[0] == 28
    assert votes.best("a", among=[82, 18, 9])[0] == 82


def test_assign_names_only_the_tracks_with_clear_evidence():
    votes = NumberVotes()
    votes.add(["corner"] * 5, np.array([one_hot(21)] * 5))
    votes.add(["safety"] * 2, np.array([one_hot(33)] * 2))         # two frames: not enough
    votes.add(["torn"] * 4, np.array([one_hot(26)] * 4))
    votes.add(["torn"] * 3, np.array([one_hot(29)] * 3))           # can't choose between two men on the roster
    sides = {"corner": "defense", "safety": "defense", "torn": "defense", "unseen": "defense"}
    named = assign(votes, sides, {"defense": [21, 26, 29, 33, 49]})
    assert named == {"corner": 21}


def test_assign_gives_a_number_to_one_track_per_side():
    votes = NumberVotes()
    votes.add(["a"] * 6, np.array([one_hot(18)] * 6))
    votes.add(["b"] * 4, np.array([one_hot(18)] * 4))
    votes.add(["d"] * 5, np.array([one_hot(18)] * 5))              # the other team's 18 is a different man
    sides = {"a": "offense", "b": "offense", "d": "defense"}
    named = assign(votes, sides, {"offense": [18, 82], "defense": [18, 5]})
    assert named == {"a": 18, "d": 18}


def test_synth_makes_a_torso_with_a_number_or_none():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(60):
        img, number = synth(rng)
        assert img.shape == (SIZE, SIZE, 3) and img.dtype == np.uint8
        assert number is None or 0 <= number <= 99
        seen.add(number is None)
    assert seen == {True, False}                       # some with nothing to read
    a, b = synth(np.random.default_rng(5)), synth(np.random.default_rng(5))
    assert a[1] == b[1] and np.array_equal(a[0], b[0])


def test_augment_keeps_the_picture_a_picture():
    rng = np.random.default_rng(1)
    img, _ = synth(rng)
    for _ in range(10):
        out = augment(img, rng)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_targets_use_the_none_class_for_a_blank():
    tens, units = _targets([21, 5, None])
    assert tens.tolist() == [2, NONE, NONE] and units.tolist() == [1, 5, NONE]


class FakeReader:
    """Reads the number painted in the crop's top-left pixel; 255 there means it can't tell."""

    def read_crops(self, crops):
        probs = np.zeros((len(crops), 100))
        for i, c in enumerate(crops):
            probs[i] = np.full(100, 0.01) if c[0, 0, 0] == 255 else one_hot(int(c[0, 0, 0]))
        return probs, np.zeros(len(crops))


def test_mark_counts_crops_and_tracks(tmp_path):
    (tmp_path / "crops").mkdir()
    rows = []

    def crop(name, shows, number, group):
        img = np.zeros((SIZE, SIZE, 3), np.uint8)
        img[0, 0] = shows
        cv2.imwrite(str(tmp_path / "crops" / name), img)
        rows.append([name, number, 0, group, "clip"])

    for i in range(4):
        crop(f"a{i}.png", 21, 21, "right")             # read correctly four times
    for i in range(4):
        crop(f"b{i}.png", 26, 29, "wrong")             # read as someone else
    for i in range(4):
        crop(f"c{i}.png", 255, 33, "unreadable")
    with open(tmp_path / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "number", "sure", "group", "clip"])
        w.writerows(rows)
    out = mark(FakeReader(), load_labels(tmp_path))
    assert out["crops"] == 12 and out["sure"] == 8 and out["sure_right"] == 4
    assert (out["tracks_right"], out["tracks_wrong"], out["tracks_none"]) == (1, 1, 1)


class BoxReader:
    """Stands in for the network: reads the number painted in each box's top-left pixel."""

    def read(self, frame, boxes):
        probs = np.array([one_hot(int(frame[int(b[1]), int(b[0]), 0])) for b in boxes])
        return np.arange(len(boxes)), probs, np.zeros(len(boxes))


def test_shot_numbers_turns_reads_into_roster_names():
    from nfl_intel.personnel import Listed, OnField

    on_field = OnField(offense=[Listed(18, "WR", "Jalen Coker"), Listed(82, "TE", "Tommy Tremble")],
                       defense=[Listed(21, "CB", "Denzel Ward"), Listed(33, "FS", "Ronnie Hickman Jr.")])
    frame = np.zeros((200, 300, 3), np.uint8)
    boxes = [(10, 10, 50, 110), (100, 10, 140, 110), (200, 10, 240, 110)]
    for (x, y, _, _), number in zip(boxes, (18, 21, 82)):
        frame[y, x] = number
    shot = ShotNumbers(BoxReader(), every=3)
    for _ in range(12):                                # read on frames 1, 4, 7, 10: four looks each
        shot.update(frame, boxes, ["wr", "cb", "stranger"])
    sides = {"wr": "offense", "cb": "defense", "stranger": "defense"}   # an 82 on defense is nobody on that roster
    named = shot.names(sides, on_field)
    assert {k: v.last_name for k, v in named.items()} == {"wr": "Coker", "cb": "Ward"}


class PixelReader:
    """Stands in for the network on the synthetic film: each player's number is painted in
    the top-left pixel of his box; 255 means his shirt can't be read."""

    def read(self, frame, boxes):
        probs = np.array([np.full(100, 0.005) if frame[int(b[1]), int(b[0]), 0] == 255 else one_hot(int(frame[int(b[1]), int(b[0]), 0]))
                          for b in boxes])
        return np.arange(len(boxes)), probs, np.zeros(len(boxes))


def test_the_analyzer_puts_roster_names_on_players_it_can_read():
    from formations import film
    from nfl_intel.analysis import PlayAnalyzer
    from nfl_intel.personnel import Listed, OnField
    from nfl_intel.playinfo import PlayHeader, PlayRecord

    frames, _ = film(direction=-1, los_rel=40.0)
    header = PlayHeader(quarter=4, clock="08:21", down=2, distance=7, offense="GB", side="ATL", yard=15)
    rec = PlayRecord(header, matched=True, defense="ATL", yards_to_goal=15, play_type="pass", charted={"qb_location": "S"})
    rec.on_field = OnField(offense=[Listed(10, "QB", "Jordan Love"), Listed(80, "WR", "Matthew Golden"), Listed(85, "TE", "Tucker Kraft")],
                           defense=[Listed(21, "CB", "A.J. Terrell"), Listed(3, "FS", "Jessie Bates")])
    # the film's boxes start at x = 10 * track id; paint a number there for three players
    img = np.full((30, 300, 3), 255, np.uint8)
    img[0, 0] = 10                                         # track 0 is the quarterback
    img[0, 80] = 80                                        # track 8 is a receiver
    img[0, 110] = 21                                       # track 11 is a defender
    an = PlayAnalyzer(None, record=rec, number_reader=PixelReader())
    for f in frames:
        an.update(img, f)
    label = {s.track_id: an.labels.get(s.id) for s in an.tracker.slots}
    assert label[0] == "Love" and label[8] == "Golden" and label[11] == "Terrell"
    assert label[10] == "R5"                               # nothing readable on him: keeps his formation label
    assert an.summary()["numbers"] == {"Love": 10, "Golden": 80, "Terrell": 21}


def test_a_number_takes_the_name_off_a_formation_guess():
    from formations import film
    from nfl_intel.analysis import PlayAnalyzer
    from nfl_intel.personnel import Listed, OnField
    from nfl_intel.playinfo import PlayHeader, PlayRecord

    frames, _ = film(direction=-1, los_rel=40.0)
    header = PlayHeader(quarter=4, clock="08:21", down=2, distance=7, offense="GB", side="ATL", yard=15)
    rec = PlayRecord(header, matched=True, defense="ATL", yards_to_goal=15, play_type="pass", charted={"qb_location": "S"})
    rec.on_field = OnField(offense=[Listed(10, "QB", "Jordan Love"), Listed(30, "RB", "Josh Jacobs")] + [Listed(80 + i, "WR", f"W{i} Wide") for i in range(4)],
                           defense=[Listed(21, "CB", "A.J. Terrell")])
    img = np.full((30, 300, 3), 255, np.uint8)
    img[0, 80] = 30                                        # the number 30 is read on track 8, a receiver out wide
    an = PlayAnalyzer(None, record=rec, number_reader=PixelReader())
    for f in frames:
        an.update(img, f)
    label = {s.track_id: an.labels.get(s.id) for s in an.tracker.slots}
    assert label[8] == "Jacobs"
    assert list(an.labels.values()).count("Jacobs") == 1   # the lone back the formation called Jacobs lost the name
