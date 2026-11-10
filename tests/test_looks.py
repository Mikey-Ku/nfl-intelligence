"""Player looks: the book of looks, the per-shot means, and a player known again on the next
play from his look alone. Synthetic pictures only."""

import numpy as np

from formations import film
from nfl_intel.analysis import PlayAnalyzer
from nfl_intel.looks import BODY, DIM, LookBook, ShotLooks, body_crop, normalised
from nfl_intel.personnel import Listed, OnField
from nfl_intel.playinfo import PlayHeader, PlayRecord


def look(seed: int, noise: float = 0.0, rng=None) -> np.ndarray:
    """A made-up look: a fixed random unit vector per man, a little different every frame."""
    v = np.random.default_rng(seed).normal(size=DIM)
    if noise and rng is not None:
        v = v + noise * rng.normal(size=DIM)
    return normalised(v)


def test_body_crop_covers_the_whole_player():
    frame = np.zeros((300, 300, 3), np.uint8)
    frame[100:200, 140:180] = (0, 0, 255)
    crop = body_crop(frame, (140, 100, 180, 200))
    assert crop.shape == (BODY[1], BODY[0], 3)
    assert crop[..., 2].mean() > 150                       # mostly the player, a little margin around him
    assert body_crop(frame, (10, 10, 12, 14)) is None


def test_book_knows_a_player_again_and_declines_a_toss_up():
    rng = np.random.default_rng(0)
    book = LookBook()
    for number in (21, 26, 33):
        book.add("CLE", number, np.stack([look(number, 0.3, rng) for _ in range(10)]))
    assert book.known("CLE") == {21, 26, 33} and book.known("CAR") == set()
    assert book.recognise("CLE", [21, 26, 33], look(26, 0.3, rng)) == 26
    assert book.recognise("CLE", [21, 33], look(26, 0.3, rng)) is None             # he isn't among the candidates
    assert book.recognise("CLE", [21, 26, 33], look(99)) is None                   # a stranger looks like nobody
    ranked = book.match("CLE", [21, 26, 33, 49], look(21))
    assert [n for n, _ in ranked][0] == 21 and len(ranked) == 3                     # 49 was never seen


def test_book_survives_a_save(tmp_path):
    book = LookBook()
    book.add("CAR", 2, look(2))
    book.add("CAR", 20, look(20))
    book.save(tmp_path / "looks.npz")
    again = LookBook.load(tmp_path / "looks.npz")
    assert again.known("CAR") == {2, 20} and again.looks[("CAR", 2)][1] == 1
    assert again.recognise("CAR", [2, 20], look(2)) == 2
    assert again.recognise("CAR", [2], look(2)) is None            # one candidate: nobody to beat, no name


class PixelEmbedder:
    """Stands in for the network: a player's look is decided by the pixel under his box."""

    def embed(self, frame, boxes):
        return np.arange(len(boxes)), np.stack([look(int(frame[int(b[1]) + 1, int(b[0]), 0])) for b in boxes])


def test_shot_looks_average_every_third_frame_and_follow_the_tracker():
    frame = np.zeros((30, 300, 3), np.uint8)
    frame[1, 0], frame[1, 10] = 7, 8
    shot = ShotLooks(PixelEmbedder(), every=3)
    for _ in range(9):
        shot.update(frame, [(0, 0, 8, 20), (10, 0, 18, 20)], ["a", "b"])
    assert shot.counts == {"a": 3, "b": 3}
    assert shot.look("a", min_frames=4) is None
    assert float(shot.look("a") @ look(7)) > 0.99
    assert shot.looks("a").shape == (3, DIM)
    shot.move("a", "c")
    assert "a" not in shot.counts and shot.counts["c"] == 3 and len(shot.recent["c"]) == 3


def test_a_vote_over_frames_outlasts_a_track_that_drifted():
    rng = np.random.default_rng(1)
    book = LookBook()
    for number in (21, 26):
        book.add("CLE", number, np.stack([look(number, 0.3, rng) for _ in range(10)]))
    frames = np.stack([look(21, 0.3, rng) for _ in range(12)] + [look(26, 0.3, rng) for _ in range(4)])
    assert book.recognise_many("CLE", [21, 26], frames) == 21
    split = np.stack([look(21, 0.3, rng) for _ in range(6)] + [look(26, 0.3, rng) for _ in range(6)])
    assert book.recognise_many("CLE", [21, 26], split) is None            # half and half: nobody
    assert book.recognise_many("CLE", [21, 26], frames[:3]) is None       # too few frames to say


class PixelReader:
    def read(self, frame, boxes):
        probs = []
        for b in boxes:
            n = int(frame[int(b[1]), int(b[0]), 0])
            p = np.full(100, 0.005)
            if n != 255:
                p[:] = 0.001
                p[n] = 0.9
            probs.append(p)
        return np.arange(len(boxes)), np.array(probs), np.zeros(len(boxes))


def _record():
    header = PlayHeader(quarter=4, clock="08:21", down=2, distance=7, offense="GB", side="ATL", yard=15)
    rec = PlayRecord(header, matched=True, defense="ATL", yards_to_goal=15, play_type="pass", charted={"qb_location": "S"})
    rec.on_field = OnField(offense=[Listed(10, "QB", "Jordan Love"), Listed(80, "WR", "Matthew Golden"), Listed(85, "TE", "Tucker Kraft"),
                                    Listed(30, "RB", "Josh Jacobs")],
                           defense=[Listed(21, "CB", "A.J. Terrell"), Listed(3, "FS", "Jessie Bates")])
    return rec


def test_a_player_named_by_number_once_is_known_by_his_look_on_the_next_play():
    frames, _ = film(direction=-1, los_rel=40.0)
    an = PlayAnalyzer(None, record=_record(), number_reader=PixelReader(), look_embedder=PixelEmbedder())
    # play one: two receivers' numbers (80, 85) are readable; every player has his own look (pixel row 1)
    img = np.full((30, 300, 3), 255, np.uint8)
    for i in range(22):
        img[1, 10 * i] = 100 + i
    img[0, 80], img[0, 100] = 80, 85
    for f in frames:
        an.update(img, f)
    label = {s.track_id: an.labels.get(s.id) for s in an.tracker.slots}
    assert label[8] == "Golden" and label[10] == "Kraft"
    # play two, a new shot: nobody's number can be read, but the same looks are there
    img[0, 80] = img[0, 100] = 255
    second = []
    for f in frames:
        g = type(f)(**{**f.__dict__, "shot": f.shot + 1, "t": f.t + 100.0})
        second.append(g)
    for g in second:
        an.update(img, g)
    assert an.book.known("GB") == {80, 85}
    label = {s.track_id: an.labels.get(s.id) for s in an.tracker.slots}
    assert label[8] == "Golden" and label[10] == "Kraft"
    assert an.summary()["looks"] == {"Golden": 80, "Kraft": 85} and an.summary()["numbers"] == {}


def test_gather_takes_only_players_named_by_number(tmp_path):
    import json

    from nfl_intel.looks_train import gather
    from nfl_intel.schema import FrameResult, TrackedPlayer, write_csv

    frames = []
    for i in range(10):
        frames.append(FrameResult(frame=i, t=i / 30, size=(100, 100), shot=1, players=[
            TrackedPlayer(1, 0, (0, 0, 10, 20), 0.9, (5, 20), (50.0, 26.0), role="REC", label="Golden"),
            TrackedPlayer(2, 1, (20, 0, 30, 20), 0.9, (25, 20), (52.0, 26.0), role="DEF", label="Terrell"),
            TrackedPlayer(3, 0, (40, 0, 50, 20), 0.9, (45, 20), (50.0, 30.0), role="REC", label="R2")]))
    write_csv(tmp_path / "run.csv", frames)
    (tmp_path / "run.plays.json").write_text(json.dumps([
        {"shot": 1, "offense": "GB", "defense": "ATL", "numbers": {"Golden": 80}, "looks": {"Terrell": 21}}]))
    rows = gather(tmp_path / "run.csv", per_track=4)
    assert len(rows) == 4 and {(r["team"], r["number"]) for r in rows} == {("GB", 80)}
    assert sorted(r["frame"] for r in rows) == [0, 3, 6, 9]
