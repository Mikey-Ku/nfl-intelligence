import math

from nfl_intel.evaluate import evaluate, evaluate_files
from nfl_intel.schema import FrameResult, TrackedPlayer, write_csv


def _player(tid, x, y, field=True):
    xy = (x, y)
    return TrackedPlayer(tid, 0, (x - 5, y - 20, x + 5, y), 0.9, xy, xy if field else None)


def _seq(ids, offset=0.0, field=True, frames=6):
    out = []
    for f in range(frames):
        players = [_player(tid, 30.0 + 2 * f + offset, 10.0 + 5 * k, field) for k, tid in enumerate(ids)]
        out.append(FrameResult(f, f / 30, (1280, 720), players))
    return out


def test_perfect_match():
    truth = _seq([1, 2, 3])
    r = evaluate(_seq([7, 8, 9]), truth)
    assert r.unit == "yd"
    assert r.position_error == 0.0
    assert r.recall == 1.0 and r.precision == 1.0
    assert r.id_switches == 0 and r.mota == 1.0


def test_offset_error_and_miss():
    truth = _seq([1, 2, 3])
    pred = _seq([7, 8], offset=1.5)          # third player never found, others 1.5 yd off
    r = evaluate(pred, truth)
    assert math.isclose(r.position_error, 1.5, abs_tol=1e-6)
    assert math.isclose(r.recall, 2 / 3)
    assert r.precision == 1.0
    assert r.misses == 6 and r.false_positives == 0


def test_id_switch_counted():
    truth = _seq([1, 2])
    pred = _seq([7, 8], frames=3) + [
        FrameResult(f, f / 30, (1280, 720), [_player(99, 30.0 + 2 * f, 10.0), _player(8, 30.0 + 2 * f, 15.0)])
        for f in range(3, 6)
    ]
    r = evaluate(pred, truth)
    assert r.id_switches == 1
    assert r.mota == 1 - 1 / 12


def test_pixel_fallback_and_files(tmp_path):
    truth = _seq([1, 2], field=False)
    pred = _seq([5, 6], offset=10.0, field=False)
    write_csv(tmp_path / "t.csv", truth)
    write_csv(tmp_path / "p.csv", pred)
    r = evaluate_files(tmp_path / "p.csv", tmp_path / "t.csv")
    assert r.unit == "px"
    assert math.isclose(r.position_error, 10.0, abs_tol=1e-6)
    assert "recall 100.00%" in str(r)


def test_a_steady_offset_is_reported_and_does_not_shuffle_who_is_who():
    """Men a yard apart, every one of them placed 0.9 yd toward his neighbour. Paired raw, a
    man is nearer the next man's truth than his own; that must not read as identities
    changing hands."""
    truth, pred = [], []
    for f in range(8):
        wobble = 0.1 if f % 2 else -0.1                                  # one frame it tips one way, the next the other
        truth.append(FrameResult(f, f / 2, (1280, 720), [_player(k + 1, 30.0, 20.0 + k) for k in range(6)]))
        pred.append(FrameResult(f, f / 2, (1280, 720), [_player(50 + k, 30.0, 20.0 + k + 0.9 + (wobble if k % 2 else -wobble)) for k in range(6)]))
    r = evaluate(pred, truth)
    assert r.id_switches == 0 and r.matched == 48
    assert math.isclose(r.offset[1], 0.9, abs_tol=0.11) and abs(r.offset[0]) < 1e-6
    assert r.position_error > 0.8 and r.spread < 0.2
    assert "steady offset" in str(r)


def test_identity_is_the_slot_when_the_run_has_one():
    truth = _seq([1, 2])
    pred = _seq([7, 8])
    for f, res in enumerate(pred):
        for k, p in enumerate(res.players):
            p.track_id = 100 * f + k                                     # the image tracker loses and finds them every frame
            p.slot = k                                                   # the field tracker keeps them
    assert evaluate(pred, truth).id_switches == 0
    for res in pred:
        for p in res.players:
            p.slot = None
    assert evaluate(pred, truth).id_switches == 10


def test_the_slot_survives_the_csv(tmp_path):
    from nfl_intel.schema import read_csv

    pred = _seq([7, 8], frames=2)
    pred[0].players[0].slot = 4
    write_csv(tmp_path / "run.csv", pred)
    back = read_csv(tmp_path / "run.csv")
    assert back[0].players[0].slot == 4 and back[0].players[1].slot is None


def test_truth_is_put_into_the_yards_of_the_run_being_scored():
    """Labelled over a run that had the play the other way round; scored against one that has it right."""
    from nfl_intel.evaluate import through_the_run

    def frame(f, flip):
        players = []
        for k in range(6):
            px = (100.0 + 40 * k + 3 * f, 300.0 + 25 * (k % 3))
            yd = (20.0 + px[0] / 20, 10.0 + px[1] / 20)
            if flip:
                yd = (120.0 - yd[0], 53.3 - yd[1])
            players.append(TrackedPlayer(k + 1, 0, (px[0] - 5, px[1] - 20, px[0] + 5, px[1]), 0.9, px, yd))
        return FrameResult(f, f / 2, (1280, 720), players)

    truth = [frame(f, flip=True) for f in range(5)]           # yards as the mirrored run gave them
    pred = [frame(f, flip=False) for f in range(5)]
    assert evaluate(pred, truth).recall < 0.5                   # as stored: nowhere near
    r = evaluate(pred, through_the_run(pred, truth))
    assert r.recall == 1.0 and r.position_error < 0.01 and r.id_switches == 0
