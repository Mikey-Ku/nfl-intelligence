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
