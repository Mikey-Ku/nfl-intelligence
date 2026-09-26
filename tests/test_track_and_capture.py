import numpy as np
import pytest
import supervision as sv

from nfl_intel.capture import Region, looks_blank
from nfl_intel.track import PlayerTracker


def _dets(boxes):
    boxes = np.array(boxes, dtype=float).reshape(-1, 4)
    return sv.Detections(xyxy=boxes, confidence=np.full(len(boxes), 0.9), class_id=np.zeros(len(boxes), dtype=int))


def _textured_frame(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.mark.parametrize("kind", ["bytetrack", "botsort"])
def test_tracker_keeps_ids_on_moving_boxes(kind):
    tracker = PlayerTracker(fps=30, kind=kind)
    frame = _textured_frame()
    ids_by_frame = []
    for f in range(10):
        dets = tracker.update(_dets([(100 + 4 * f, 100, 140 + 4 * f, 200), (500, 300 + 3 * f, 540, 400 + 3 * f)]), frame)
        if dets.tracker_id is not None and len(dets):
            ids_by_frame.append(tuple(sorted(dets.tracker_id.tolist())))
    assert ids_by_frame, "tracker never returned ids"
    assert len(set(ids_by_frame[-5:])) == 1, "ids changed while boxes moved smoothly"
    tracker.reset()
    assert len(tracker.update(_dets([]), frame)) == 0


def test_unknown_tracker_rejected():
    with pytest.raises(ValueError):
        PlayerTracker(kind="deepsort")


def test_region_parse():
    r = Region.parse("10, 20, 300,400")
    assert r == Region(10, 20, 300, 400)
    assert r.as_mss() == {"left": 10, "top": 20, "width": 300, "height": 400}
    with pytest.raises(ValueError):
        Region.parse("1,2,3")
    with pytest.raises(ValueError):
        Region.parse("1,2,0,4")


def test_looks_blank():
    assert looks_blank(np.zeros((10, 10, 3), dtype=np.uint8))
    rng = np.random.default_rng(0)
    assert not looks_blank(rng.integers(0, 255, (10, 10, 3), dtype=np.uint8))
