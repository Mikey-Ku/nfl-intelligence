"""End-to-end smoke test with a fake detector, so it runs without weights or a GPU."""

import numpy as np
import supervision as sv

from nfl_intel.field import FIELD_WIDTH, FieldCalibration
from nfl_intel.pipeline import PlayTracker

GRASS = (40, 160, 60)
STANDS = (90, 90, 90)


class FakeDetector:
    """Returns the boxes it was told to, in the coordinates of whatever image it is given.

    The pipeline hands it a crop around the field, so it records the crop size to prove that.
    """

    def __init__(self, boxes_by_frame):
        self.boxes_by_frame = boxes_by_frame
        self.calls = 0
        self.sizes = []

    def __call__(self, frame):
        self.sizes.append(frame.shape[:2])
        boxes = np.array(self.boxes_by_frame[min(self.calls, len(self.boxes_by_frame) - 1)], dtype=float)
        self.calls += 1
        return sv.Detections(xyxy=boxes, confidence=np.full(len(boxes), 0.9), class_id=np.zeros(len(boxes), dtype=int))


def _stadium_frame():
    frame = np.full((720, 1280, 3), STANDS, dtype=np.uint8)
    frame[150:600, 100:1180] = GRASS                  # the field; stands above and below
    frame[300:420, 200:250] = (240, 240, 240)        # white jersey on the field
    frame[300:420, 600:650] = (30, 60, 20)           # dark green jersey on the field
    return frame


def test_pipeline_tracks_maps_to_field_and_drops_bench():
    # Boxes are in crop coordinates (the crop starts a little above/left of the field), so the
    # pipeline adds the crop offset back. The last box's feet land in the stands below the field.
    frames = [[(100 + 5 * f, 150, 150 + 5 * f, 270), (500, 150, 550, 270), (20, 20, 30, 30), (300, 560, 340, 610)] for f in range(20)]
    cal = FieldCalibration.from_correspondences(
        [(0, 720), (1280, 720), (1280, 0), (0, 0)],
        [(0, 0), (120, 0), (120, FIELD_WIDTH), (0, FIELD_WIDTH)],
    )
    det = FakeDetector(frames)
    tracker = PlayTracker(detector=det, fps=30, calibration=cal, team_warmup_frames=2)
    frame = _stadium_frame()

    results = [tracker.process(frame, i, i / 30) for i in range(20)]
    last = results[-1]
    assert last.size == (1280, 720)
    assert tracker.field is not None
    # the detector saw a crop of the field, not the whole window
    assert det.sizes[-1][0] < 720 and det.sizes[-1][1] <= 1280
    # the 10px box is below min height; the box whose feet are in the stands is dropped
    assert all(len(r.players) == 2 for r in results[2:])
    ids = {p.track_id for p in last.players}
    assert ids == {p.track_id for p in results[-5].players}
    assert all(p.track_id >= 0 for r in results for p in r.players)
    assert all(p.foot_px[1] < 600 for r in results for p in r.players)
    for p in last.players:
        assert p.field_yd is not None
        assert 0 <= p.field_yd[0] <= 120 and 0 <= p.field_yd[1] <= FIELD_WIDTH
        assert p.foot_px == ((p.box[0] + p.box[2]) / 2, p.box[3])
        assert tracker.field.contains_foot(*p.foot_px)
    tracker.reset()
    assert tracker.teams.classifier.fitted is False and tracker.field is None


def test_pipeline_without_field_uses_whole_frame():
    frames = [[(100, 300, 150, 420)]] * 5
    det = FakeDetector(frames)
    tracker = PlayTracker(detector=det, fps=30, team_warmup_frames=2)
    frame = np.full((720, 1280, 3), STANDS, dtype=np.uint8)   # no grass anywhere
    for i in range(5):
        res = tracker.process(frame, i, i / 30)
    assert tracker.field is None
    assert det.sizes[-1] == (720, 1280)
    assert len(res.players) == 1 and res.players[0].field_yd is None
