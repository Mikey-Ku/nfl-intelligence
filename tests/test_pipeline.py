"""End-to-end smoke test with a fake detector, so it runs without weights or a GPU."""

import numpy as np
import supervision as sv

from nfl_intel.field import FIELD_WIDTH, FieldCalibration
from nfl_intel.pipeline import PlayTracker


class FakeDetector:
    """Returns the boxes it was told to for the frame index it's on."""

    def __init__(self, boxes_by_frame):
        self.boxes_by_frame = boxes_by_frame
        self.calls = 0

    def __call__(self, frame):
        boxes = np.array(self.boxes_by_frame[min(self.calls, len(self.boxes_by_frame) - 1)], dtype=float)
        self.calls += 1
        return sv.Detections(xyxy=boxes, confidence=np.full(len(boxes), 0.9), class_id=np.zeros(len(boxes), dtype=int))


def test_pipeline_tracks_and_maps_to_field():
    frames = [[(100 + 5 * f, 300, 150 + 5 * f, 420), (600, 300, 650, 420), (20, 20, 30, 30)] for f in range(20)]
    cal = FieldCalibration.from_correspondences(
        [(0, 720), (1280, 720), (1280, 0), (0, 0)],
        [(0, 0), (120, 0), (120, FIELD_WIDTH), (0, FIELD_WIDTH)],
    )
    tracker = PlayTracker(detector=FakeDetector(frames), fps=30, calibration=cal, team_warmup_frames=2)
    frame = np.full((720, 1280, 3), (40, 160, 60), dtype=np.uint8)
    frame[300:420, 100:250] = (240, 240, 240)
    frame[300:420, 600:650] = (30, 30, 200)

    results = [tracker.process(frame, i, i / 30) for i in range(20)]
    last = results[-1]
    assert last.size == (1280, 720)
    # the 10px box is below min height and is dropped
    assert len(last.players) == 2
    ids = {p.track_id for p in last.players}
    assert ids == {p.track_id for p in results[-5].players}
    for p in last.players:
        assert p.field_yd is not None
        assert 0 <= p.field_yd[0] <= 120 and 0 <= p.field_yd[1] <= FIELD_WIDTH
    # foot is the bottom-centre of the box
    p0 = next(p for p in last.players if p.box[0] < 400)
    assert p0.foot_px == ((p0.box[0] + p0.box[2]) / 2, p0.box[3])
    tracker.reset()
    assert tracker.teams.classifier.fitted is False
