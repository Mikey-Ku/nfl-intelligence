"""Stable player IDs across frames (Roboflow `trackers` package).

Two choices:
  bytetrack  — fast IoU + Kalman association; fine for a fixed camera.
  botsort    — adds camera-motion compensation (sparse optical flow on the background), which
               matters on All-22 film where the sideline camera pans with the play.
"""

from __future__ import annotations

import supervision as sv
from trackers import BoTSORTTracker, ByteTrackTracker

TRACKERS = ("bytetrack", "botsort")


class PlayerTracker:
    def __init__(
        self,
        fps: float = 30.0,
        kind: str = "botsort",
        activation_threshold: float = 0.25,
        lost_seconds: float = 1.0,
        min_iou: float = 0.1,
        min_consecutive_frames: int = 2,
    ):
        if kind not in TRACKERS:
            raise ValueError(f"tracker must be one of {TRACKERS}, got {kind!r}")
        self.fps = fps
        self.kind = kind
        common = dict(
            lost_track_buffer=max(1, int(round(lost_seconds * fps))),
            frame_rate=fps,
            track_activation_threshold=activation_threshold,
            high_conf_det_threshold=max(activation_threshold, 0.5),
            minimum_consecutive_frames=min_consecutive_frames,
        )
        if kind == "bytetrack":
            self._factory = lambda: ByteTrackTracker(minimum_iou_threshold=min_iou, **common)
        else:
            self._factory = lambda: BoTSORTTracker(
                minimum_iou_threshold_first_assoc=max(min_iou, 0.2),
                enable_cmc=True,
                cmc_method="sparseOptFlow",
                **common,
            )
        self.reset()

    def reset(self) -> None:
        self.tracker = self._factory()

    def update(self, detections: sv.Detections, frame=None) -> sv.Detections:
        """Returns the detections that got a track, with `tracker_id` filled in.

        BoT-SORT needs the frame for camera-motion compensation; ByteTrack ignores it.
        """
        if self.kind == "botsort":
            return self.tracker.update(detections, frame=frame)
        return self.tracker.update(detections)
