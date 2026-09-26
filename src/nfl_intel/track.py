"""Stable player IDs across frames via ByteTrack (Roboflow `trackers` package)."""

from __future__ import annotations

import supervision as sv
from trackers import ByteTrackTracker


class PlayerTracker:
    def __init__(
        self,
        fps: float = 30.0,
        activation_threshold: float = 0.25,
        lost_seconds: float = 1.0,
        min_iou: float = 0.1,
        min_consecutive_frames: int = 2,
    ):
        self.fps = fps
        self._kwargs = dict(
            lost_track_buffer=max(1, int(round(lost_seconds * fps))),
            frame_rate=fps,
            track_activation_threshold=activation_threshold,
            high_conf_det_threshold=max(activation_threshold, 0.5),
            minimum_iou_threshold=min_iou,
            minimum_consecutive_frames=min_consecutive_frames,
        )
        self.reset()

    def reset(self) -> None:
        self.tracker = ByteTrackTracker(**self._kwargs)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Returns the detections that got a track, with `tracker_id` filled in."""
        return self.tracker.update(detections)
