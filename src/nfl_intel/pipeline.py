"""PlayTracker: detector + tracker + team clustering (+ field calibration) per frame."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import supervision as sv

from .detect import DEFAULT_WEIGHTS, PlayerDetector
from .field import FieldCalibration
from .schema import FrameResult, TrackedPlayer
from .teams import TeamAssigner
from .track import PlayerTracker


class PlayTracker:
    def __init__(
        self,
        detector: PlayerDetector | None = None,
        fps: float = 30.0,
        calibration: FieldCalibration | None = None,
        weights: str = DEFAULT_WEIGHTS,
        device: str | None = None,
        conf: float = 0.3,
        imgsz: int = 1280,
        min_box_height_frac: float = 0.03,
        team_warmup_frames: int = 15,
    ):
        self.detector = detector or PlayerDetector(weights, device=device, conf=conf, imgsz=imgsz)
        self.tracker = PlayerTracker(fps=fps)
        self.teams = TeamAssigner(warmup_frames=team_warmup_frames)
        self.calibration = calibration
        self.min_box_height_frac = min_box_height_frac

    def reset(self) -> None:
        """Call between plays so IDs and team clusters start fresh."""
        self.tracker.reset()
        self.teams.reset()

    def _filter(self, dets: sv.Detections, frame_h: int) -> sv.Detections:
        if len(dets) == 0:
            return dets
        heights = dets.xyxy[:, 3] - dets.xyxy[:, 1]
        return dets[heights >= self.min_box_height_frac * frame_h]

    def process(self, frame: np.ndarray, frame_idx: int, t: float) -> FrameResult:
        h, w = frame.shape[:2]
        dets = self._filter(self.detector(frame), h)
        dets = self.tracker.update(dets)
        players: list[TrackedPlayer] = []
        if len(dets) and dets.tracker_id is not None:
            teams = self.teams.assign(frame, dets.xyxy, dets.tracker_id)
            feet = np.column_stack([(dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2, dets.xyxy[:, 3]])
            field = self.calibration.to_field(feet) if self.calibration is not None else None
            for i in range(len(dets)):
                players.append(TrackedPlayer(
                    track_id=int(dets.tracker_id[i]),
                    team=teams[i],
                    box=tuple(float(v) for v in dets.xyxy[i]),
                    conf=float(dets.confidence[i]) if dets.confidence is not None else 0.0,
                    foot_px=(float(feet[i, 0]), float(feet[i, 1])),
                    field_yd=None if field is None else (float(field[i, 0]), float(field[i, 1])),
                ))
        return FrameResult(frame=frame_idx, t=t, size=(w, h), players=players)

    def run(self, source: Iterable[tuple[int, float, np.ndarray]]) -> Iterator[FrameResult]:
        for idx, t, frame in source:
            yield self.process(frame, idx, t)
