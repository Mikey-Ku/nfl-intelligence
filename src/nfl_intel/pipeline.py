"""PlayTracker: field crop + detector + tracker + team clustering (+ calibration) per frame."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import supervision as sv

from .detect import DEFAULT_WEIGHTS, PlayerDetector
from .field import FieldCalibration
from .fieldmask import FieldInfo, expand_bbox, find_field
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
        tracker: str = "botsort",
        tiles: int = 1,
        field_every: int = 15,
        on_field_only: bool = True,
    ):
        self.detector = detector or PlayerDetector(weights, device=device, conf=conf, imgsz=imgsz, tiles=tiles)
        self.tracker = PlayerTracker(fps=fps, kind=tracker)
        self.teams = TeamAssigner(warmup_frames=team_warmup_frames)
        self.calibration = calibration
        self.min_box_height_frac = min_box_height_frac
        self.field_every = field_every
        self.on_field_only = on_field_only
        self.field: FieldInfo | None = None
        self._frames_seen = 0

    def reset(self) -> None:
        """Call between plays so IDs and team clusters start fresh."""
        self.tracker.reset()
        self.teams.reset()
        self.field = None
        self._frames_seen = 0

    def _update_field(self, frame: np.ndarray) -> None:
        if self._frames_seen % self.field_every == 0 or self.field is None:
            found = find_field(frame)
            if found is not None or self._frames_seen % self.field_every == 0:
                self.field = found
        self._frames_seen += 1

    def _detect(self, frame: np.ndarray) -> sv.Detections:
        """Run the detector on the field crop (more pixels per player) and map boxes back."""
        if self.field is None:
            return self.detector(frame)
        x1, y1, x2, y2 = expand_bbox(self.field.bbox, frame.shape)
        if (x2 - x1) < frame.shape[1] * 0.3 or (y2 - y1) < frame.shape[0] * 0.3:
            return self.detector(frame)
        dets = self.detector(np.ascontiguousarray(frame[y1:y2, x1:x2]))
        if len(dets):
            dets.xyxy = dets.xyxy + np.array([x1, y1, x1, y1], dtype=dets.xyxy.dtype)
        return dets

    def _filter(self, dets: sv.Detections, frame_shape) -> sv.Detections:
        if len(dets) == 0:
            return dets
        h = frame_shape[0]
        heights = dets.xyxy[:, 3] - dets.xyxy[:, 1]
        keep = heights >= self.min_box_height_frac * h
        if self.on_field_only and self.field is not None:
            feet_x = (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2
            feet_y = dets.xyxy[:, 3]
            on_field = np.array([self.field.foot_on_field(x, y) for x, y in zip(feet_x, feet_y)])
            keep &= on_field
        return dets[keep]

    def process(self, frame: np.ndarray, frame_idx: int, t: float) -> FrameResult:
        h, w = frame.shape[:2]
        self._update_field(frame)
        dets = self._filter(self._detect(frame), frame.shape)
        dets = self.tracker.update(dets, frame)
        if len(dets) and dets.tracker_id is not None:
            dets = dets[dets.tracker_id >= 0]          # BoT-SORT marks unconfirmed tracks with -1
        players: list[TrackedPlayer] = []
        if len(dets) and dets.tracker_id is not None:
            grass = self.field.grass_lab if self.field is not None else None
            teams = self.teams.assign(frame, dets.xyxy, dets.tracker_id, grass)
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
