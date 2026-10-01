"""PlayTracker: field crop + detector + tracker + team clustering (+ calibration) per frame."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import cv2
import numpy as np
import supervision as sv

from .detect import DEFAULT_WEIGHTS, PlayerDetector
from .field import FieldCalibration
from .fieldmask import FieldInfo, expand_bbox, find_field
from .imaging import shrink
from .registration import FieldRegistrar
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
        cut_threshold: float = 10.0,
        register_field: bool = True,
    ):
        self.detector = detector or PlayerDetector(weights, device=device, conf=conf, imgsz=imgsz, tiles=tiles)
        self.tracker = PlayerTracker(fps=fps, kind=tracker)
        self.teams = TeamAssigner(warmup_frames=team_warmup_frames)
        self.calibration = calibration
        self.min_box_height_frac = min_box_height_frac
        self.field_every = field_every
        self.on_field_only = on_field_only
        self.cut_threshold = cut_threshold
        # a fixed calibration file wins; otherwise the field is registered from its painted lines
        self.registrar = FieldRegistrar() if (register_field and calibration is None) else None
        self.field: FieldInfo | None = None
        self.shot = 0
        self._frames_seen = 0
        self._thumb: np.ndarray | None = None
        self._id_offset = 0
        self._max_id = -1
        self._field_votes: dict[int, list[int]] = {}

    def reset(self) -> None:
        """Start a new shot: fresh tracks, team clusters and field. Track IDs keep counting up
        so an ID never refers to two different players in one session."""
        self.tracker.reset()
        self.teams.reset()
        self.field = None
        self._frames_seen = 0
        self._id_offset = self._max_id + 1
        self._field_votes = {}
        if self.registrar is not None:
            self.registrar.reset()

    def _is_cut(self, frame: np.ndarray) -> bool:
        """A camera cut: the whole picture changes at once. Pans and zooms move the mean
        thumbnail difference by 1-4 grey levels; cuts on NFL Pro film move it by 20+."""
        thumb = cv2.cvtColor(shrink(frame[::4, ::4], (96, 54)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev, self._thumb = self._thumb, thumb
        return prev is not None and float(np.abs(thumb - prev).mean()) > self.cut_threshold

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
        """Drop boxes too small to be a player on the field (crowd, far background)."""
        if len(dets) == 0:
            return dets
        heights = dets.xyxy[:, 3] - dets.xyxy[:, 1]
        return dets[heights >= self.min_box_height_frac * frame_shape[0]]

    def _on_field(self, dets: sv.Detections) -> np.ndarray:
        """Which tracks are players on the field, decided per track by majority vote.

        Everyone is tracked, bench included, so a player near the sideline doesn't drop out
        and come back under a new ID; the vote then hides tracks that mostly stand off the field.
        """
        if not self.on_field_only or self.field is None:
            return np.ones(len(dets), dtype=bool)
        feet_x = (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2
        feet_y = dets.xyxy[:, 3]
        keep = np.zeros(len(dets), dtype=bool)
        for i, (tid, x, y) in enumerate(zip(dets.tracker_id, feet_x, feet_y)):
            votes = self._field_votes.setdefault(int(tid), [0, 0])
            votes[0] += int(self.field.foot_on_field(x, y))
            votes[1] += 1
            keep[i] = votes[0] * 2 >= votes[1]
        return keep

    def process(self, frame: np.ndarray, frame_idx: int, t: float) -> FrameResult:
        h, w = frame.shape[:2]
        if self._is_cut(frame):
            self.reset()
            self.shot += 1
        self._update_field(frame)
        dets = self._filter(self._detect(frame), frame.shape)
        mapping = None
        if self.registrar is not None and self.field is not None:
            mapping = self.registrar.update(frame, self.field, dets.xyxy if len(dets) else None)
        dets = self.tracker.update(dets, frame)
        unconfirmed = None
        if len(dets) and dets.tracker_id is not None:
            unconfirmed = dets[dets.tracker_id < 0]    # BoT-SORT marks unconfirmed tracks with -1
            dets = dets[dets.tracker_id >= 0]
        if len(dets) and dets.tracker_id is not None:
            dets.tracker_id = dets.tracker_id + self._id_offset
            self._max_id = max(self._max_id, int(dets.tracker_id.max()))
            dets = dets[self._on_field(dets)]
        players: list[TrackedPlayer] = []
        if len(dets) and dets.tracker_id is not None:
            grass = self.field.grass_lab if self.field is not None else None
            teams = self.teams.assign(frame, dets.xyxy, dets.tracker_id, grass)
            feet = np.column_stack([(dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2, dets.xyxy[:, 3]])
            if self.calibration is not None:
                field = self.calibration.to_field(feet)
            elif mapping is not None:
                field = cv2.perspectiveTransform(feet.reshape(-1, 1, 2).astype(np.float64), mapping).reshape(-1, 2)
            else:
                field = None
            for i in range(len(dets)):
                players.append(TrackedPlayer(
                    track_id=int(dets.tracker_id[i]),
                    team=teams[i],
                    box=tuple(float(v) for v in dets.xyxy[i]),
                    conf=float(dets.confidence[i]) if dets.confidence is not None else 0.0,
                    foot_px=(float(feet[i, 0]), float(feet[i, 1])),
                    field_yd=None if field is None else (float(field[i, 0]), float(field[i, 1])),
                ))
        loose = self._loose(frame, unconfirmed, mapping)
        return FrameResult(frame=frame_idx, t=t, size=(w, h), players=players, shot=self.shot, loose=loose)

    def _loose(self, frame: np.ndarray, dets, mapping) -> list[TrackedPlayer]:
        """People on the field that the image tracker hasn't confirmed yet. On a mapped field
        they are worth keeping: tracking in yards can tell who they are when the picture can't."""
        if dets is None or len(dets) == 0 or mapping is None or self.field is None:
            return []
        feet = np.column_stack([(dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2, dets.xyxy[:, 3]])
        on = np.array([bool(self.field.foot_on_field(x, y)) for x, y in feet])
        if not on.any():
            return []
        boxes, feet = dets.xyxy[on], feet[on]
        conf = dets.confidence[on] if dets.confidence is not None else np.zeros(len(boxes))
        teams = self.teams.look(frame, boxes, self.field.grass_lab)
        yards = cv2.perspectiveTransform(feet.reshape(-1, 1, 2).astype(np.float64), mapping).reshape(-1, 2)
        return [TrackedPlayer(track_id=-1, team=teams[i], box=tuple(float(v) for v in boxes[i]), conf=float(conf[i]),
                              foot_px=(float(feet[i, 0]), float(feet[i, 1])), field_yd=(float(yards[i, 0]), float(yards[i, 1])))
                for i in range(len(boxes))]

    def run(self, source: Iterable[tuple[int, float, np.ndarray]]) -> Iterator[FrameResult]:
        for idx, t, frame in source:
            yield self.process(frame, idx, t)
