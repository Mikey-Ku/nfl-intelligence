"""PlayTracker: one frame in, a FrameResult out.

Per frame: notice a camera cut, find the grass, detect people on it, keep their ids, colour
them by team, and map their feet to yards through the field registration (or a hand-made
calibration). Detections the image tracker hasn't confirmed yet are passed on as `loose`
for the yard-space tracker to use.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

import cv2
import numpy as np
import supervision as sv

from .detect import DEFAULT_WEIGHTS, PlayerDetector
from .field import FieldCalibration
from .fieldmask import FieldInfo, border_edges, expand_bbox, find_field
from .imaging import shrink
from .registration import FieldRegistrar
from .schema import FrameResult, TrackedPlayer
from .teams import TeamAssigner
from .track import PlayerTracker


OUT_OF_BOUNDS = 0.75        # yards past the border's inner edge at which feet are off the field


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
        self._edges: tuple[deque, deque] = (deque(maxlen=9), deque(maxlen=9))   # the border, in the mapping's yards

    def reset(self) -> None:
        """Start a new shot: fresh tracks, team clusters and field. Track IDs keep counting up
        so an ID never refers to two different players in one session."""
        self.tracker.reset()
        self.teams.reset()
        self.field = None
        self._frames_seen = 0
        self._id_offset = self._max_id + 1
        self._field_votes = {}
        self._edges = (deque(maxlen=9), deque(maxlen=9))
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

    def _measure_border(self, frame: np.ndarray, mapping) -> None:
        """Every so often, find the white border through the mapping and remember where it was."""
        if mapping is None or self.registrar is None or self.registrar.rect is None or self._frames_seen % 5 != 1:
            return
        for kept, edge in zip(self._edges, border_edges(frame, mapping, self.registrar.rect)):
            if edge is not None:
                kept.append(edge)

    def _in_bounds(self, feet: np.ndarray, yards: np.ndarray | None) -> np.ndarray:
        """Whose feet are on the field. With a mapping and a measured border the test is in
        yards: inside the border's inner edge, give or take the foot point's own error.
        Without them it falls back to the painted sideline found in the picture."""
        if self.field is None:
            return np.ones(len(feet), dtype=bool)
        on = np.array([bool(self.field.contains_foot(x, y)) for x, y in feet], dtype=bool)
        low, high = (float(np.median(e)) if len(e) >= 3 else None for e in self._edges)
        if yards is None or (low is None and high is None):
            return np.array([bool(self.field.foot_on_field(x, y)) for x, y in feet], dtype=bool)
        if low is not None:
            on &= yards[:, 1] >= low - OUT_OF_BOUNDS
        if high is not None:
            on &= yards[:, 1] <= high + OUT_OF_BOUNDS
        return on

    def _on_field(self, dets: sv.Detections, yards: np.ndarray | None = None) -> np.ndarray:
        """Which tracks are players on the field, decided per track by majority vote.

        Everyone is tracked, bench included, so a player near the sideline doesn't drop out
        and come back under a new ID; the vote then hides tracks that mostly stand off the
        field. A track that starts off the field is hidden from its first frame.
        """
        if not self.on_field_only or self.field is None:
            return np.ones(len(dets), dtype=bool)
        feet = np.column_stack([(dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2, dets.xyxy[:, 3]])
        inside = self._in_bounds(feet, yards)
        keep = np.zeros(len(dets), dtype=bool)
        for i, tid in enumerate(dets.tracker_id):
            votes = self._field_votes.setdefault(int(tid), [0, 0])
            votes[0] += int(inside[i])
            votes[1] += 1
            keep[i] = votes[0] * 2 >= votes[1]
        return keep

    def _to_yards(self, feet: np.ndarray, mapping) -> np.ndarray | None:
        if self.calibration is not None:
            return self.calibration.to_field(feet)
        if mapping is not None and len(feet):
            return cv2.perspectiveTransform(feet.reshape(-1, 1, 2).astype(np.float64), mapping).reshape(-1, 2)
        return None

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
            self._measure_border(frame, mapping)
        dets = self.tracker.update(dets, frame)
        unconfirmed = None
        if len(dets) and dets.tracker_id is not None:
            unconfirmed = dets[dets.tracker_id < 0]    # BoT-SORT marks unconfirmed tracks with -1
            dets = dets[dets.tracker_id >= 0]
        if len(dets) and dets.tracker_id is not None:
            dets.tracker_id = dets.tracker_id + self._id_offset
            self._max_id = max(self._max_id, int(dets.tracker_id.max()))
            feet = np.column_stack([(dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2, dets.xyxy[:, 3]])
            dets = dets[self._on_field(dets, self._to_yards(feet, mapping))]
        players: list[TrackedPlayer] = []
        if len(dets) and dets.tracker_id is not None:
            grass = self.field.grass_lab if self.field is not None else None
            teams = self.teams.assign(frame, dets.xyxy, dets.tracker_id, grass)
            feet = np.column_stack([(dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2, dets.xyxy[:, 3]])
            field = self._to_yards(feet, mapping)
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
        yards = cv2.perspectiveTransform(feet.reshape(-1, 1, 2).astype(np.float64), mapping).reshape(-1, 2)
        on = self._in_bounds(feet, yards)
        if not on.any():
            return []
        boxes, feet, yards = dets.xyxy[on], feet[on], yards[on]
        conf = dets.confidence[on] if dets.confidence is not None else np.zeros(len(boxes))
        teams = self.teams.look(frame, boxes, self.field.grass_lab)
        return [TrackedPlayer(track_id=-1, team=teams[i], box=tuple(float(v) for v in boxes[i]), conf=float(conf[i]),
                              foot_px=(float(feet[i, 0]), float(feet[i, 1])), field_yd=(float(yards[i, 0]), float(yards[i, 1])))
                for i in range(len(boxes))]

    def run(self, source: Iterable[tuple[int, float, np.ndarray]]) -> Iterator[FrameResult]:
        for idx, t, frame in source:
            yield self.process(frame, idx, t)
