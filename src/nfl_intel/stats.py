"""Live numbers for the side panel, computed from the tracking stream.

Three tiers, each filling in as the pipeline gains a capability:
  now              players on the field per team, fps, tracking health, play segmentation
  with calibration metrics in yards (formation, box count, separation...) via `metrics_fn`
  with a model     play value frame by frame, its change since the snap (the plus/minus),
                   and the biggest swings, via `value_fn`

`metrics_fn(result) -> list[(label, value)]` and `value_fn(result) -> float in [0, 1]` are
plugged in by the caller; both only run on frames that have field coordinates.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from .schema import FrameResult

MIN_PLAYERS_FOR_PLAY = 6


@dataclass
class Swing:
    t: float          # seconds into the play
    delta: float      # change in value, -1..1
    label: str = ""   # who/what, once credit assignment exists


@dataclass
class StatsSnapshot:
    t: float
    fps: float
    play: int
    play_clock: float
    on_field: int
    team_counts: tuple[int, int, int]              # team 0, team 1, unknown
    active_ids: int
    reacquired: int                                # tracks that appeared after the play settled
    lost: int                                      # tracks seen this play and now missing
    history: list[tuple[float, int]]               # (seconds ago, players on field)
    calibrated: bool
    rows: list[tuple[str, str]] = field(default_factory=list)
    value: float | None = None
    value_delta: float | None = None
    value_history: list[tuple[float, float]] = field(default_factory=list)   # (play clock, value)
    swings: list[Swing] = field(default_factory=list)
    field_bbox: tuple[int, int, int, int] | None = None


class LiveStats:
    def __init__(
        self,
        window: float = 10.0,
        settle: float = 1.0,
        gap_reset: float = 1.0,
        lost_after: float = 0.5,
        value_fn: Callable[[FrameResult], float] | None = None,
        metrics_fn: Callable[[FrameResult], list[tuple[str, str]]] | None = None,
        swing_window: float = 0.5,
    ):
        self.window = window
        self.settle = settle
        self.gap_reset = gap_reset
        self.lost_after = lost_after
        self.value_fn = value_fn
        self.metrics_fn = metrics_fn
        self.swing_window = swing_window
        self.play = 0
        self._fps = 0.0
        self._last_t: float | None = None
        self._history: deque[tuple[float, int]] = deque()
        self._in_play = False
        self._empty_since: float | None = None
        self._shot: int | None = None
        self._start_play(0.0)

    def _start_play(self, t: float) -> None:
        self._play_start = t
        self._first_seen: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}
        self._values: list[tuple[float, float]] = []

    def _segment(self, t: float, n_players: int, shot: int) -> None:
        """A play starts when enough players appear, and ends at a camera cut or after the
        field has been empty a while."""
        if self._shot is not None and shot != self._shot:
            self._in_play = False
        self._shot = shot
        if n_players >= MIN_PLAYERS_FOR_PLAY:
            self._empty_since = None
            if not self._in_play:
                self._in_play = True
                self.play += 1
                self._start_play(t)
        else:
            if self._empty_since is None:
                self._empty_since = t
            elif self._in_play and t - self._empty_since >= self.gap_reset:
                self._in_play = False

    def _swings(self) -> list[Swing]:
        """The three largest changes in value over `swing_window` seconds, non-overlapping."""
        vals = self._values
        if len(vals) < 3:
            return []
        changes = []
        j = 0
        for i in range(len(vals)):
            while vals[i][0] - vals[j][0] > self.swing_window:
                j += 1
            if i != j:
                changes.append(Swing(t=vals[i][0], delta=vals[i][1] - vals[j][1]))
        changes.sort(key=lambda s: -abs(s.delta))
        picked: list[Swing] = []
        for s in changes:
            if abs(s.delta) < 0.01:
                break
            if all(abs(s.t - p.t) > self.swing_window for p in picked):
                picked.append(s)
            if len(picked) == 3:
                break
        return sorted(picked, key=lambda s: s.t)

    def update(self, result: FrameResult, field_bbox: tuple[int, int, int, int] | None = None) -> StatsSnapshot:
        t = result.t
        if self._last_t is not None and t > self._last_t:
            inst = 1.0 / (t - self._last_t)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._last_t = t

        players = result.players
        self._segment(t, len(players), result.shot)
        for p in players:
            self._first_seen.setdefault(p.track_id, t)
            self._last_seen[p.track_id] = t

        self._history.append((t, len(players)))
        while self._history and t - self._history[0][0] > self.window:
            self._history.popleft()

        current = {p.track_id for p in players}
        reacquired = sum(1 for first in self._first_seen.values() if first - self._play_start > self.settle)
        lost = sum(1 for tid, last in self._last_seen.items() if tid not in current and t - last > self.lost_after)
        teams = (
            sum(1 for p in players if p.team == 0),
            sum(1 for p in players if p.team == 1),
            sum(1 for p in players if p.team is None),
        )
        calibrated = bool(players) and all(p.field_yd is not None for p in players)
        clock = max(0.0, t - self._play_start) if self._in_play else 0.0

        rows: list[tuple[str, str]] = []
        value = delta = None
        if calibrated and self._in_play:
            if self.metrics_fn is not None:
                rows = self.metrics_fn(result)
            if self.value_fn is not None:
                value = float(self.value_fn(result))
                self._values.append((clock, value))
                delta = value - self._values[0][1]

        return StatsSnapshot(
            t=t,
            fps=self._fps,
            play=self.play,
            play_clock=clock,
            on_field=len(players),
            team_counts=teams,
            active_ids=len(current),
            reacquired=reacquired,
            lost=lost,
            history=[(t - ht, n) for ht, n in self._history],
            calibrated=calibrated,
            rows=rows,
            value=value,
            value_delta=delta,
            value_history=list(self._values),
            swings=self._swings() if self._values else [],
            field_bbox=field_bbox,
        )
