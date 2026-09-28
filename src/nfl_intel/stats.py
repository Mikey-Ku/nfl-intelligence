"""Live numbers for the side panel, computed from the tracking stream.

Three tiers, each filling in as the pipeline gains a capability:
  now              players on the field per team, fps, tracking health, play segmentation
  with calibration metrics in yards (formation, box count, separation...) via `metrics_fn`
  with a model     play value frame by frame, its change since the snap (the plus/minus),
                   and the biggest swings, via `value_fn`

`metrics_fn(result) -> list[(label, value)]` and `value_fn(result) -> float` are plugged in by
the caller; both only run on frames that have field coordinates. The value is a fraction
(shown as a percentage) or yards, per `value_unit`. `explain_fn(t0, t1) -> str` names what
moved the value between two moments of the play.
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
    value_unit: str = "%"                          # "%" (value is 0..1) or "yd"
    team_names: tuple[str, str, str] = ("Dark jerseys", "Light jerseys", "Unassigned")
    on_field_note: str = "of 22 players + officials"
    delta_note: str = "since the snap"
    value_note: str = ""                           # why there is no value, when the caller knows
    clock_text: str = ""                           # the caller's own clock ("snap +1.2s"), shown instead of the play clock


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
        value_unit: str = "%",
        min_swing: float | None = None,
        explain_fn: Callable[[float, float], str] | None = None,
        team_names: tuple[str, str, str] = ("Dark jerseys", "Light jerseys", "Unassigned"),
        on_field_note: str = "of 22 players + officials",
        baseline_fn: Callable[[float], float] | None = None,
        value_clock_fn: Callable[[], float | None] | None = None,
        value_note_fn: Callable[[], str] | None = None,
        team_names_fn: Callable[[], tuple[str, str, str] | None] | None = None,
        rows_always: bool = False,
        clock_text_fn: Callable[[], str] | None = None,
    ):
        self.window = window
        self.settle = settle
        self.gap_reset = gap_reset
        self.lost_after = lost_after
        self.value_fn = value_fn
        self.metrics_fn = metrics_fn
        self.swing_window = swing_window
        self.value_unit = value_unit
        self.min_swing = min_swing if min_swing is not None else (0.01 if value_unit == "%" else 0.5)
        self.explain_fn = explain_fn
        self.team_names = team_names
        self.on_field_note = on_field_note
        self.baseline_fn = baseline_fn      # typical value at a given play clock; the plus/minus is measured against it
        self.value_clock_fn = value_clock_fn  # seconds since the snap, when the caller knows it; None = not live
        self.value_note_fn = value_note_fn
        self.team_names_fn = team_names_fn
        self.clock_text_fn = clock_text_fn
        self.rows_always = rows_always        # the caller has rows to show even without yards (the play's name)
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
                changes.append((vals[j][0], Swing(t=vals[i][0], delta=vals[i][1] - vals[j][1])))
        changes.sort(key=lambda c: -abs(c[1].delta))
        picked: list[Swing] = []
        for t0, s in changes:
            if abs(s.delta) < self.min_swing:
                break
            if all(abs(s.t - p.t) > self.swing_window for p in picked):
                if self.explain_fn is not None:
                    s.label = self.explain_fn(t0, s.t)
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
        if self.rows_always and self.metrics_fn is not None and not (calibrated and self._in_play):
            rows = self.metrics_fn(result)
        if calibrated and self._in_play:
            if self.metrics_fn is not None:
                rows = self.metrics_fn(result)
            vclock = clock if self.value_clock_fn is None else self.value_clock_fn()
            if self.value_fn is not None and vclock is not None:
                v = self.value_fn(result)
                if v is not None:
                    value = float(v)
                    self._values.append((vclock, value))
                    delta = value - (self.baseline_fn(vclock) if self.baseline_fn is not None else self._values[0][1])
            elif self.value_fn is not None and self._values:
                value = self._values[-1][1]                  # the play is over: hold the last reading
                delta = value - (self.baseline_fn(self._values[-1][0]) if self.baseline_fn is not None else self._values[0][1])

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
            value_unit=self.value_unit,
            team_names=(self.team_names_fn() if self.team_names_fn is not None else None) or self.team_names,
            on_field_note=self.on_field_note,
            value_note=self.value_note_fn() if self.value_note_fn is not None else "",
            clock_text=self.clock_text_fn() if self.clock_text_fn is not None else "",
            delta_note="vs a typical play" if self.baseline_fn is not None else "since the snap",
        )


def geometry_rows(result: FrameResult) -> list[tuple[str, str]]:
    """What can be said in yards before roles are known: how the 22 are spread out."""
    pts = [(p.field_yd, p.team) for p in result.players if p.field_yd is not None]
    if len(pts) < 6:
        return []
    import numpy as np

    xy = np.array([p for p, _ in pts])
    rows = [
        ("Spread across the field", f"{np.ptp(xy[:, 1]):.0f} yd"),
        ("Depth, front to back", f"{np.ptp(xy[:, 0]):.0f} yd"),
    ]
    a = np.array([p for p, t in pts if t == 0])
    b = np.array([p for p, t in pts if t == 1])
    if len(a) >= 4 and len(b) >= 4:
        gap = abs(float(np.median(a[:, 0]) - np.median(b[:, 0])))
        rows.append(("Between the two teams", f"{gap:.1f} yd"))
    return rows
