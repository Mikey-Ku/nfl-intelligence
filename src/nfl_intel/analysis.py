"""From tracked film to football: one object that follows a play from the first frame.

For each camera shot it
  1. reads the play off NFL Pro's header and finds it in the play-by-play,
  2. tracks the players on the field in yards,
  3. finds where the two teams face each other and which end zone is being attacked, which
     pins the field to real yard lines with the offense moving toward +x,
  4. finds the snap, and at the snap names the roles (QB, receivers, linemen, coverage),
  5. from the snap on, scores every receiver with the pass value model.

It only speaks when it is sure enough: no mapping, no anchor or no snap means no value.

`AnalyzedTracker` puts all of that behind PlayTracker's interface, so the overlay, `nfl run`
and the panel get real yard lines, roles and the value without knowing how.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .anchor import Anchor, anchor_play, find_end_zone
from .fieldtrack import FieldTracker, SnapDetector
from .playinfo import PlayRecord, match_play, read_header
from .roles import assign_roles, describe_look, lined_up, update_roles
from .schema import FrameResult, TrackedPlayer
from .stats import LiveStats
from .value import PassValueModel, ReceiverScore, play_value, receiver_features

LIVE_FOR = 4.5                # seconds after the snap that "best throw available" still means something
FORMATION_AT = 0.1            # roles are read from where players stood this long before the snap
LOOK_EVERY = 8                # mapped frames between pre-snap reads of the formation


@dataclass
class Analysis:
    record: PlayRecord | None = None
    anchor: Anchor | None = None
    snap_t: float | None = None
    since_snap: float | None = None
    live: bool = False                       # between the snap and LIVE_FOR seconds later
    mapped: bool = False                     # this frame has positions in yards
    scores: list[ReceiverScore] = field(default_factory=list)
    value: float | None = None
    best: ReceiverScore | None = None
    coverage: str = ""
    look: tuple[str, str] = ("", "")         # offense, defense, as lined up
    players: list[TrackedPlayer] = field(default_factory=list)      # in absolute yards when anchored
    slot_ids: list[int] = field(default_factory=list)               # the field tracker's id for each mapped player
    unseen: list[TrackedPlayer] = field(default_factory=list)       # players the detector lost a moment ago, carried on their last heading
    notes: list[str] = field(default_factory=list)


class AnalysisView:
    """What the panel asks of an analysis: the value, its clock, the rows, who moved it."""

    def __init__(self):
        self.last = Analysis()
        self.history: list[tuple[float, list[ReceiverScore]]] = []      # (seconds since snap, scores)
        self.has_model = True

    def clock_text(self) -> str:
        a = self.last
        if a.since_snap is not None:
            return f"snap {a.since_snap:+.1f}s"
        return "before the snap" if a.anchor is not None else ""

    def value_clock(self) -> float | None:
        return self.last.since_snap if (self.last.live and self.last.value is not None) else None

    def value(self, _result=None) -> float | None:
        return self.last.value

    def value_note(self) -> str:
        """Why there is no value on screen, when there isn't."""
        a = self.last
        if a.value is not None:
            return ""
        if not a.mapped:
            return "Field not mapped from this view.\nThe sideline angle maps; the end-zone angle doesn't yet."
        if a.record is not None and a.record.matched and not a.record.is_pass:
            return f"{a.record.play_type.capitalize()} play.\nThe value model scores passes."
        if a.anchor is None:
            return "Field mapped.\nFinding the line of scrimmage."
        if a.snap_t is None:
            return "Waiting for the offense\nto line up." if a.look[0].startswith("Not at") else "Lined up.\nWaiting for the snap."
        if not self.has_model:
            return "Needs the value model:\nrun `nfl value-train`."
        if a.since_snap is not None and a.since_snap > LIVE_FOR:
            return "Play over."
        return "Could not find the quarterback\nand his receivers."

    def team_names(self) -> tuple[str, str, str] | None:
        a = self.last
        if a.anchor is None or a.anchor.offense_team is None:
            return None
        off, dfn = "Offense", "Defense"
        if a.record is not None and a.record.header.offense:
            off = f"{a.record.header.offense} offense"
            dfn = f"{a.record.defense} defense" if a.record.defense else dfn
        names = [dfn, dfn, "Unassigned"]
        names[a.anchor.offense_team] = off
        return tuple(names)

    def explain(self, t0: float, t1: float) -> str:
        """Which receiver moved the value between two moments after the snap, and how."""
        if len(self.history) < 2:
            return ""
        times = np.array([h[0] for h in self.history])
        a = {s.index: s for s in self.history[int(np.argmin(np.abs(times - t0)))][1]}
        b = {s.index: s for s in self.history[int(np.argmin(np.abs(times - t1)))][1]}
        va = max((s.expected_yards for s in a.values()), default=0.0)
        vb = max((s.expected_yards for s in b.values()), default=0.0)
        rising = vb >= va
        pick, change = None, 0.0
        for idx, s1 in b.items():
            if idx in a:
                d = s1.expected_yards - a[idx].expected_yards
                if (rising and d > change) or (not rising and d < change):
                    pick, change = idx, d
        if pick is None:
            return ""
        s0, s1 = a[pick], b[pick]
        if rising:
            if s1.separation - s0.separation >= 0.5:
                return f"{s1.label} opens up, {s0.separation:.1f} to {s1.separation:.1f} yd of space"
            if s1.depth - s0.depth >= 1.0:
                return f"{s1.label} gets deeper, {s0.depth:.0f} to {s1.depth:.0f} yd downfield"
            return f"{s1.label} becomes the better throw"
        if s1.separation - s0.separation <= -0.5:
            return f"{s1.label} gets covered, {s0.separation:.1f} to {s1.separation:.1f} yd of space"
        return f"{s1.label} defender closing, catch chance {s0.p_catch:.0%} to {s1.p_catch:.0%}"

    def rows(self, _result=None) -> list[tuple[str, str]]:
        a = self.last
        rows: list[tuple[str, str]] = []
        if a.record is not None and a.record.header.complete:
            rows.append(("Play", a.record.summary))
        if a.anchor is None:
            return rows
        over = a.since_snap is not None and a.since_snap > LIVE_FOR
        if a.scores and not over:
            best = a.best
            most_open = max(a.scores, key=lambda s: s.separation)
            rows.append(("Best throw", f"{best.label}  {best.p_catch:.0%} caught, {best.yards_if_caught:.0f} yd"))
            rows.append(("Most open", f"{most_open.label}  {most_open.separation:.1f} yd"))
        if a.look[0]:
            rows.append(("Offense", a.look[0]))
        if a.coverage and a.since_snap is not None:
            rows.append(("Coverage read", a.coverage))
        elif a.look[1]:
            rows.append(("Defense", a.look[1]))
        if over and a.record is not None and a.record.matched:
            rows.append(("Result", a.record.outcome))
            if a.record.charted_look:
                rows.append(("Charted", a.record.charted_look))
        return rows


class PlayAnalyzer(AnalysisView):
    def __init__(self, value_model: PassValueModel | None = None, coverage_model=None, read_plays: bool = True,
                 record: PlayRecord | None = None, score_every: int = 1):
        super().__init__()
        self.value_model = value_model
        self.coverage_model = coverage_model
        self.has_model = value_model is not None
        self.read_plays = read_plays
        self.fixed_record = record
        self.score_every = max(1, score_every)
        self.tracker = FieldTracker()
        self.snap = SnapDetector()
        self._shot = None
        self._lock = None
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self.snap.reset()
        self.record: PlayRecord | None = self.fixed_record
        self.anchor: Anchor | None = None
        self.end_zone = None
        self.roles: dict[int, str] = {}
        self.labels: dict[int, str] = {}
        self.history = []
        self.at_snap: dict[int, np.ndarray] = {}
        self.snap_t: float | None = None
        self.look: tuple[str, str] = ("", "")
        self._recent: list[tuple[float, dict[int, np.ndarray]]] = []     # the last couple of seconds of positions
        self._header_tries = 0
        self._mapped_frames = 0
        self._held: tuple[list[ReceiverScore], str] = ([], "")
        self.last = Analysis(record=self.record)

    # ---------------------------------------------------------------- steps

    def _read_play(self, frame: np.ndarray) -> None:
        if not self.read_plays or self.fixed_record is not None or self._header_tries >= 3:
            return
        if self.record is not None and self.record.header.complete:
            return
        self._header_tries += 1
        try:
            self.record = match_play(read_header(frame))
        except Exception as e:                                   # OCR is a convenience, never a reason to stop
            self.record = None
            self.last.notes.append(f"could not read the play header: {e}")

    def _anchor(self, frame, registrar, field_info, boxes) -> None:
        if self.record is None or self.record.yards_to_goal is None:
            return
        slots = [s for s in self.tracker.players if s.hits >= 6 and s.team in (0, 1)]
        if len(slots) < 10:
            return
        if self.end_zone is None and registrar is not None and registrar.locked and field_info is not None:
            self.end_zone = find_end_zone(frame, registrar.H, registrar.rect, field_info.grass_lab, boxes) or False
        x = np.array([s.xy[0] for s in slots])
        team = np.array([s.team for s in slots])
        a = anchor_play(x, team, float(self.record.yards_to_goal), self.end_zone or None)
        if a is not None and (self.anchor is None or a.split_accuracy >= self.anchor.split_accuracy - 0.05):
            self.anchor = a

    def _read_look(self) -> None:
        """Before the snap: what the formation looks like right now."""
        slots = self.tracker.players
        if len(slots) < 10:
            return
        xy = self.anchor.to_absolute(np.array([s.xy for s in slots]))
        roles = assign_roles(slots, xy, self.anchor.offense_team, self.anchor.los_abs, self._shotgun, self._hash)
        look = describe_look(xy, [roles.get(s.id) for s in slots], self.anchor.los_abs, self._shotgun, self._hash)
        if look[0]:
            self.look = look

    @property
    def _shotgun(self) -> bool | None:
        return self.record.shotgun if (self.record is not None and self.record.matched) else None

    @property
    def _hash(self) -> str | None:
        return self.record.hash_mark if (self.record is not None and self.record.matched) else None

    @property
    def _passing(self) -> bool:
        return self.record is None or not self.record.matched or self.record.is_pass

    def _name_roles(self) -> None:
        # the formation as it stood just before the snap, not as it looks once everyone is moving
        want = self.snap_t - FORMATION_AT
        _, stood = min(self._recent, key=lambda r: abs(r[0] - want)) if self._recent else (None, {})
        slots = [s for s in self.tracker.players if s.id in stood]
        if len(slots) < 10:
            slots = self.tracker.players
            stood = {s.id: s.xy for s in slots}
        xy = self.anchor.to_absolute(np.array([stood[s.id] for s in slots]))
        self.roles = assign_roles(slots, xy, self.anchor.offense_team, self.anchor.los_abs, self._shotgun, self._hash)
        self.at_snap = {s.id: p for s, p in zip(slots, xy)}
        look = describe_look(xy, [self.roles.get(s.id) for s in slots], self.anchor.los_abs, self._shotgun, self._hash)
        if look[0]:
            self.look = look
        self._relabel()

    def _relabel(self) -> None:
        """Receivers and defenders are numbered across the field, from the offense's left."""
        self.labels = {}
        for prefix, role in (("R", "REC"), ("D", "DEF")):
            ids = sorted((sid for sid, r in self.roles.items() if r == role and sid in self.at_snap), key=lambda i: -self.at_snap[i][1])
            for n, sid in enumerate(ids, 1):
                self.labels[sid] = f"{prefix}{n}"
        for sid, r in self.roles.items():
            if r == "QB":
                self.labels[sid] = "QB"
        for s in self.tracker.slots:
            s.role, s.label = self.roles.get(s.id), self.labels.get(s.id)

    def _watch_roles(self, since: float) -> None:
        """What players do after the snap corrects what their alignment suggested."""
        now = {s.id: self.anchor.to_absolute(s.xy)[0] for s in self.tracker.slots if s.id in self.roles}
        changed = update_roles(self.roles, now, self.anchor.los_abs, since, self._passing)
        if any(self.roles[sid] in ("REC", "DEF") or sid in self.labels for sid in changed):
            if since <= 1.0:
                self._relabel()                          # early enough that nobody has learned the names yet
            else:
                for sid in changed:                      # later, keep everyone's name and add to the end
                    self.labels.pop(sid, None)
                    role = self.roles[sid]
                    if role in ("REC", "DEF"):
                        prefix = "R" if role == "REC" else "D"
                        taken = {v for v in self.labels.values() if v.startswith(prefix)}
                        n = 1
                        while f"{prefix}{n}" in taken:
                            n += 1
                        self.labels[sid] = f"{prefix}{n}"
                for s in self.tracker.slots:
                    s.role, s.label = self.roles.get(s.id), self.labels.get(s.id)

    def _score(self, since: float) -> tuple[list[ReceiverScore], str]:
        slots = [s for s in self.tracker.slots if s.id in self.roles]
        by_role = {r: [s for s in slots if self.roles[s.id] == r] for r in ("QB", "REC", "DEF")}
        if not by_role["QB"] or not by_role["REC"] or len(by_role["DEF"]) < 2 or self.value_model is None:
            return [], ""
        used = by_role["QB"][:1] + by_role["REC"] + by_role["DEF"]
        xy = self.anchor.to_absolute(np.array([s.xy for s in used]))
        v = np.array([s.v for s in used]) * self.anchor.direction
        n_rec = len(by_role["REC"])
        receivers = list(range(1, 1 + n_rec))
        defenders = list(range(1 + n_rec, len(used)))
        X = np.stack([receiver_features(xy, v, r, 0, defenders, self.anchor.los_abs, since) for r in receivers])
        p, y = self.value_model.score(X)
        scores = [ReceiverScore(used[r].id, float(p[i]), float(y[i]), float(p[i] * y[i]), float(X[i, 0]),
                                self.labels.get(used[r].id, "R"), float(X[i, 4])) for i, r in enumerate(receivers)]
        coverage = ""
        if self.coverage_model is not None and len(defenders) >= 5:
            start = np.array([self.at_snap.get(s.id, p_) for s, p_ in zip(used, xy)])
            read = self.coverage_model.read(np.stack([start, xy]), defenders, receivers, self.anchor.los_abs, 1, seconds=since)
            coverage = read.text if read is not None else ""
        return scores, coverage

    # ---------------------------------------------------------------- per frame

    def update(self, frame: np.ndarray, result: FrameResult, play_tracker=None) -> Analysis:
        registrar = getattr(play_tracker, "registrar", None)
        field_info = getattr(play_tracker, "field", None)
        lock = getattr(registrar, "lock_id", None)
        if result.shot != self._shot or lock != self._lock:
            self._shot, self._lock = result.shot, lock
            self.reset()
        out = Analysis(record=self.record, anchor=self.anchor, snap_t=self.snap_t, look=self.look)
        mapped = [p for p in result.players if p.field_yd is not None]
        if len(mapped) < 6:
            if self.record is None and result.players and self._header_tries == 0:
                self._read_play(frame)                           # the header is there even when the field isn't mapped
                out.record = self.record
            self.last = out
            return out
        out.mapped = True
        self._mapped_frames += 1
        if self._mapped_frames in (1, 8, 20):
            self._read_play(frame)
            out.record = self.record
        boxes = np.array([p.box for p in result.players]) if result.players else None
        xy_rel = np.array([p.field_yd for p in mapped])
        slots = self.tracker.update(result.t, xy_rel, [p.team for p in mapped], [p.track_id for p in mapped])

        if self.snap_t is None:
            self._recent.append((result.t, {s.id: s.xy.copy() for s in self.tracker.players}))
            while self._recent and result.t - self._recent[0][0] > 2.5:
                self._recent.pop(0)
            if self._mapped_frames % 6 == 3:
                self._anchor(frame, registrar, field_info, boxes)
            if self.anchor is not None:
                players = self.tracker.players
                at_line = len(players) >= 8 and lined_up(
                    players, self.anchor.to_absolute(np.array([s.xy for s in players])), self.anchor.offense_team, self.anchor.los_abs)
                if not at_line:
                    self.snap.reset()                   # breaking the huddle is a lot of movement too, but it isn't the snap
                    self.look = ("Not at the line yet", "")
                else:
                    if self._mapped_frames % LOOK_EVERY == 4 or self.look[0].startswith("Not at"):
                        self._read_look()
                    if self.snap.update(result.t, players) is not None:
                        self.snap_t = self.snap.snap_t
                        self._name_roles()
        out.anchor, out.snap_t, out.look = self.anchor, self.snap_t, self.look
        out.slot_ids = [s.id for s in slots]

        if self.anchor is not None:
            xy_abs = self.anchor.to_absolute(np.array([s.xy for s in slots]))
            for p, s, q in zip(mapped, slots, xy_abs):
                role = self.roles.get(s.id)
                team = s.team
                if role in ("QB", "REC", "OL"):                 # where he lined up outvotes the colour of his shirt
                    team = self.anchor.offense_team
                elif role in ("DEF", "DL"):
                    team = 1 - self.anchor.offense_team
                out.players.append(TrackedPlayer(
                    track_id=p.track_id, team=team, box=p.box, conf=p.conf, foot_px=p.foot_px,
                    field_yd=(float(q[0]), float(q[1])), role=role, label=self.labels.get(s.id),
                ))
            shown = {s.id for s in slots}
            for s in self.tracker.players:
                if s.id not in shown and s.id in self.roles and self.roles[s.id] != "OFF":
                    q = self.anchor.to_absolute(s.xy)[0]
                    side = self.anchor.offense_team if self.roles[s.id] in ("QB", "REC", "OL") else 1 - self.anchor.offense_team
                    out.unseen.append(TrackedPlayer(
                        track_id=-1 - s.id, team=side, box=(0.0, 0.0, 0.0, 0.0), conf=0.0, foot_px=(0.0, 0.0),
                        field_yd=(float(q[0]), float(q[1])), role=self.roles[s.id], label=self.labels.get(s.id),
                    ))
        passing = self._passing
        if self.snap_t is not None and self.anchor is not None and self.roles:
            since = result.t - self.snap_t
            out.since_snap = since
            out.live = 0.0 <= since <= LIVE_FOR
            if out.live:
                self._watch_roles(since)
            if out.live and passing:
                if self._mapped_frames % self.score_every == 0 or not self._held[0]:
                    self._held = self._score(since)
                out.scores, out.coverage = self._held
                if out.scores:
                    out.value, out.best = play_value(out.scores)
                    self.history.append((since, out.scores))
            elif since > LIVE_FOR:
                out.coverage = self._held[1]
        self.last = out
        return out


class AnalyzedTracker:
    """PlayTracker and PlayAnalyzer behind PlayTracker's interface.

    `process` returns the frame with positions on the real field (offense moving +x, line of
    scrimmage at 110 minus the yards to go to the goal line) and with roles, once the play is
    anchored. Frames from before that moment are rewritten when the shot ends, so a saved CSV
    is in one coordinate frame per shot.
    """

    def __init__(self, tracker, analyzer: PlayAnalyzer, keep: int = 4000):
        self.tracker = tracker
        self.analyzer = analyzer
        self.keep = keep
        self.analysis = Analysis()
        self._key = None
        self._frames: list[tuple[FrameResult, list[int], list[int]]] = []     # frame, which players, their slots

    @property
    def field(self):
        return self.tracker.field

    @property
    def registrar(self):
        return self.tracker.registrar

    @property
    def shot(self) -> int:
        return self.tracker.shot

    def reset(self) -> None:
        self.finish()
        self.tracker.reset()
        self.analyzer.reset()
        self._key = None

    def finish(self) -> None:
        """Put every stored frame of the shot into the final anchor's coordinates and roles."""
        a = self.analyzer
        if a.anchor is not None:
            for res, which, slot_ids in self._frames:
                if not res.anchored:
                    xy = a.anchor.to_absolute(np.array([res.players[i].field_yd for i in which]))
                    for k, i in enumerate(which):
                        res.players[i].field_yd = (float(xy[k, 0]), float(xy[k, 1]))
                    res.anchored = True
                for k, i in enumerate(which):
                    p = res.players[i]
                    p.role = a.roles.get(slot_ids[k], p.role)
                    p.label = a.labels.get(slot_ids[k], p.label)
        self._frames = []

    def process(self, frame: np.ndarray, frame_idx: int, t: float) -> FrameResult:
        res = self.tracker.process(frame, frame_idx, t)
        key = (res.shot, getattr(self.tracker.registrar, "lock_id", None))
        if key != self._key:
            self.finish()                      # the analyzer still holds the old shot's anchor here
            self._key = key
        which = [i for i, p in enumerate(res.players) if p.field_yd is not None]
        a = self.analysis = self.analyzer.update(frame, res, self.tracker)
        if a.players and len(a.players) == len(which):
            for i, p in zip(which, a.players):
                res.players[i] = p
            res.anchored = True
        if len(a.slot_ids) == len(which) and which:
            self._frames.append((res, which, list(a.slot_ids)))
            del self._frames[: -self.keep]
        return res


def live_stats(view: AnalysisView, value_model: PassValueModel | None = None, **kwargs) -> LiveStats:
    """The side panel's numbers, fed by an analysis."""
    baseline = value_model.typical if (value_model is not None and value_model.meta.get("baseline")) else None
    return LiveStats(value_fn=view.value, metrics_fn=view.rows, explain_fn=view.explain, baseline_fn=baseline,
                     value_clock_fn=view.value_clock, value_unit="yd", value_note_fn=view.value_note,
                     team_names_fn=view.team_names, clock_text_fn=view.clock_text, rows_always=True, **kwargs)


def load_models(value_path: str | None = None, coverage_path: str | None = None):
    """The trained models, or None for each one that hasn't been trained on this machine."""
    from pathlib import Path

    from .coverage import DEFAULT_MODEL as COVERAGE
    from .coverage import CoverageModel
    from .value import DEFAULT_MODEL as VALUE

    vp, cp = Path(value_path or VALUE), Path(coverage_path or COVERAGE)
    value = PassValueModel.load(vp) if vp.exists() else None
    coverage = CoverageModel.load(cp) if cp.exists() else None
    return value, coverage
