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

import cv2
import numpy as np

from .anchor import Anchor, anchor_play, find_end_zone, find_huddle
from .field import FIELD_WIDTH
from .fieldnumbers import flatten, goal_line, read_numbers
from .fieldtrack import FieldTracker, SnapDetector
from .jersey import ShotNumbers
from .looks import LookBook, ShotLooks
from .personnel import BACKFIELD, BACKS, D_LINE, LINEBACKERS, LINEMEN, name_players
from .playinfo import PlayRecord, match_play, read_screen
from .roles import assign_roles, describe_look, lined_up, update_roles
from .schema import FrameResult, TrackedPlayer
from .stats import PER_SIDE, LiveStats
from .value import PassValueModel, ReceiverScore, play_value, receiver_features, what_changed

LIVE_FOR = 4.5                # seconds after the snap that "best throw available" still means something
FORMATION_AT = 0.1            # roles are read from where players stood this long before the snap
LOOK_EVERY = 8                # mapped frames between pre-snap reads of the formation
NUMBERS_EVERY = 15            # mapped frames between looks at who the jersey numbers say is who
OFFENSE_ROLES = ("QB", "REC", "OL")
DEFENSE_ROLES = ("DEF", "DL")
ROLE_POSITIONS = {"QB": {"QB"}, "REC": BACKS | {"WR", "TE"}, "OL": LINEMEN | {""},      # which roster positions a role can be
                  "DL": D_LINE | {"OLB"}, "DEF": BACKFIELD | LINEBACKERS}


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
    replayed: bool = False                   # this view isn't mapped: the read comes from the same play's sideline view
    offense_team: int | None = None          # which jersey group (0 dark, 1 light) has the ball
    rush: tuple[float, float] | None = None  # the closest rusher to the quarterback: yards away, yards a second closing
    loose: list[TrackedPlayer] = field(default_factory=list)        # seen this frame, not yet confirmed by the image tracker
    throw: object = None                     # decision.ThrowGrade once the ball has been thrown and the throw graded
    notes: list[str] = field(default_factory=list)


@dataclass
class PlayMemory:
    """What one view of a play worked out, kept for the other view of the same play."""
    snap_offset: float                       # seconds from the start of the shot to the snap
    history: list                            # (seconds since the snap, scores)
    look: tuple[str, str]
    coverage: str
    offense_team: int | None = None
    throw: object = None                     # decision.ThrowGrade, when the throw could be graded


class AnalysisView:
    """What the panel asks of an analysis: the value, its clock, the rows, who moved it."""

    def __init__(self):
        self.last = Analysis()
        self.history: list[tuple[float, list[ReceiverScore]]] = []      # (seconds since snap, scores)
        self.has_model = True

    def to_goal(self) -> float | None:
        """Yards from the line of scrimmage to the goal line being attacked."""
        a = self.last
        return None if a.anchor is None else 110.0 - a.anchor.los_abs

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
        if a.record is not None and not a.record.from_scrimmage:
            kind = {"no_play": "No play (penalty)", "qb_kneel": "Kneel-down", "qb_spike": "Spike"}.get(a.record.play_type, a.record.play_type.replace("_", " ").capitalize())
            return f"{kind}.\nOnly runs and passes are analysed."
        if not a.mapped and not a.replayed:
            return "Field not mapped from this view.\nThe sideline angle maps; the end-zone angle doesn't yet."
        if a.record is not None and a.record.matched and not a.record.is_pass:
            return f"{a.record.play_type.capitalize()} play.\nThe value model scores passes."
        if a.replayed:
            if a.since_snap is not None and a.since_snap > LIVE_FOR:
                return "Play over."
            return "Read from the sideline view.\nWaiting for the snap."
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
        has_ball = a.anchor.offense_team if a.anchor is not None else a.offense_team
        if has_ball is None:
            return None
        off, dfn = "Offense", "Defense"
        if a.record is not None and a.record.header.offense:
            off = f"{a.record.header.offense} offense"
            dfn = f"{a.record.defense} defense" if a.record.defense else dfn
        names = [dfn, dfn, "Unassigned"]
        names[has_ball] = off
        return tuple(names)

    def team_counts(self) -> tuple[int, int, int] | None:
        """Players on each side once the roles are named, in jersey-group order, then everyone
        else in the picture. A side is the eleven named at the snap, so a player the detector
        has lost for a moment still counts: there are never six men on one team."""
        a = self.last
        if a.anchor is None or a.snap_t is None:
            return None
        offense_team = a.anchor.offense_team
        named = [p.role for p in list(a.players) + list(a.loose) + list(a.unseen)]
        unnamed = [p.team for p in a.players if p.role is None]        # found again after the snap, not yet told apart
        offense = sum(r in ("QB", "REC", "OL") for r in named) + sum(t == offense_team for t in unnamed)
        defense = sum(r in ("DEF", "DL") for r in named) + sum(t == 1 - offense_team for t in unnamed)
        if not offense and not defense:
            return None
        counts = [defense, defense, sum(p.role == "OFF" for p in a.players) + sum(t is None for t in unnamed)]
        counts[offense_team] = offense
        return counts[0], counts[1], counts[2]

    def explain(self, t0: float, t1: float) -> str:
        """Which receiver moved the value between two moments after the snap, and how."""
        if len(self.history) < 2:
            return ""
        times = np.array([h[0] for h in self.history])
        a = {s.index: s for s in self.history[int(np.argmin(np.abs(times - t0)))][1]}
        b = {s.index: s for s in self.history[int(np.argmin(np.abs(times - t1)))][1]}
        va = max((s.expected_yards for s in a.values()), default=0.0)
        vb = max((s.expected_yards for s in b.values()), default=0.0)
        return what_changed(a, b, vb >= va)

    def rows(self, _result=None) -> list[tuple[str, str]]:
        a = self.last
        rows: list[tuple[str, str]] = []
        if a.record is not None and a.record.header.complete:
            rows.append(("Play", a.record.summary))
        if a.anchor is None and not a.replayed:
            return rows
        over = a.since_snap is not None and a.since_snap > LIVE_FOR
        if a.scores and not over:
            best = a.best
            most_open = max(a.scores, key=lambda s: s.separation)
            rows.append(("Best throw", f"{best.label}  {best.p_catch:.0%} caught, {best.yards_if_caught:.0f} yd"))
            rows.append(("Most open", f"{most_open.label}  {most_open.separation:.1f} yd"))
            if a.rush is not None:
                rows.append(("Nearest rusher", f"{a.rush[0]:.1f} yd" + (", closing" if a.rush[1] > 1.0 else "")))
        listed = a.record.on_field if a.record is not None else None
        before = a.since_snap is None or a.since_snap < 0
        if a.replayed:
            rows.append(("Read from", "the sideline view"))
        if a.look[0]:
            rows.append(("Offense", a.look[0]))
        if listed is not None and listed.personnel and (before or over):
            rows.append(("Personnel", listed.personnel.split(":")[0] + " v " + listed.package.split(",")[0].lower()))
        if a.coverage and a.since_snap is not None and a.since_snap >= 0:
            rows.append(("Coverage read", a.coverage))
        elif a.look[1]:
            rows.append(("Defense", a.look[1]))
        if a.throw is not None:
            t = a.throw
            rank = f"option {t.rank}" if t.firm else f"option {t.ranks[0]}-{t.ranks[1]}"
            rows.append(("Threw to", f"{t.target}, {rank} of {len(t.options)}"))
        if over and a.record is not None and a.record.matched:
            rows.append(("Result", a.record.outcome))
            if a.record.charted_look:
                rows.append(("Charted", a.record.charted_look))
        return rows


class PlayAnalyzer(AnalysisView):
    def __init__(self, value_model: PassValueModel | None = None, coverage_model=None, read_plays: bool = True,
                 record: PlayRecord | None = None, score_every: int = 1, use_hints: bool = True, number_reader=None,
                 look_embedder=None):
        super().__init__()
        self.number_reader = number_reader  # jersey.NumberReader: puts roster names on players by their numbers
        self.look_embedder = look_embedder  # looks.LookEmbedder: knows a named player again by his look
        self.book = LookBook()              # what each player has looked like, over the whole game
        self.use_hints = use_hints          # False: roles from the formation alone, to mark them against the charting
        self.value_model = value_model
        self.coverage_model = coverage_model
        self.has_model = value_model is not None
        self.read_plays = read_plays
        self.fixed_record = record
        self.score_every = max(1, score_every)
        self.tracker = FieldTracker()
        self.snap = SnapDetector()
        self.memory: dict[tuple, PlayMemory] = {}       # plays already worked out, by their header
        self._ways: dict[tuple, int] = {}               # (quarter, team with the ball) -> its direction, where the paint settled it
        self.plays: list[dict] = []                     # one summary per play worked out, in order
        self._shot = None
        self._lock = None
        self._shot_t0: float | None = None
        self.reset()

    def _remember(self) -> None:
        """Keep what this shot worked out, for the other camera angle of the same play."""
        rec = getattr(self, "record", None)
        self._learn_looks(rec)
        if rec is None or not rec.header.complete or getattr(self, "snap_t", None) is None or self._shot_t0 is None:
            return
        if not rec.from_scrimmage:
            return
        self.memory[rec.header.key] = PlayMemory(self.snap_t - self._shot_t0, list(self.history), self.look, self._held[1],
                                                 self.anchor.offense_team if self.anchor is not None else None, self.throw)
        self.plays.append(self.summary())
        for key in list(self.memory)[:-8]:
            del self.memory[key]

    def _learn_looks(self, rec) -> None:
        """At the end of a shot, every player whose number was read leaves his look in the book."""
        looks = getattr(self, "_looks", None)
        if looks is None or rec is None or not rec.header.complete:
            return
        for sid, listed in getattr(self, "by_number", {}).items():
            look = looks.look(sid)
            if look is not None:
                self.book.add(self._team_of(sid, rec), listed.number, look)

    def _team_of(self, sid: int, rec) -> str:
        return rec.header.offense if self.roles.get(sid) in OFFENSE_ROLES else rec.defense

    def summary(self) -> dict:
        """The play as worked out from this shot, in plain values: for the play log."""
        rec, a = self.record, self.anchor
        out: dict = {"shot": self._shot, "snap": None if self.snap_t is None else round(float(self.snap_t), 2),
                     "offense_look": self.look[0], "defense_look": self.look[1]}
        if rec is not None and rec.header.complete:
            h = rec.header
            out.update({"play": rec.summary, "quarter": h.quarter, "clock": h.clock, "down": h.down, "distance": h.distance,
                        "offense": h.offense, "defense": rec.defense, "matched": rec.matched})
            if rec.matched:
                out.update({"game_id": rec.game_id, "play_id": rec.play_id, "play_type": rec.play_type, "result": rec.outcome,
                            "yards": rec.yards_gained, "epa": None if rec.epa is None else round(float(rec.epa), 2),
                            "charted": rec.charted_look, "charting": dict(rec.charted), "shotgun": rec.shotgun,
                            "target": rec.receiver, "air_yards": rec.air_yards, "pass_location": rec.pass_location})
            if rec.on_field is not None:
                out.update({"personnel": rec.on_field.personnel, "package": rec.on_field.package})
        if a is not None:
            out.update({"yards_to_goal": round(110.0 - a.los_abs, 1), "anchored_by": a.source})
        roles = list(self.roles.values())
        out["found"] = {r: roles.count(r) for r in ("QB", "REC", "OL", "DL", "DEF") if roles.count(r)}
        out["hints_used"] = self.use_hints
        out["names"] = sorted(v for v in self.labels.values() if v != "QB" and not (v[0] in "RD" and (v[1:].isdigit() or v[1:] == "?")))
        out["numbers"] = {p.last_name: p.number for p in self.by_number.values()}
        out["looks"] = {p.last_name: p.number for p in self.by_look.values()}
        if self.history:
            to_goal = None if a is None else 110.0 - a.los_abs
            curve = []
            for since, scores in self.history:
                value, best = play_value(scores)
                typical = self.value_model.typical(since, to_goal=to_goal) if self.value_model is not None else None
                curve.append({"t": round(float(since), 2), "value": round(float(value), 2), "best": best.label,
                              "catch": round(float(best.p_catch), 2), "typical": None if typical is None else round(float(typical), 2)})
            out["value"] = curve
            peak = max(curve, key=lambda c: c["value"])
            out["peak"] = {"t": peak["t"], "value": peak["value"], "best": peak["best"]}
            out["coverage_read"] = self._held[1]
        if self.rushers is not None:
            out["rushers"] = self.rushers
        if self.throw is not None:
            out["throw"] = self.throw.as_dict()
        if self.closest_rush is not None:
            out["closest_rush"] = {"yards": round(self.closest_rush[0], 1), "t": round(float(self.closest_rush[1]), 2)}
        return out

    def reset(self) -> None:
        self._remember()
        self.tracker.reset()
        self.snap.reset()
        self.record: PlayRecord | None = self.fixed_record
        self.anchor: Anchor | None = None
        self.end_zone = None
        self.numbers = None                 # (goal line, side) read off the painted yard numbers
        self.huddle = None                  # (team, x) of the offense's huddle, if the shot showed one
        self.closest_rush = None            # (yards, seconds after the snap) when a rusher was nearest the quarterback
        self.rushers: int | None = None     # how many came after the passer, counted a second and a half in
        self.throw = None                   # decision.ThrowGrade for this play
        self._known: dict[int, tuple] = {}   # slot id -> (time seen, position, velocity) of players with a role
        self._number_tries = 0
        self.roles: dict[int, str] = {}
        self.lineup: dict[int, str] = {}     # the roles as they stand before the snap: only which side a man is on is used
        self.labels: dict[int, str] = {}
        self._numbers = ShotNumbers(self.number_reader) if self.number_reader is not None else None
        self._looks = ShotLooks(self.look_embedder) if self.look_embedder is not None else None
        self.by_number: dict[int, object] = {}       # slot id -> personnel.Listed, from the jersey number
        self.by_look: dict[int, object] = {}         # slot id -> personnel.Listed, from his look on earlier plays
        self.history = []
        self.at_snap: dict[int, np.ndarray] = {}
        self.snap_t: float | None = None
        self.look: tuple[str, str] = ("", "")
        self._recent: list[tuple[float, dict[int, np.ndarray]]] = []     # the last couple of seconds of positions
        self._header_tries = 0
        self._mapped_frames = 0
        self._frames = 0
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
            header, on_field = read_screen(frame)
            self.record = match_play(header)
            self.record.on_field = on_field
        except Exception as e:                                   # OCR is a convenience, never a reason to stop
            self.record = None
            self.last.notes.append(f"could not read the play header: {e}")

    def _anchor(self, frame, registrar, field_info, boxes) -> None:
        slots = [s for s in self.tracker.players if s.hits >= 6 and s.team in (0, 1)]
        if len(slots) < 10:
            return
        locked = registrar is not None and registrar.locked
        if self.numbers is None and locked and self._number_tries < 4:
            self._number_tries += 1
            try:
                flat = flatten(frame, registrar.H, registrar.rect, boxes)
                self.numbers = goal_line(read_numbers(flat)) if flat is not None else None
            except Exception as e:                               # reading paint is a help, never a reason to stop
                self.last.notes.append(f"could not read the painted numbers: {e}")
        if self.end_zone is None and self.numbers is None and locked and field_info is not None:
            self.end_zone = find_end_zone(frame, registrar.H, registrar.rect, field_info.grass_lab, boxes) or False
        goal, source = (self.numbers, "painted numbers") if self.numbers else (self.end_zone or None, "end zone")
        to_goal = self.record.yards_to_goal if self.record is not None else None
        x = np.array([s.xy[0] for s in slots])
        team = np.array([s.team for s in slots])
        seen = find_huddle(np.array([s.xy for s in slots]), team)
        if seen is not None:
            self.huddle = seen
        h = self.record.header if self.record is not None else None
        drive = (h.quarter, h.offense) if h is not None and h.quarter is not None and h.offense else None
        a = anchor_play(x, team, None if to_goal is None else float(to_goal), goal, source, self.huddle,
                        self._ways.get(drive) if drive is not None else None)
        if to_goal is None:
            # with only the formation to say which way the offense is going, the latest look is
            # the one that counts: a guess made while the offense walked up must not outlive it
            self.anchor = a
        elif a is not None and (self.anchor is None or a.split_accuracy >= self.anchor.split_accuracy - 0.05):
            self.anchor = a
        if drive is not None and self.record.matched and self.anchor is not None and self.anchor.source in ("painted numbers", "end zone"):
            self._ways[drive] = self.anchor.direction            # the paint has said which way this team is going this quarter
            other = self.record.defense or next((t for t in (h.game or ()) if t != h.offense), "")
            if other:
                self._ways[(h.quarter, other)] = -self.anchor.direction

    def _read_look(self) -> None:
        """Before the snap: what the formation looks like right now."""
        slots = self.tracker.players
        if len(slots) < 10:
            return
        xy = self.anchor.to_absolute(np.array([s.xy for s in slots]))
        roles = assign_roles(slots, xy, self.anchor.offense_team, self.anchor.los_abs, self._shotgun, self._hash, *self._listed)
        self.lineup = roles
        look = describe_look(xy, [roles.get(s.id) for s in slots], self.anchor.los_abs, self._shotgun, self._hash, self._listed[1])
        if look[0]:
            self.look = look

    @property
    def _shotgun(self) -> bool | None:
        return self.record.shotgun if (self.use_hints and self.record is not None and self.record.matched) else None

    @property
    def _hash(self) -> str | None:
        return self.record.hash_mark if (self.use_hints and self.record is not None and self.record.matched) else None

    @property
    def _listed(self) -> tuple[int | None, int | None]:
        """How many linemen and how many eligible receivers the on-field list shows."""
        from .personnel import BACKS, LINEMEN

        listed = self.record.on_field if (self.use_hints and self.record is not None) else None
        if listed is None or len(listed.offense) != 11:
            return None, None
        return listed.count(listed.offense, LINEMEN), listed.count(listed.offense, BACKS | {"TE", "WR"})

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
        self.roles = assign_roles(slots, xy, self.anchor.offense_team, self.anchor.los_abs, self._shotgun, self._hash, *self._listed)
        self.at_snap = {s.id: p for s, p in zip(slots, xy)}
        look = describe_look(xy, [self.roles.get(s.id) for s in slots], self.anchor.los_abs, self._shotgun, self._hash, self._listed[1])
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
        self._name_receivers()
        self._apply_numbers()
        for s in self.tracker.slots:
            s.role, s.label = self.roles.get(s.id), self.labels.get(s.id)

    def _name_by_number(self) -> None:
        """Roster names for the players whose jersey number has been read clearly this shot.
        A number read off the shirt beats a name the formation guessed."""
        sides = {sid: "offense" if r in OFFENSE_ROLES else "defense"
                 for sid, r in self.roles.items() if r in OFFENSE_ROLES + DEFENSE_ROLES}
        self.by_number = self._numbers.names(sides, self.record.on_field)
        self._name_by_look(sides)
        self._apply_numbers()

    def _name_by_look(self, sides: dict) -> None:
        """Players without a number read, named from how they looked when their number was
        read on an earlier play. Only among the men on the field, on his side, in his kind of
        role, and only when one is clearly the best match."""
        self.by_look = {}
        if self._looks is None or not self.book.looks:
            return
        rec, on_field = self.record, self.record.on_field
        taken = {(sides[sid], listed.number) for sid, listed in self.by_number.items() if sid in sides}
        colour = {s.id: s.team for s in self.tracker.slots}
        for sid, side in sides.items():
            if sid in self.by_number:
                continue
            # a look is only compared with his side's roster, so his shirt had better agree with his role
            if self.anchor is None:
                return
            expected = self.anchor.offense_team if side == "offense" else 1 - self.anchor.offense_team
            if colour.get(sid) not in (None, expected):
                continue
            looks = self._looks.looks(sid)
            if looks is None:
                continue
            roster = on_field.offense if side == "offense" else on_field.defense
            fits = {p.number: p for p in roster if p.position in ROLE_POSITIONS[self.roles[sid]]}
            number = self.book.recognise_many(self._team_of(sid, rec), fits, looks)
            if number is not None and (side, number) not in taken:
                self.by_look[sid] = fits[number]
                taken.add((side, number))

    def _apply_numbers(self) -> None:
        """The names the numbers gave go on their players, and off anyone the formation had
        guessed them for: one man, one name."""
        named = {sid: listed.last_name for sid, listed in self.by_look.items() if sid in self.roles}
        named.update({sid: listed.last_name for sid, listed in self.by_number.items() if sid in self.roles})
        for sid, name in list(self.labels.items()):
            if sid not in named and name in named.values():
                role = self.roles.get(sid)
                self.labels[sid] = "R?" if role == "REC" else "D?" if role == "DEF" else "QB" if role == "QB" else ""
                if not self.labels[sid]:
                    del self.labels[sid]
        self.labels.update(named)
        for s in self.tracker.slots:
            s.label = self.labels.get(s.id)

    def _name_receivers(self) -> None:
        """Put names on the receivers the formation identifies beyond doubt."""
        rec = self.record
        if rec is None or rec.on_field is None:
            return
        qb = next((sid for sid, r in self.roles.items() if r == "QB" and sid in self.at_snap), None)
        if qb is None:
            return
        ball = float(np.clip(self.at_snap[qb][1], 23.58, 29.75))
        found = [(sid, float(self.at_snap[sid][0] - self.anchor.los_abs), float(self.at_snap[sid][1]))
                 for sid, r in self.roles.items() if r == "REC" and sid in self.at_snap]
        for sid, name in name_players(rec.on_field, found, ball, rec.receiver if rec.matched else "", rec.pass_location if rec.matched else "").items():
            self.labels[sid] = name

    def _eleven_a_side(self, out) -> None:
        """After the snap a man nobody has named is a player only while his side is short of
        eleven; the men well inside the field are taken first. Anyone left over is the official
        on the sideline or a second box on a player already counted, and gets no box."""
        room = {0: PER_SIDE, 1: PER_SIDE}
        for p in out.players + out.loose + out.unseen:
            if p.role not in (None, "OFF") and p.team in room:
                room[p.team] -= 1
        strangers = [p for p in out.players if p.role is None]
        for p in sorted(strangers, key=lambda p: -min(p.field_yd[1], FIELD_WIDTH - p.field_yd[1])):
            if p.team in room and room[p.team] > 0:
                room[p.team] -= 1
            else:
                p.role, p.team = "OFF", None

    def _reclaim(self, t: float) -> None:
        """Players the tracker lost and found again come back as strangers. Give each stranger
        the name of the missing player he most likely is; a stranger on the defense's side
        with nobody missing is one more defender. Without this a corner who drops out for a
        second leaves his receiver looking wide open."""
        live = {s.id: s for s in self.tracker.slots}
        for sid in self.roles:
            s = live.get(sid)
            if s is not None and t - s.seen <= 0.3:
                self._known[sid] = (s.seen, s.xy.copy(), s.v.copy())
        missing = {sid: k for sid, k in self._known.items() if sid in self.roles and self.roles[sid] != "OFF"
                   and (sid not in live or t - live[sid].seen > 0.3) and t - k[0] <= 4.0}
        for s in self.tracker.players:
            if s.id in self.roles or t - s.seen > 0.1:
                continue
            best, near = None, None
            for sid, (seen, xy, v) in missing.items():
                dt = t - seen
                reach = 2.5 + 1.5 * dt
                d = float(np.linalg.norm(xy + v * min(dt, 1.0) - s.xy))
                if self.roles[sid] in ("QB", "REC", "OL") and s.team not in (None, self.anchor.offense_team):
                    d += 1.5
                if self.roles[sid] in ("DEF", "DL") and s.team == self.anchor.offense_team:
                    d += 1.5
                if d <= min(reach, 6.0) and (near is None or d < near):
                    best, near = sid, d
            if best is not None:
                self.roles[s.id] = self.roles.pop(best)
                if best in self.labels:
                    self.labels[s.id] = self.labels.pop(best)
                for named in (self.by_number, self.by_look):
                    if best in named:
                        named[s.id] = named.pop(best)
                if self._numbers is not None and best in self._numbers.votes.score:
                    self._numbers.votes.score[s.id] = self._numbers.votes.score.pop(best)
                if self._looks is not None:
                    self._looks.move(best, s.id)
                if best in self.at_snap:
                    self.at_snap[s.id] = self.at_snap[best]
                self._known.pop(best, None)
                missing.pop(best, None)
                if best in live:
                    live[best].seen = -1e9                      # the old slot is the same man: let it go
            elif s.team is not None and s.team != self.anchor.offense_team \
                    and sum(1 for r in self.roles.values() if r in ("DEF", "DL")) < 11 \
                    and self.anchor.to_absolute(s.xy)[0, 0] > self.anchor.los_abs - 1.0:
                self.roles[s.id] = "DEF"
                taken = {v for v in self.labels.values() if v.startswith("D") and v[1:].isdigit()}
                n = 1
                while f"D{n}" in taken:
                    n += 1
                self.labels[s.id] = f"D{n}"
                self.at_snap[s.id] = self.anchor.to_absolute(s.xy)[0]
            s.role, s.label = self.roles.get(s.id), self.labels.get(s.id)

    def _watch_roles(self, since: float) -> None:
        """What players do after the snap corrects what their alignment suggested."""
        now = {s.id: self.anchor.to_absolute(s.xy)[0] for s in self.tracker.slots if s.id in self.roles}
        qb = next((sid for sid, r in self.roles.items() if r == "QB" and sid in self.at_snap), None)
        changed = update_roles(self.roles, now, self.anchor.los_abs, since, self._passing,
                               None if qb is None else float(self.at_snap[qb][1]), self._listed[1] or 5)
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
                self._apply_numbers()
                for s in self.tracker.slots:
                    s.role, s.label = self.roles.get(s.id), self.labels.get(s.id)

    def _count_rushers(self) -> int:
        """The pass rush as the charting counts it: defenders who came for the passer, a second
        and a half in, near enough to the ball to be rushing rather than running down a route."""
        qb = next((sid for sid, r in self.roles.items() if r == "QB" and sid in self.at_snap), None)
        ball = float(self.at_snap[qb][1]) if qb is not None else 26.7
        n = 0
        for s in self.tracker.slots:
            if self.roles.get(s.id) != "DL":
                continue
            x, y = self.anchor.to_absolute(s.xy)[0]
            if x - self.anchor.los_abs <= 1.0 and abs(y - ball) <= 9.0:
                n += 1
        return n

    def _rush(self) -> tuple[float, float] | None:
        """How close the nearest rusher is to the quarterback and how fast he is closing.
        The value model can't see the rush (its training data has no linemen), so the panel
        shows it next to the value instead of inside it."""
        qb = next((s for s in self.tracker.slots if self.roles.get(s.id) == "QB"), None)
        rushers = [s for s in self.tracker.slots if self.roles.get(s.id) == "DL"]
        if qb is None or not rushers:
            return None
        gaps = [(float(np.linalg.norm(r.xy - qb.xy)), r) for r in rushers]
        gap, r = min(gaps, key=lambda g: g[0])
        toward = (qb.xy - r.xy) / max(gap, 1e-6)
        return gap, float(np.dot(r.v - qb.v, toward))

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

    def _replay(self, out: Analysis, t: float) -> None:
        """A view that can't be mapped (the end-zone angle) shows the same play as the sideline
        view did, from the same moment: replay that read in step with it."""
        rec = self.record
        mem = self.memory.get(rec.header.key) if (rec is not None and rec.header.complete) else None
        if mem is None or self._shot_t0 is None or self.snap_t is not None:
            return
        since = (t - self._shot_t0) - mem.snap_offset
        out.replayed, out.look, out.offense_team = True, mem.look, mem.offense_team
        if mem.throw is not None and (t - self._shot_t0) - mem.snap_offset >= mem.throw.thrown_at:
            out.throw = mem.throw
        out.snap_t, out.since_snap = self._shot_t0 + mem.snap_offset, since
        out.live = 0.0 <= since <= LIVE_FOR
        self.history = mem.history
        if out.live and mem.history:
            times = np.array([h[0] for h in mem.history])
            k = int(np.argmin(np.abs(times - since)))
            if abs(times[k] - since) <= 0.5:
                out.scores = mem.history[k][1]
                out.value, out.best = play_value(out.scores)
        if since >= 0:
            out.coverage = mem.coverage

    # ---------------------------------------------------------------- per frame

    def update(self, frame: np.ndarray, result: FrameResult, play_tracker=None) -> Analysis:
        registrar = getattr(play_tracker, "registrar", None)
        field_info = getattr(play_tracker, "field", None)
        lock = getattr(registrar, "lock_id", None)
        if result.shot != self._shot or lock != self._lock:
            new_shot = result.shot != self._shot
            self.reset()                                 # closes the old shot's play under the old shot's number
            self._shot, self._lock = result.shot, lock
            if new_shot or self._shot_t0 is None:
                self._shot_t0 = result.t
        out = Analysis(record=self.record, anchor=self.anchor, snap_t=self.snap_t, look=self.look)
        mapped = [p for p in result.players if p.field_yd is not None]
        if len(mapped) < 6:
            self._frames += 1
            if result.players and self._frames in (1, 8, 20):
                self._read_play(frame)                           # the header is there even when the field isn't mapped
                out.record = self.record
            self._replay(out, result.t)
            self.last = out
            return out
        out.mapped = True
        self._mapped_frames += 1
        if self._mapped_frames in (1, 8, 20):
            self._read_play(frame)
            out.record = self.record
        if self.record is not None and not self.record.from_scrimmage:
            self.last = out                                      # a kick, a kneel-down, a flag before the snap
            return out
        boxes = np.array([p.box for p in result.players]) if result.players else None
        loose = [p for p in getattr(result, "loose", []) if p.field_yd is not None]
        both = mapped + loose
        xy_rel = np.array([p.field_yd for p in both])
        slots = self.tracker.update(result.t, xy_rel, [p.team for p in both], [p.track_id if p.track_id >= 0 else None for p in both],
                                    confirmed=[True] * len(mapped) + [False] * len(loose))
        loose_slots, slots = slots[len(mapped):], slots[: len(mapped)]
        if self._numbers is not None:
            self._numbers.update(frame, [p.box for p in mapped], [s.id for s in slots])
        if self._looks is not None:
            self._looks.update(frame, [p.box for p in mapped], [s.id for s in slots])

        at_line = False
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
        if self._numbers is not None and self.roles and self.record is not None and self.record.on_field is not None \
                and self._mapped_frames % NUMBERS_EVERY == 0:
            self._name_by_number()
        out.anchor, out.snap_t, out.look = self.anchor, self.snap_t, self.look
        out.slot_ids = [s.id for s in slots]

        if self.anchor is not None:
            xy_abs = self.anchor.to_absolute(np.array([s.xy for s in slots]))
            for p, s, q in zip(mapped, slots, xy_abs):
                role = self.roles.get(s.id)
                stands = role or (self.lineup.get(s.id) if at_line else None)    # before the snap, once they are lined up
                team = s.team
                if stands in ("QB", "REC", "OL"):               # where he lined up outvotes the colour of his shirt
                    team = self.anchor.offense_team
                elif stands in ("DEF", "DL"):
                    team = 1 - self.anchor.offense_team
                elif stands == "OFF":                           # an official or a stray: on neither team
                    team = None
                out.players.append(TrackedPlayer(
                    track_id=p.track_id, team=team, box=p.box, conf=p.conf, foot_px=p.foot_px,
                    field_yd=(float(q[0]), float(q[1])), role=role, label=self.labels.get(s.id), slot=s.id,
                ))
            for p, s in zip(loose, loose_slots):
                if s is not None and s.hits >= self.tracker.min_hits and self.roles.get(s.id) not in (None, "OFF"):
                    q = self.anchor.to_absolute(s.xy)[0]
                    side = self.anchor.offense_team if self.roles[s.id] in ("QB", "REC", "OL") else 1 - self.anchor.offense_team
                    out.loose.append(TrackedPlayer(track_id=-1, team=side, box=p.box, conf=p.conf, foot_px=p.foot_px,
                                                   field_yd=(float(q[0]), float(q[1])), role=self.roles[s.id], label=self.labels.get(s.id)))
            shown = {s.id for s in slots} | {s.id for s in loose_slots if s is not None}
            for s in self.tracker.players:
                if s.id not in shown and s.id in self.roles and self.roles[s.id] != "OFF":
                    q = self.anchor.to_absolute(s.xy)[0]
                    side = self.anchor.offense_team if self.roles[s.id] in ("QB", "REC", "OL") else 1 - self.anchor.offense_team
                    out.unseen.append(TrackedPlayer(
                        track_id=-1 - s.id, team=side, box=(0.0, 0.0, 0.0, 0.0), conf=0.0, foot_px=(0.0, 0.0),
                        field_yd=(float(q[0]), float(q[1])), role=self.roles[s.id], label=self.labels.get(s.id),
                    ))
            if self.roles:
                self._eleven_a_side(out)
        passing = self._passing
        if self.snap_t is not None and self.anchor is not None and self.roles:
            since = result.t - self.snap_t
            out.since_snap = since
            out.live = 0.0 <= since <= LIVE_FOR
            if out.live:
                self._reclaim(result.t)
                self._watch_roles(since)
                out.rush = self._rush()
                if self.rushers is None and since >= 1.5 and self._passing:
                    self.rushers = self._count_rushers()
                if out.rush is not None and (self.closest_rush is None or out.rush[0] < self.closest_rush[0]):
                    self.closest_rush = (out.rush[0], since)
            if out.live and passing:
                if self._mapped_frames % self.score_every == 0 or not self._held[0]:
                    self._held = self._score(since)
                out.scores, out.coverage = self._held
                if out.scores:
                    out.value, out.best = play_value(out.scores)
                    self.history.append((since, out.scores))
            elif since > LIVE_FOR:
                out.coverage = self._held[1]
            out.throw = self.throw
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
        self._graded = False

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

    GRADE_AFTER = 6.0                 # seconds after the snap by which nearly every pass has arrived

    def _labelled(self) -> list[FrameResult]:
        """The shot's frames with today's names and roles on every player, on the real field."""
        a, out = self.analyzer, []
        for res, which, slot_ids in self._frames:
            xy = np.array([res.players[i].field_yd for i in which], dtype=np.float64).reshape(-1, 2)
            if not res.anchored:
                xy = a.anchor.to_absolute(xy)
            players = [TrackedPlayer(res.players[i].track_id, res.players[i].team, res.players[i].box, res.players[i].conf,
                                     res.players[i].foot_px, (float(xy[k, 0]), float(xy[k, 1])),
                                     a.roles.get(slot_ids[k]), a.labels.get(slot_ids[k]), slot_ids[k]) for k, i in enumerate(which)]
            out.append(FrameResult(res.frame, res.t, res.size, players, res.shot, True))
        return out

    def _grade(self) -> None:
        """Grade the throw from the shot so far. Called once the ball must have arrived, and at the end of the shot."""
        from .decision import grade_throw

        a, rec = self.analyzer, self.analyzer.record
        if a.anchor is None or a.snap_t is None or a.value_model is None or rec is None or not rec.matched or not rec.is_pass or rec.sack:
            return
        name = next((p.last_name for p in (rec.on_field.offense if rec.on_field is not None else []) if p.is_called(rec.receiver)), "")
        try:
            a.throw = grade_throw(self._labelled(), a.snap_t, a.anchor.los_abs, a.value_model, rec.air_yards, rec.pass_location,
                                  name, rec.yards_gained, rec.complete) or a.throw
        except Exception as e:                                   # a grade is a bonus; a bad play must not stop the film
            a.last.notes.append(f"could not grade the throw: {e}")

    def finish(self) -> None:
        """Put every stored frame of the shot into the final anchor's coordinates and roles."""
        a = self.analyzer
        if a.anchor is not None and self._frames:
            self._grade()
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
            self._graded = False
        which = [i for i, p in enumerate(res.players) if p.field_yd is not None]
        a = self.analysis = self.analyzer.update(frame, res, self.tracker)
        if a.players and len(a.players) == len(which):
            for i, p in zip(which, a.players):
                res.players[i] = p
            res.anchored = True
        if len(a.slot_ids) == len(which) and which:
            for k, i in enumerate(which):
                res.players[i].slot = a.slot_ids[k]
            self._frames.append((res, which, list(a.slot_ids)))
            del self._frames[: -self.keep]
        if a.since_snap is not None and a.since_snap >= self.GRADE_AFTER and self.analyzer.throw is None and not self._graded:
            self._graded = True
            self._grade()
        res.lines = self._field_lines(a)
        res.loose = list(a.loose)                    # only the ones the analysis could name: they are drawn with their names
        return res

    def _field_lines(self, a: Analysis) -> list:
        """The line of scrimmage and the line to gain, in the picture, as television draws them."""
        registrar = self.tracker.registrar
        if a.anchor is None or registrar is None or registrar.H is None or registrar.rect is None:
            return []
        marks = [("scrimmage", a.anchor.los_rel)]
        to_go = a.record.header.distance if a.record is not None else None
        if to_go is not None and a.anchor.los_abs + to_go < 110.0:
            marks.append(("to gain", a.anchor.to_relative_x(a.anchor.los_abs + to_go)))
        x1, y1, x2, y2 = registrar.rect
        inverse = np.linalg.inv(registrar.H)
        out = []
        for name, x in marks:
            across = np.array([[x, y] for y in np.linspace(0.0, FIELD_WIDTH, 25)], dtype=np.float64).reshape(-1, 1, 2)
            px = cv2.perspectiveTransform(across, inverse).reshape(-1, 2)
            keep = np.all(np.isfinite(px), axis=1) & (px[:, 0] >= x1) & (px[:, 0] <= x2) & (px[:, 1] >= y1) & (px[:, 1] <= y2)
            pts = [(float(u), float(v)) for (u, v), k in zip(px, keep) if k]
            if len(pts) >= 2:
                out.append((name, pts))
        return out


def live_stats(view: AnalysisView, value_model: PassValueModel | None = None, **kwargs) -> LiveStats:
    """The side panel's numbers, fed by an analysis."""
    baseline = None
    if value_model is not None and value_model.meta.get("baseline"):
        def baseline(clock: float) -> float:
            return value_model.typical(clock, to_goal=view.to_goal())
    return LiveStats(value_fn=view.value, metrics_fn=view.rows, explain_fn=view.explain, baseline_fn=baseline,
                     value_clock_fn=view.value_clock, value_unit="yd", value_note_fn=view.value_note,
                     team_names_fn=view.team_names, team_counts_fn=view.team_counts, clock_text_fn=view.clock_text, rows_always=True,
                     value_needs_yards=False, delta_note="vs typical from here", **kwargs)


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
