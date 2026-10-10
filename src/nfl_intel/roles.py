"""Who is who, from how they line up at the snap.

Works in absolute coordinates: the offense moves toward +x and the line of scrimmage is at
`los`. Roles are the ones the value model knows:

  QB    the passer
  REC   eligible receivers: wide receivers, tight ends, backs
  OL    the interior linemen
  DEF   defenders in coverage
  DL    defenders rushing: on the line, inside the box
  OFF   officials and anyone else who doesn't fit

The detector loses linemen in the pile and the jersey split gets the odd player wrong, so
nothing here needs a complete line or a perfect split. Before the snap the two teams are on
their own sides of the ball, which says more about a player than the colour of his shirt, and
the ball is always spotted between the hash marks.
"""

from __future__ import annotations

import numpy as np

from .field import FIELD_WIDTH
from .fieldtrack import Slot

HASH_LOW, HASH_HIGH = 23.58, 29.75      # the ball is never spotted outside the hash marks
NEUTRAL = 1.0                           # closer to the line than this, the jersey decides the side
SIDELINE = 1.0                          # closer to a sideline than this is an official or a coach
BOX = 4.2                               # tackle to tackle is about eight yards
BACKFIELD = 9.0


def line_chain(y: np.ndarray, gap: float = 2.3) -> list[list[int]]:
    """Groups of players standing shoulder to shoulder across the field (indices into y)."""
    order = np.argsort(y)
    chains, cur = [], [int(order[0])] if len(order) else []
    for a, b in zip(order, order[1:]):
        if y[b] - y[a] <= gap:
            cur.append(int(b))
        else:
            chains.append(cur)
            cur = [int(b)]
    if cur:
        chains.append(cur)
    return chains


def sides(slots: list[Slot], xy: np.ndarray, offense_team: int, los: float) -> tuple[list[int], list[int]]:
    """Indices of the offense and of the defense, by which side of the ball they stand on.

    Each side has eleven men. Anyone beyond that is an official or a stray from the bench,
    and the ones that look least like players are the ones left out: on offense whoever is
    deepest behind the ball (the referee stands well behind the passer), on defense whoever
    is deepest or widest (the back judge, the sideline officials).
    """
    off, dfn = [], []
    for i, s in enumerate(slots):
        d, y = xy[i, 0] - los, xy[i, 1]
        if y < SIDELINE or y > FIELD_WIDTH - SIDELINE or d < -14.0 or d > 30.0:
            continue
        if d <= -NEUTRAL:
            off.append(i)
        elif d >= NEUTRAL:
            dfn.append(i)
        elif s.team is not None:
            (off if s.team == offense_team else dfn).append(i)
    if off or dfn:
        mid = float(np.median(xy[off + dfn, 1]))
        off = _eleven(off, lambda i: max(0.0, -(xy[i, 0] - los) - 8.0) + max(0.0, abs(xy[i, 1] - mid) - 24.0))
        dfn = _eleven(dfn, lambda i: max(0.0, (xy[i, 0] - los) - 16.0) + max(0.0, abs(xy[i, 1] - mid) - 24.0))
    return off, dfn


def _eleven(side: list[int], oddness) -> list[int]:
    """At most eleven, dropping the oddest first."""
    if len(side) <= 11:
        return side
    keep = sorted(side, key=oddness)[:11]
    return [i for i in side if i in keep]


HASH_SPOT = {"L": HASH_HIGH, "R": HASH_LOW, "M": (HASH_LOW + HASH_HIGH) / 2.0}     # the offense's left is +y


def ball_spot(xy: np.ndarray, off: list[int], dfn: list[int], los: float, hash_mark: str | None = None) -> float:
    """Where across the field the ball is: on the charted hash mark when the play is known,
    otherwise where both lines crowd around it."""
    if hash_mark in HASH_SPOT:
        return HASH_SPOT[hash_mark]
    crowd = [i for i in off + dfn if abs(xy[i, 0] - los) <= 2.5 and 14.0 <= xy[i, 1] <= FIELD_WIDTH - 14.0]
    if len(crowd) >= 3:
        y = float(np.median(xy[crowd, 1]))
    elif off:
        y = float(np.median(xy[off, 1]))
    else:
        y = FIELD_WIDTH / 2.0
    return float(np.clip(y, HASH_LOW, HASH_HIGH))


def lined_up(slots: list[Slot], xy_abs: np.ndarray, offense_team: int, los: float, need: int = 4) -> bool:
    """Is the offense at the line? In the huddle, and on the walk up from it, nobody is
    within three yards of the ball; in any formation most of the offense is."""
    xy = np.asarray(xy_abs, dtype=np.float64).reshape(-1, 2)
    off, dfn = sides(slots, xy, offense_team, los)
    at_line = sum(1 for i in off if xy[i, 0] - los >= -3.0)
    return at_line >= need and len(dfn) >= 4


def find_quarterback(xy: np.ndarray, off: list[int], los: float, ball: float, shotgun: bool | None = None,
                     sure: bool = False) -> int | None:
    """The passer: in the backfield, straight behind the ball.

    In the shotgun or pistol he is three to six yards deep with nobody between him and the
    centre. Under centre he is right behind a lineman, and any back stands behind him.
    `shotgun` is what the play-by-play says, when the play is known; it settles which to look
    for. `sure` means the ball's spot is known (charted), not estimated from the crowd.
    """
    lane = [i for i in off if HASH_LOW - 1.3 <= xy[i, 1] <= HASH_HIGH + 1.3 and -BACKFIELD <= xy[i, 0] - los <= -0.6]
    if shotgun is False:
        # under centre he is the man on the ball's line, a yard or two back: behind the centre if we can see both
        reach = 1.3 if sure else 1.0
        close = [i for i in lane if xy[i, 0] - los >= -2.8 and abs(xy[i, 1] - ball) <= reach]
        if close:
            return min(close, key=lambda i: (round(abs(xy[i, 1] - ball) / 0.7), xy[i, 0]))
        if sure:
            return None
    if shotgun is True and sure:
        gun = [i for i in lane if -6.5 <= xy[i, 0] - los <= -2.8 and abs(xy[i, 1] - ball) <= 1.6]
        if gun:
            return min(gun, key=lambda i: abs(xy[i, 1] - ball))

    def someone_ahead(i: int) -> bool:
        return any(j != i and 0.3 <= xy[j, 0] - xy[i, 0] <= 1.6 and abs(xy[j, 1] - xy[i, 1]) <= 0.8 for j in off)

    def back_behind(i: int) -> bool:
        return any(j != i and xy[j, 0] - xy[i, 0] <= -2.5 and abs(xy[j, 1] - xy[i, 1]) <= 1.2 for j in lane)

    under = [i for i in lane if xy[i, 0] - los >= -2.5 and someone_ahead(i) and back_behind(i)]
    if under and shotgun is not True:
        return min(under, key=lambda i: abs(xy[i, 1] - ball))
    gun = [i for i in lane if -6.2 <= xy[i, 0] - los <= -2.8]
    if gun and shotgun is not False:
        return min(gun, key=lambda i: (round(abs(xy[i, 1] - ball) / 0.75), -xy[i, 0]))
    under = [i for i in lane if xy[i, 0] - los >= -2.5 and someone_ahead(i)]
    if under and shotgun is not True:
        return min(under, key=lambda i: abs(xy[i, 1] - ball))
    return None


def assign_roles(slots: list[Slot], xy_abs: np.ndarray, offense_team: int, los: float,
                 shotgun: bool | None = None, hash_mark: str | None = None, linemen: int | None = None,
                 eligible: int | None = None) -> dict[int, str]:
    """Roles by slot id. `xy_abs` are the slots' positions, offense moving +x.

    `shotgun` and `hash_mark` ("L", "M", "R" from the offense's side) come from the play's
    charting when the play is known; `linemen` and `eligible` (how many of each are on the
    field) from the on-field list. Without them the formation alone decides."""
    xy = np.asarray(xy_abs, dtype=np.float64).reshape(-1, 2)
    roles: dict[int, str] = {s.id: "OFF" for s in slots}
    off, dfn = sides(slots, xy, offense_team, los)
    if len(off) < 4 or len(dfn) < 4:
        return roles
    ball = ball_spot(xy, off, dfn, los, hash_mark)

    qb = find_quarterback(xy, off, los, ball, shotgun, sure=hash_mark in HASH_SPOT)
    if qb is not None:
        ball = float(np.clip(xy[qb, 1], HASH_LOW, HASH_HIGH))        # he stands straight behind it

    # linemen: on the ball, inside the box, the five nearest the ball (more when an extra one is in)
    n_line = max(5, linemen or 5)
    line = [i for i in off if i != qb and xy[i, 0] - los >= -2.6 and abs(xy[i, 1] - ball) <= BOX]
    line.sort(key=lambda i: abs(xy[i, 1] - ball))
    line = line[:n_line]
    if True:
        # more receivers than the offense has on the field (five, unless the list says fewer):
        # the extras nearest the ball are linemen the box missed
        spare = [i for i in off if i != qb and i not in line]
        extra = len(spare) - (5 if eligible is None else eligible)
        if extra > 0:
            spare.sort(key=lambda i: (xy[i, 0] - los < -2.6, abs(xy[i, 1] - ball)))
            line += spare[:extra]
    for i in line:
        roles[slots[i].id] = "OL"
    if qb is not None:
        roles[slots[qb].id] = "QB"
    for i in off:
        if roles[slots[i].id] == "OFF":
            roles[slots[i].id] = "REC"

    # the defense: rushers are on the line inside the box, everyone else covers
    for i in dfn:
        on_line = xy[i, 0] - los <= 1.8
        in_box = abs(xy[i, 1] - ball) <= BOX + 2.8             # edge rushers line up outside the tackles
        roles[slots[i].id] = "DL" if (on_line and in_box) else "DEF"
    return roles


def describe_look(xy_abs: np.ndarray, roles: list[str | None], los: float, shotgun: bool | None = None,
                  hash_mark: str | None = None, eligible: int | None = None) -> tuple[str, str]:
    """The formation in a football fan's words: ("Shotgun, 2 x 2, 1 back", "Two high, 6 in the box").

    Receivers are counted left x right of the ball from the offense's side. Linemen hidden in
    the pile go uncounted, so the box is "at least". `shotgun`, from the play-by-play, names
    the quarterback's spot even when he couldn't be picked out.
    """
    xy = np.asarray(xy_abs, dtype=np.float64).reshape(-1, 2)
    idx = {r: [i for i, x in enumerate(roles) if x == r] for r in ("QB", "REC", "OL", "DEF", "DL")}
    if not idx["REC"] and not idx["DEF"]:
        return "", ""
    qb = idx["QB"][0] if idx["QB"] else None
    near = [i for r in ("OL", "DL") for i in idx[r]]
    if hash_mark in HASH_SPOT:
        ball = HASH_SPOT[hash_mark]
    else:
        ball = float(xy[qb, 1]) if qb is not None else (float(np.median(xy[near, 1])) if near else FIELD_WIDTH / 2.0)
        ball = float(np.clip(ball, HASH_LOW, HASH_HIGH))

    backs = [i for i in idx["REC"] if xy[i, 0] - los <= -3.0 and abs(xy[i, 1] - ball) <= 5.0]
    left = [i for i in idx["REC"] if i not in backs and xy[i, 1] > ball]
    right = [i for i in idx["REC"] if i not in backs and xy[i, 1] <= ball]
    offense = []
    if shotgun is not None:
        offense.append("Shotgun" if shotgun else "Under centre")
    elif qb is not None:
        offense.append("Shotgun" if xy[qb, 0] - los <= -2.8 else "Under centre")
    there = eligible or 5                         # five, unless an extra lineman is in for one
    if len(idx["REC"]) < there - 1:               # with this few found, a count would mislead
        return ", ".join(offense + [f"{len(idx['REC'])} of {there} receivers found"]), _defense_look(xy, idx, los, ball)
    offense.append(f"{len(left)} x {len(right)}")
    if backs:
        offense.append(f"{len(backs)} back" + ("s" if len(backs) > 1 else ""))
    elif qb is not None:
        offense.append("empty")
    return ", ".join(offense), _defense_look(xy, idx, los, ball)


def _defense_look(xy: np.ndarray, idx: dict, los: float, ball: float) -> str:

    deep = [i for i in idx["DEF"] if xy[i, 0] - los >= 9.0 and abs(xy[i, 1] - ball) <= 14.0]     # a corner playing off isn't a safety
    # the box: the men on the line out to the edge rushers, and the linebackers stacked behind them
    box = [i for r in ("DEF", "DL") for i in idx[r]
           if (xy[i, 0] - los <= 2.0 and abs(xy[i, 1] - ball) <= BOX + 3.3) or (xy[i, 0] - los <= 6.0 and abs(xy[i, 1] - ball) <= BOX + 0.3)]
    high = {0: "No deep safety", 1: "Single high", 2: "Two high"}.get(len(deep), f"{len(deep)} deep")
    return f"{high}, {len(box)}+ in the box"


def update_roles(roles: dict[int, str], xy_abs: dict[int, np.ndarray], los: float, since_snap: float, passing: bool = True,
                 ball_y: float | None = None, eligible: int = 5) -> list[int]:
    """Correct roles from what players do after the snap. Returns the ids that changed.

    Where a man stands says what he probably is; where he goes says what he is. A defender
    who comes through the line near the ball in the first two seconds of a pass is rushing;
    later, or out wide, or on a run, he is chasing the ball like everyone else. A man on the
    defensive line who backs off is covering. On a pass, a "lineman" who runs downfield was a
    tight end, as long as the offense still has a receiver unaccounted for.
    """
    changed = []
    receivers = sum(1 for r in roles.values() if r == "REC")
    for sid, role in list(roles.items()):
        if sid not in xy_abs:
            continue
        d = float(xy_abs[sid][0]) - los
        near_ball = ball_y is None or abs(float(xy_abs[sid][1]) - ball_y) <= 9.0
        new = role
        if role == "DEF" and passing and since_snap <= 2.0 and d < -0.7 and near_ball:
            new = "DL"
        elif role == "DL" and since_snap >= 0.6 and d >= 2.5:
            new = "DEF"
        elif role == "OL" and passing and 0.4 <= since_snap <= 2.5 and d >= 1.5 and receivers < eligible:
            new = "REC"
            receivers += 1
        if new != role:
            roles[sid] = new
            changed.append(sid)
    return changed
