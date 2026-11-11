"""Who is on the field, read from NFL Pro's "On Field Players" list beside the video.

The list sat under that title until the autumn of 2026; now it starts straight at
"<Team> Offense" or "<Team> Defense". Either is found.

The list gives each player's number, position and name for both sides. From it come the
personnel grouping a coach would call out ("11 personnel", "nickel") and, where the
formation leaves no doubt about who is who, names to put on the players we track.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

BACKS = {"RB", "FB", "HB"}
LINEMEN = {"C", "G", "T", "OL", "OT", "OG", "LT", "LG", "RG", "RT"}
D_LINE = {"DT", "DE", "NT", "DL"}
LINEBACKERS = {"OLB", "ILB", "MLB", "LB"}
BACKFIELD = {"CB", "SS", "FS", "S", "DB"}
KNOWN = BACKS | LINEMEN | D_LINE | LINEBACKERS | BACKFIELD | {"QB", "WR", "TE", "K", "P", "LS"}
MISREAD = {"OB": "QB", "0B": "QB", "O8": "QB", "Q8": "QB", "C8": "CB", "WA": "WR", "TF": "TE", "5S": "SS", "F5": "FS"}
SUFFIXES = {"jr.", "jr", "sr.", "sr", "ii", "iii", "iv", "v"}


@dataclass
class Listed:
    number: int
    position: str            # "" when the list's single letter (C, G, T) was too small to read
    name: str

    @property
    def last_name(self) -> str:
        parts = [p for p in self.name.split() if p.lower() not in SUFFIXES]
        return parts[-1] if parts else self.name

    def is_called(self, short: str) -> bool:
        """Does the play-by-play's "M.Golden" mean this player?"""
        if not short or "." not in short:
            return False
        initial, last = short.split(".", 1)[0], short.rsplit(".", 1)[-1]
        first = self.name.split()[0] if self.name.split() else ""
        return last.lower() == self.last_name.lower() and first[:1].lower() == initial[:1].lower()


@dataclass
class OnField:
    offense_team: str = ""
    defense_team: str = ""
    offense: list[Listed] = field(default_factory=list)
    defense: list[Listed] = field(default_factory=list)

    def count(self, side: list[Listed], positions: set[str]) -> int:
        return sum(1 for p in side if p.position in positions)

    @property
    def complete(self) -> bool:
        return len(self.offense) == 11 and len(self.defense) == 11

    @property
    def personnel(self) -> str:
        """The offense's grouping: backs then tight ends, as coaches say it."""
        if len(self.offense) < 9:
            return ""
        rb, te, wr = self.count(self.offense, BACKS), self.count(self.offense, {"TE"}), self.count(self.offense, {"WR"})
        line = self.count(self.offense, LINEMEN)
        if line > 5:                                   # an extra lineman in for a receiver
            return f"Jumbo, {line} linemen: {rb} RB, {te} TE, {wr} WR"
        return f"{rb}{te} personnel: {rb} RB, {te} TE, {wr} WR"

    @property
    def package(self) -> str:
        """The defense's grouping: by how many defensive backs, then linemen-linebackers-backs."""
        if len(self.defense) < 9:
            return ""
        dl, lb, db = (self.count(self.defense, s) for s in (D_LINE, LINEBACKERS, BACKFIELD))
        name = {3: "Heavy", 4: "Base", 5: "Nickel", 6: "Dime", 7: "Quarter"}.get(db, f"{db} backs")
        return f"{name}, {dl}-{lb}-{db}"

    def eligible(self, positions: set[str]) -> list[Listed]:
        return [p for p in self.offense if p.position in positions]


LOOKALIKES = str.maketrans("АВЕКМНОРСТХ", "ABEKMHOPCTX")      # Cyrillic capitals the reader sometimes returns for Latin ones


def _position(text: str) -> str | None:
    t = text.strip().upper().translate(LOOKALIKES)
    t = MISREAD.get(t, t)
    return t if t in KNOWN else None


SIDE_HEAD = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .']{1,20}\s+(Offense|Defense)", re.IGNORECASE)


def list_head(boxes) -> tuple[float, float] | None:
    """Where the list starts: (left edge of its number column, top of its first line).
    Under the "ON FIELD PLAYERS" title when there is one, otherwise at the first
    "<Team> Offense / Defense" bar, whose text starts three line-heights right of the numbers."""
    title = next((b for b in boxes if re.search(r"ON\s*FIELD\s*PLAYERS", b.text.upper())), None)
    if title is not None:
        return title.box[0] - 25.0, title.box[3]
    bars = [b for b in boxes if SIDE_HEAD.fullmatch(b.text.strip())]
    if not bars:
        return None
    bar = min(bars, key=lambda b: b.box[1])
    return bar.box[0] - 4.5 * (bar.box[3] - bar.box[1]), bar.box[1]


def parse_on_field(boxes) -> OnField | None:
    """Build the list from OCR boxes (anything with .text and .box = x1, y1, x2, y2)."""
    head = list_head(boxes)
    if head is None:
        return None
    left, top = head
    heights = sorted(b.box[3] - b.box[1] for b in boxes)
    line = max(4.0, heights[len(heights) // 2])
    mine = sorted((b for b in boxes if b.box[0] >= left and b.box[1] >= top - 2), key=lambda b: (b.box[1] + b.box[3]) / 2)
    rows: list[list] = []
    for b in mine:
        mid = (b.box[1] + b.box[3]) / 2
        if rows and abs(mid - rows[-1][0]) <= 0.7 * line:
            rows[-1][1].append(b)
        else:
            rows.append([mid, [b]])
    out, side = OnField(), None
    for _, items in rows:
        items.sort(key=lambda b: b.box[0])
        text = " ".join(b.text.strip() for b in items)
        m = re.search(r"^(.*?)\s*\b(Offense|Defense)\b", text, re.IGNORECASE)
        if m and not re.search(r"\d", text):
            side = m.group(2).lower()
            if side == "offense":
                out.offense_team = m.group(1).strip()
            else:
                out.defense_team = m.group(1).strip()
            continue
        if side is None:
            continue
        m = re.match(r"^[#·•.*]?\s*(\d{1,2})\b\s*(.*)$", text)
        if not m:
            if out.offense and out.defense:
                break                                  # past the list: the rest of the page
            continue
        rest = m.group(2).split()
        pos = _position(rest[0]) if rest else None
        name = " ".join(rest[1:] if pos else rest)
        if not re.search(r"[A-Za-z]{2}", name):
            continue
        listed = Listed(int(m.group(1)), pos or "", name)
        target = out.offense if side == "offense" else out.defense
        if len(target) < 11:
            target.append(listed)
    if not out.offense and not out.defense:
        return None
    for p in out.offense:                              # what's left unread on offense is the line
        if not p.position:
            p.position = "OL"
    return out


def name_players(on_field: OnField, receivers: list[tuple[int, float, float]], ball: float,
                 target: str = "", pass_location: str = "") -> dict[int, str]:
    """Names for the receivers that the formation identifies beyond doubt.

    `receivers`: (id, yards behind (-) or past (+) the line, y) at the snap, for every
    eligible receiver found. Returns {id: last name}. A lone back in the backfield is the
    back on the list; a lone receiver tight to the line is the tight end; a receiver alone
    on the side the ball was thrown to is the man it was thrown to. Anyone else keeps his
    number rather than a guess.
    """
    names: dict[int, str] = {}
    if on_field is None or len(on_field.offense) < 9:
        return names
    backs = [r for r in receivers if r[1] <= -3.0 and abs(r[2] - ball) <= 5.0]
    tight = [r for r in receivers if r not in backs and r[1] >= -2.6 and 3.5 <= abs(r[2] - ball) <= 7.5]
    listed_backs, listed_te = on_field.eligible(BACKS), on_field.eligible({"TE"})
    taken: set[str] = set()
    if len(backs) == 1 and len(listed_backs) == 1:
        names[backs[0][0]] = listed_backs[0].last_name
        taken.add(listed_backs[0].name)
    if len(tight) == 1 and len(listed_te) == 1:
        names[tight[0][0]] = listed_te[0].last_name
        taken.add(listed_te[0].name)
    thrown_to = next((p for p in on_field.offense if p.is_called(target)), None)
    if thrown_to is not None and thrown_to.name not in taken and pass_location in ("left", "right"):
        wide = [r for r in receivers if r[0] not in names and r not in backs
                and ((r[2] - ball >= 4.0) if pass_location == "left" else (ball - r[2] >= 4.0))]
        if len(wide) == 1:
            names[wide[0][0]] = thrown_to.last_name
            taken.add(thrown_to.name)
    rest = [r for r in receivers if r[0] not in names]
    unnamed = [p for p in on_field.offense if p.position in BACKS | {"TE", "WR"} and p.name not in taken]
    if len(rest) == 1 and len(unnamed) == 1 and len(receivers) == 5:
        names[rest[0][0]] = unnamed[0].last_name
    return names
