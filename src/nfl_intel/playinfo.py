"""Which play is on screen: read NFL Pro's header, find the play in the public play-by-play.

NFL Pro shows, above the video, "Q3 03:41", "3RD & 1", "ATL ON GB 7", and under it the
play's description. Those identify one row of nflverse play-by-play, which carries what the
film alone can't tell us: the real yard line, who has the ball, and what happened.

Since the autumn of 2026 the header is a strip on the video's top edge: "3RD & 1 GB 7" over
a small "Q3 03:41". It no longer says who has the ball; that comes from the on-field list
("Falcons Offense") or, failing that, from the play-by-play row itself. Both are read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

NFLVERSE = Path("data/nflverse")
ORDINAL = {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4}
DOWN_LINE = re.compile(r"(1ST|2ND|3RD|4TH)\s*&\s*(\d{1,2}|GOAL)\b")      # capitals: the play list under the video writes "1st & 10"
TEAMS = {                                                              # nickname on the on-field list -> nflverse abbreviation
    "cardinals": "ARI", "falcons": "ATL", "ravens": "BAL", "bills": "BUF", "panthers": "CAR", "bears": "CHI",
    "bengals": "CIN", "browns": "CLE", "cowboys": "DAL", "broncos": "DEN", "lions": "DET", "packers": "GB",
    "texans": "HOU", "colts": "IND", "jaguars": "JAX", "chiefs": "KC", "rams": "LA", "chargers": "LAC",
    "raiders": "LV", "dolphins": "MIA", "vikings": "MIN", "patriots": "NE", "saints": "NO", "giants": "NYG",
    "jets": "NYJ", "eagles": "PHI", "steelers": "PIT", "seahawks": "SEA", "49ers": "SF", "buccaneers": "TB",
    "titans": "TEN", "commanders": "WAS",
}


@dataclass
class PlayHeader:
    quarter: int | None = None
    clock: str | None = None                 # "03:41"
    down: int | None = None
    distance: int | None = None
    offense: str | None = None               # team abbreviation with the ball
    side: str | None = None                  # whose half the ball is in; None at midfield
    yard: int | None = None                  # yard line number, 1-50
    description: str = ""
    season: int | None = None
    week: int | None = None
    game: tuple | None = None                # (away, home) when the page is filtered to one game

    @property
    def complete(self) -> bool:
        return None not in (self.quarter, self.clock, self.offense, self.yard)

    @property
    def placed(self) -> bool:
        """When and where, without who: all the newer header gives."""
        return None not in (self.quarter, self.clock, self.yard)

    @property
    def yards_to_goal(self) -> int | None:
        """Distance from the line of scrimmage to the end zone the offense is attacking."""
        if self.yard is None or self.offense is None:
            return None
        if self.side is None or self.yard == 50:
            return 50
        return 100 - self.yard if self.side == self.offense else self.yard

    @property
    def key(self) -> tuple:
        return (self.quarter, self.clock, self.down, self.distance, self.offense, self.side, self.yard)


def parse_header(lines: list[str]) -> PlayHeader:
    """Pull the play's identity out of text lines (any order, anything else ignored)."""
    h = PlayHeader()
    wrapped = None                         # the description's second line, if it wraps
    for raw in lines:
        text = raw.strip()
        up = text.upper()
        m = re.search(r"\b(?:Q\s?([1-4])|(OT))\s+(\d{1,2}):(\d{2})\b", up)
        if m and h.quarter is None:
            h.quarter = 5 if m.group(2) else int(m.group(1))
            h.clock = f"{int(m.group(3)):02d}:{m.group(4)}"
            continue
        m = DOWN_LINE.fullmatch(text)                # as written, in capitals: the play list's "1st & 10" is another play
        if m and h.down is None:
            h.down = ORDINAL[m.group(1)]
            h.distance = None if m.group(2) == "GOAL" else int(m.group(2))
            continue
        # the newer header, one line: "2ND & 6 BAL 17" ("1ST & 10 50" at midfield); the ball's side but not whose ball
        m = re.fullmatch(DOWN_LINE.pattern + r"\s+(?:([A-Z]{2,3})\s+)?(\d{1,2})", text)
        if m and h.down is None and h.yard is None:
            h.down = ORDINAL[m.group(1)]
            h.distance = None if m.group(2) == "GOAL" else int(m.group(2))
            h.side, h.yard = m.group(3), int(m.group(4))
            continue
        m = re.fullmatch(r"([A-Z]{2,3})\s+ON\s+(?:([A-Z]{2,3})\s+)?(\d{1,2})", up)
        if m and h.offense is None:
            h.offense, h.side, h.yard = m.group(1), m.group(2), int(m.group(3))
            continue
        m = re.search(r"SEASON=(\d{4})", up)
        if m:
            h.season = int(m.group(1))
            w = re.search(r"WEEK_?(\d{1,2})", up)
            if w:
                h.week = int(w.group(1))
            continue
        # full screen hides the address bar; the page's own filter chips say the same thing
        m = re.fullmatch(r"SEASON:?\s*(\d{4})", up)
        if m and h.season is None:
            h.season = int(m.group(1))
            continue
        m = re.fullmatch(r"WEEK:?\s*(\d{1,2})", up)
        if m and h.week is None:
            h.week = int(m.group(1))
            continue
        m = re.fullmatch(r"GAME:?\s*([A-Z]{2,3})\s+(?:AT|@|VS\.?)\s+([A-Z]{2,3})", up)
        if m and not h.game:
            h.game = (m.group(1), m.group(2))
            continue
        if re.match(r"\(\d{1,2}:\d{2}\)", text) and not h.description:
            h.description = text
            continue
        if h.description and wrapped is None:
            wrapped = text if (re.search(r"[a-z]", text) and not re.search(r"[.!]$", h.description)) else ""
            if wrapped:
                h.description += " " + wrapped
    h.description = " ".join(h.description.split())
    return h


def read_screen(frame: np.ndarray, downscale: int = 2):
    """OCR the captured window once: the play header, and the "On Field Players" list if it
    is showing. Under a second; call it once per shot."""
    from .ocr import read_text
    from .personnel import parse_on_field

    from .personnel import list_head

    small = frame[::downscale, ::downscale] if downscale > 1 else frame
    boxes = [b for b in read_text(np.ascontiguousarray(small)) if b.confidence >= 0.3]
    on_field = None
    head = list_head(boxes)
    if head is not None:
        # the list again at full size: its one-letter positions (C, G, T, S) are too small at half size
        x0 = max(0, int(head[0] * downscale) - 40)
        y0 = max(0, int(head[1] * downscale) - 60)                  # the title or the first bar stays in the crop
        # ...and half as large again: two-letter positions next to a long name otherwise go unread now and then
        crop = cv2.resize(frame[y0:, x0:], None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        on_field = parse_on_field([b for b in read_text(crop) if b.confidence >= 0.3])
    header = parse_header([b.text for b in boxes])
    if not header.complete:
        strip = _header_strip(frame, boxes, downscale)
        if strip:
            header = parse_header(strip + [b.text for b in boxes])
    if header.offense is None and on_field is not None:
        header.offense = TEAMS.get(on_field.offense_team.strip().lower())
    return header, on_field


def text_rows(boxes) -> list[str]:
    """Boxes that sit on one line, joined left to right: "Q1" and "04:25" read apart are one row."""
    rows: list[list] = []
    for b in sorted(boxes, key=lambda b: (b.box[1] + b.box[3]) / 2):
        mid, tall = (b.box[1] + b.box[3]) / 2, b.box[3] - b.box[1]
        if rows and abs(mid - rows[-1][0]) <= 0.5 * tall:
            rows[-1][1].append(b)
        else:
            rows.append([mid, [b]])
    return [" ".join(b.text.strip() for b in sorted(items, key=lambda b: b.box[0])) for _, items in rows]


def _header_strip(frame: np.ndarray, boxes, downscale: int) -> list[str]:
    """The newer header's two lines, read again at twice the size: the quarter and clock
    under the down are too small for the whole-window pass, and the yard number can come
    apart from its line."""
    from .ocr import read_text

    downs = [b for b in boxes if DOWN_LINE.match(b.text.strip())]
    if not downs:
        return []
    x1, y1, x2, y2 = (int(v * downscale) for v in min(downs, key=lambda b: b.box[1]).box)
    crop = frame[max(0, y1 - 16): y2 + 44, max(0, x1 - 120): x2 + 120]
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return text_rows([b for b in read_text(np.ascontiguousarray(big)) if b.confidence >= 0.3])


def load_play_by_play(season: int, root: str | Path = NFLVERSE):
    import polars as pl

    path = Path(root) / f"play_by_play_{season}.parquet"
    if not path.exists():
        return None
    cols = ["game_id", "old_game_id", "play_id", "week", "posteam", "defteam", "qtr", "time", "down", "ydstogo",
            "yrdln", "yardline_100", "play_type", "yards_gained", "epa", "touchdown", "complete_pass", "desc",
            "shotgun", "pass_location", "run_location", "run_gap", "air_yards", "passer_player_name",
            "receiver_player_name", "rusher_player_name", "sack", "interception"]
    df = pl.read_parquet(path)
    return df.select([c for c in cols if c in df.columns])


def latest_season(root: str | Path = NFLVERSE) -> int | None:
    """The newest season with a play-by-play file on disk."""
    years = [int(m.group(1)) for f in Path(root).glob("play_by_play_*.parquet") if (m := re.search(r"(\d{4})", f.name))]
    return max(years) if years else None


def load_charting(season: int, root: str | Path = NFLVERSE):
    """FTN's charting of each play: where the quarterback lined up, the box, the rush."""
    import polars as pl

    path = Path(root) / f"ftn_charting_{season}.parquet"
    return pl.read_parquet(path) if path.exists() else None


_TABLES: dict = {}


def _cached(kind: str, season: int, root: str | Path):
    key = (kind, season, str(root))
    if key not in _TABLES:
        _TABLES[key] = (load_play_by_play if kind == "pbp" else load_charting)(season, root)
    return _TABLES[key]


@dataclass
class PlayRecord:
    header: PlayHeader
    matched: bool = False
    game_id: str = ""
    play_id: int = 0
    defense: str = ""
    yards_to_goal: int | None = None
    play_type: str = ""
    yards_gained: float | None = None
    epa: float | None = None
    touchdown: bool = False
    description: str = ""
    candidates: int = 0
    notes: list[str] = field(default_factory=list)
    complete: bool = False
    sack: bool = False
    interception: bool = False
    passer: str = ""
    receiver: str = ""
    rusher: str = ""
    air_yards: float | None = None
    pass_location: str = ""                  # left / middle / right, from the offense's side
    shotgun: bool | None = None              # the play-by-play says whether the quarterback was in the shotgun
    charted: dict = field(default_factory=dict)      # FTN: qb_location, n_defense_box, n_pass_rushers, ...
    on_field: object = None                  # personnel.OnField, when the list was on screen

    @property
    def hash_mark(self) -> str | None:
        """Where the ball was spotted, from the charting: "L", "M" or "R" from the offense's side."""
        mark = self.charted.get("starting_hash")
        return mark if mark in ("L", "M", "R") else None

    @property
    def from_scrimmage(self) -> bool:
        """A run or a pass. Kicks, kneel-downs, spikes and plays wiped out by a flag are not analysed."""
        return not self.matched or self.play_type in ("pass", "run")

    @property
    def is_pass(self) -> bool:
        return self.play_type == "pass"

    @property
    def outcome(self) -> str:
        """What happened, in a few words."""
        if not self.matched:
            return ""
        name = lambda n: n.split(".")[-1] if n else ""          # "J.Love" -> "Love"
        yards = 0 if self.yards_gained is None else int(round(self.yards_gained))
        if self.sack:
            text = f"Sack, {yards} yd"
        elif self.interception:
            text = "Intercepted"
        elif self.is_pass:
            who = f"{name(self.passer)} to {name(self.receiver)}" if self.receiver else "Pass"
            text = f"{who}, {yards} yd" if self.complete else f"{who}, incomplete"
        elif self.play_type == "run":
            text = f"{name(self.rusher) or 'Run'}, {yards} yd"
        else:
            text = f"{self.play_type or 'play'}, {yards} yd"
        return text + (", touchdown" if self.touchdown else "")

    @property
    def charted_look(self) -> str:
        c = self.charted
        if not c:
            return ""
        parts = [{"S": "Shotgun", "U": "Under centre", "P": "Pistol"}.get(c.get("qb_location") or "", "")]
        if c.get("n_defense_box") is not None:
            parts.append(f"{c['n_defense_box']} in the box")
        if self.is_pass and c.get("n_pass_rushers") is not None:
            parts.append(f"{c['n_pass_rushers']} rush")
        return ", ".join(x for x in parts if x)

    @property
    def summary(self) -> str:
        h = self.header
        where = f"{h.offense} on {h.side + ' ' if h.side else ''}{h.yard}"
        down = f"{['', '1st', '2nd', '3rd', '4th'][h.down]} & {h.distance if h.distance is not None else 'goal'}" if h.down else ""
        return f"Q{h.quarter} {h.clock}  {down}  {where}".strip()


def _clock_seconds(text: str) -> int | None:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def _who_has_the_ball(header: PlayHeader, pbp) -> str | None:
    """The newer header gives the clock and the yard line but not the offense. Within one
    game that is one play; the row says whose ball it is. None unless exactly one team fits."""
    import polars as pl

    if header.game is None:
        return None
    secs = _clock_seconds(header.clock)
    df = pbp.filter(pl.col("posteam").is_in(list(header.game)) & pl.col("defteam").is_in(list(header.game))
                    & (pl.col("qtr") == header.quarter))
    df = df.filter(pl.col("time").map_elements(lambda s: _clock_seconds(s) if s else None, return_dtype=pl.Int64) == secs)
    if header.week is not None:
        df = df.filter(pl.col("week") == header.week)
    if header.down is not None:
        df = df.filter(pl.col("down") == header.down)
    want = f"{header.side} {header.yard}" if header.side else str(header.yard)
    narrowed = df.filter(pl.col("yrdln") == want)
    if narrowed.height:
        df = narrowed
    teams = set(df["posteam"].to_list())
    return teams.pop() if len(teams) == 1 else None


def match_play(header: PlayHeader, pbp=None, root: str | Path = NFLVERSE) -> PlayRecord:
    """Find the header's play in the play-by-play. Unmatched plays still carry what the
    header itself says (the yard line is enough to anchor the field)."""
    rec = PlayRecord(header=header, yards_to_goal=header.yards_to_goal, description=header.description)
    if not header.placed:
        rec.notes.append("could not read the play header")
        return rec
    if pbp is None:
        if header.season is None:
            header.season = latest_season(root)            # nothing on screen says: assume the newest film we have plays for
        if header.season is not None:
            pbp = _cached("pbp", header.season, root)
    if header.offense is None and pbp is not None:
        header.offense = _who_has_the_ball(header, pbp)
        rec.yards_to_goal = header.yards_to_goal
    if not header.complete:
        rec.notes.append("could not read the play header")
        return rec
    if pbp is None:
        rec.notes.append("no play-by-play file for this season")
        return rec
    import polars as pl

    secs = _clock_seconds(header.clock)
    df = pbp.filter((pl.col("posteam") == header.offense) & (pl.col("qtr") == header.quarter))
    df = df.with_columns(
        pl.col("time").map_elements(lambda s: _clock_seconds(s) if s else None, return_dtype=pl.Int64).alias("_secs")
    ).filter(pl.col("_secs") == secs)
    if header.week is not None:
        df = df.filter(pl.col("week") == header.week)
    if header.game is not None and "defteam" in df.columns:
        df = df.filter(pl.col("defteam").is_in(list(header.game)))
    if header.down is not None:
        df = df.filter(pl.col("down") == header.down)
    rec.candidates = df.height
    if df.height == 0:
        rec.notes.append("no play at that clock in the play-by-play")
        return rec
    if df.height > 1 and header.yard is not None:
        want = f"{header.side} {header.yard}" if header.side else str(header.yard)
        narrowed = df.filter(pl.col("yrdln") == want)
        if narrowed.height:
            df = narrowed
    row = df.row(0, named=True)
    rec.matched = True
    rec.game_id = row["game_id"]
    rec.play_id = int(row["play_id"])
    rec.defense = row["defteam"] or ""
    rec.yards_to_goal = int(row["yardline_100"]) if row["yardline_100"] is not None else header.yards_to_goal
    rec.play_type = row["play_type"] or ""
    rec.yards_gained = row["yards_gained"]
    rec.epa = row["epa"]
    rec.touchdown = bool(row["touchdown"])
    rec.description = row["desc"] or header.description
    rec.complete = bool(row.get("complete_pass"))
    rec.sack = bool(row.get("sack"))
    rec.interception = bool(row.get("interception"))
    rec.passer = row.get("passer_player_name") or ""
    rec.receiver = row.get("receiver_player_name") or ""
    rec.rusher = row.get("rusher_player_name") or ""
    rec.air_yards = row.get("air_yards")
    rec.pass_location = row.get("pass_location") or ""
    rec.shotgun = None if row.get("shotgun") is None else bool(row["shotgun"])
    season = header.season or int(str(rec.game_id)[:4])
    charting = _cached("ftn", season, root)
    if charting is not None:
        hit = charting.filter((pl.col("nflverse_game_id") == rec.game_id) & (pl.col("nflverse_play_id") == rec.play_id))
        if hit.height:
            rec.charted = {k: v for k, v in hit.row(0, named=True).items() if k.startswith(("n_", "is_", "qb_", "starting_", "read_"))}
    return rec
