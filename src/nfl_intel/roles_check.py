"""Score role assignment against real snaps.

Big Data Bowl tracking says who the passer, the route runners and the coverage defenders
were on every pass play, so the rules in roles.py can be marked instead of eyeballed. The
set has no linemen at all, which makes it a hard version of what film gives us (linemen
lost in the pile), and noise can be added to stand in for positions measured from video.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .bdb import HZ, Play
from .fieldtrack import Slot
from .roles import assign_roles, update_roles


@dataclass
class RoleScore:
    plays: int = 0
    counts: Counter = field(default_factory=Counter)

    def share(self, hit: str, of: str) -> float:
        return self.counts[hit] / self.counts[of] if self.counts[of] else float("nan")

    def lines(self) -> list[str]:
        c = self.counts

        def row(name: str, hit: str, of: str) -> str:
            return f"  {name:42s} {self.share(hit, of):6.1%}   ({c[hit]} of {c[of]})"

        return [
            row("quarterback picked out", "qb right", "plays"),
            row("   from the shotgun", "qb right gun", "gun"),
            row("   under centre", "qb right under", "under"),
            row("route runners named receivers", "rec right", "rec"),
            row("   after a second of the play", "rec right later", "rec"),
            row("coverage men named coverage", "def right", "def"),
            row("   after a second of the play", "def right later", "def"),
        ]


def score_roles(plays: Iterable[Play], hints: dict | None = None, noise: float = 0.0, seed: int = 0, later: float = 1.2,
                use_hints: bool = True) -> RoleScore:
    """`hints` maps (game_id, play_id) to (shotgun, hash_mark), as the play's charting would
    give them. With `use_hints` off the rules see the formation alone; the hints then only
    split the score into shotgun and under-centre plays."""
    rng = np.random.default_rng(seed)
    out = RoleScore()
    c = out.counts
    for play in plays:
        if play.qb < 0 or len(play.xy) < 2:
            continue
        shotgun, hash_mark = (hints or {}).get((play.game_id, play.play_id), (None, None))
        xy = play.xy[0] + rng.normal(0.0, noise, play.xy[0].shape)
        slots = [Slot(i, xy[i], np.zeros(2), 0 if play.offense[i] else 1, 0.0, 0.0, 0.0, hits=10) for i in range(len(xy))]
        roles = assign_roles(slots, xy, 0, play.los, *((shotgun, hash_mark) if use_hints else (None, None)))
        out.plays += 1
        c["plays"] += 1
        right = [i for i, r in roles.items() if r == "QB"] == [play.qb]
        c["qb right"] += right
        if shotgun is not None:
            kind = "gun" if shotgun else "under"
            c[kind] += 1
            c[f"qb right {kind}"] += right
        k = min(len(play.xy) - 1, int(round(later * HZ)))
        moved = dict(roles)
        update_roles(moved, {i: play.xy[k, i] + rng.normal(0.0, noise, 2) for i in range(len(xy))}, play.los, k / HZ)
        for i in play.receivers:
            c["rec"] += 1
            c["rec right"] += roles[i] == "REC"
            c["rec right later"] += moved[i] == "REC"
        for i in play.defenders:
            c["def"] += 1
            c["def right"] += roles[i] == "DEF"
            c["def right later"] += moved[i] == "DEF"
    return out


def charted_hints(season: int = 2023, root: str = "data/nflverse") -> dict:
    """(old game id, play id) -> (shotgun, hash mark) from play-by-play and FTN charting."""
    from pathlib import Path

    import polars as pl

    pbp = pl.read_parquet(Path(root) / f"play_by_play_{season}.parquet", columns=["game_id", "old_game_id", "play_id", "shotgun"])
    pbp = pbp.with_columns(pl.col("play_id").cast(pl.Int64))
    hints = {(int(g), int(p)): (None if s is None else bool(s), None) for _, g, p, s in pbp.iter_rows()}
    ftn_path = Path(root) / f"ftn_charting_{season}.parquet"
    if ftn_path.exists():
        ftn = pl.read_parquet(ftn_path).with_columns(pl.col("nflverse_play_id").cast(pl.Int64))
        ftn = ftn.join(pbp, left_on=["nflverse_game_id", "nflverse_play_id"], right_on=["game_id", "play_id"], how="inner")
        for r in ftn.iter_rows(named=True):
            key = (int(r["old_game_id"]), int(r["nflverse_play_id"]))
            shotgun = hints.get(key, (None, None))[0]
            if r["qb_location"] in ("S", "P", "U"):
                shotgun = r["qb_location"] != "U"
            hints[key] = (shotgun, r["starting_hash"] if r["starting_hash"] in ("L", "M", "R") else None)
    return hints


def check(weeks: list[int], noise: float = 0.4, root: str = "data/bdb/2026-prediction", labels: str = "data/nflverse") -> dict[str, RoleScore]:
    from .bdb import iter_plays

    hints = charted_hints(2023, labels)
    out = {}
    for name, use in (("formation alone", False), ("with the play's charting", True)):
        out[name] = score_roles((p for w in weeks for p in iter_plays(w, root)), hints, noise=noise, use_hints=use)
        print(f"{name}: {out[name].plays} pass plays, positions jittered by {noise:g} yd")
        print("\n".join(out[name].lines()))
    return out
