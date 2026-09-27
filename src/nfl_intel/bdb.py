"""NFL Big Data Bowl 2026 tracking, loaded one play at a time and normalised.

The set covers pass plays from the 2023 season: about 2.8 s of 10 Hz positions from the snap
to the throw for the passer, the route runners and the coverage defenders (no linemen, no
ball). Outcomes are not in it; they come from nflverse play-by-play joined on game and play.

Everything is flipped so the offense moves toward +x, in the same yard coordinates as
field.py (x 0-120 with goal lines at 10 and 110, y 0-53.3).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .field import FIELD_LENGTH, FIELD_WIDTH

DEFAULT_ROOT = Path("data/bdb/2026-prediction")
DEFAULT_OUTCOMES = Path("data/nflverse/play_by_play_2023.parquet")
DEFAULT_PARTICIPATION = Path("data/nflverse/pbp_participation_2023.parquet")
HZ = 10.0
ROLE = {"Passer": "QB", "Targeted Receiver": "REC", "Other Route Runner": "REC", "Defensive Coverage": "DEF"}


@dataclass
class Play:
    game_id: int
    play_id: int
    week: int
    los: float                  # line of scrimmage, normalised x
    xy: np.ndarray              # (frames, players, 2) from the snap to the throw
    offense: np.ndarray         # (players,) bool
    roles: list[str]            # QB / REC / DEF
    names: list[str]
    positions: list[str]
    qb: int                     # index of the passer, -1 if missing
    target: int                 # index of the targeted receiver, -1 if missing
    ball_land: tuple[float, float]
    complete: bool | None = None
    yards: float | None = None
    epa: float | None = None
    desc: str = ""
    # charted per play by NFL Next Gen Stats, published in nflverse participation data
    pressure: bool | None = None
    time_to_throw: float | None = None
    man_zone: str | None = None        # "man" / "zone"
    coverage: str | None = None        # COVER_1, COVER_3, ...
    formation: str | None = None       # SHOTGUN, UNDER CENTER, ...
    pass_rushers: int | None = None

    @property
    def receivers(self) -> list[int]:
        return [i for i, r in enumerate(self.roles) if r == "REC"]

    @property
    def defenders(self) -> list[int]:
        return [i for i, r in enumerate(self.roles) if r == "DEF"]


def load_outcomes(path: str | Path = DEFAULT_OUTCOMES, participation: str | Path | None = DEFAULT_PARTICIPATION) -> pl.DataFrame:
    """Play results from nflverse play-by-play, plus pressure / coverage / formation labels
    from nflverse participation data when that file is present."""
    pbp = pl.read_parquet(path, columns=[
        "old_game_id", "play_id", "complete_pass", "interception", "yards_gained", "air_yards",
        "epa", "down", "ydstogo", "desc",
    ])
    out = pbp.select(
        pl.col("old_game_id").cast(pl.Int64).alias("game_id"),
        pl.col("play_id").cast(pl.Int64),
        (pl.col("complete_pass") == 1).alias("complete"),
        (pl.col("interception") == 1).alias("interception"),
        pl.col("yards_gained").cast(pl.Float64).alias("yards"),
        pl.col("air_yards").cast(pl.Float64),
        pl.col("epa").cast(pl.Float64),
        "down", "ydstogo", "desc",
    )
    labels = ["pressure", "time_to_throw", "man_zone", "coverage", "formation", "pass_rushers"]
    if participation is not None and Path(participation).exists():
        part = pl.read_parquet(participation).select(
            pl.col("old_game_id").cast(pl.Int64).alias("game_id"),
            pl.col("play_id").cast(pl.Int64),
            pl.col("was_pressure").alias("pressure"),
            pl.col("time_to_throw").cast(pl.Float64),
            pl.col("defense_man_zone_type").replace({"MAN_COVERAGE": "man", "ZONE_COVERAGE": "zone", "": None}).alias("man_zone"),
            pl.col("defense_coverage_type").replace({"": None}).alias("coverage"),
            pl.col("offense_formation").alias("formation"),
            pl.col("number_of_pass_rushers").cast(pl.Int64).alias("pass_rushers"),
        )
        return out.join(part, on=["game_id", "play_id"], how="left")
    return out.with_columns([pl.lit(None).alias(c) for c in labels])


def load_week(week: int, root: str | Path = DEFAULT_ROOT) -> pl.DataFrame:
    """One week of input tracking with normalised coordinates (columns xn, yn, los, blx, bly)."""
    df = pl.read_csv(Path(root) / "train" / f"input_2023_w{week:02d}.csv")
    flip = pl.col("play_direction") == "left"
    return df.with_columns(
        xn=pl.when(flip).then(FIELD_LENGTH - pl.col("x")).otherwise(pl.col("x")),
        yn=pl.when(flip).then(FIELD_WIDTH - pl.col("y")).otherwise(pl.col("y")),
        los=pl.when(flip).then(FIELD_LENGTH - pl.col("absolute_yardline_number")).otherwise(pl.col("absolute_yardline_number")).cast(pl.Float64),
        blx=pl.when(flip).then(FIELD_LENGTH - pl.col("ball_land_x")).otherwise(pl.col("ball_land_x")),
        bly=pl.when(flip).then(FIELD_WIDTH - pl.col("ball_land_y")).otherwise(pl.col("ball_land_y")),
    )


def iter_plays(week: int, root: str | Path = DEFAULT_ROOT, outcomes: pl.DataFrame | None = None) -> Iterator[Play]:
    df = load_week(week, root).sort(["game_id", "play_id", "nfl_id", "frame_id"])
    out = {}
    if outcomes is not None:
        out = {(r["game_id"], r["play_id"]): r for r in outcomes.iter_rows(named=True)}
    for (game_id, play_id), g in df.group_by(["game_id", "play_id"], maintain_order=True):
        n_frames = int(g["frame_id"].max())
        players = g.unique("nfl_id", maintain_order=True)
        ids = players["nfl_id"].to_list()
        if g.height != n_frames * len(ids):
            continue                                   # a player missing frames: skip the play
        xy = np.stack([g["xn"].to_numpy(), g["yn"].to_numpy()], axis=1).reshape(len(ids), n_frames, 2).transpose(1, 0, 2)
        roles_raw = players["player_role"].to_list()
        roles = [ROLE.get(r, "DEF") for r in roles_raw]
        o = out.get((game_id, play_id))
        yield Play(
            game_id=int(game_id), play_id=int(play_id), week=week,
            los=float(players["los"][0]),
            xy=np.ascontiguousarray(xy),
            offense=(players["player_side"] == "Offense").to_numpy(),
            roles=roles,
            names=players["player_name"].to_list(),
            positions=players["player_position"].to_list(),
            qb=roles.index("QB") if "QB" in roles else -1,
            target=roles_raw.index("Targeted Receiver") if "Targeted Receiver" in roles_raw else -1,
            ball_land=(float(players["blx"][0]), float(players["bly"][0])),
            complete=None if o is None else bool(o["complete"]),
            yards=None if o is None or o["yards"] is None else float(o["yards"]),
            epa=None if o is None or o["epa"] is None else float(o["epa"]),
            desc="" if o is None else (o["desc"] or ""),
            pressure=None if o is None or o["pressure"] is None else bool(o["pressure"]),
            time_to_throw=None if o is None else o["time_to_throw"],
            man_zone=None if o is None else o["man_zone"],
            coverage=None if o is None else o["coverage"],
            formation=None if o is None else o["formation"],
            pass_rushers=None if o is None else o["pass_rushers"],
        )
