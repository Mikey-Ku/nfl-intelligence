"""Fetch the public nflverse files the analysis reads: play-by-play, FTN charting, participation.

They are release assets of github.com/nflverse/nflverse-data, rebuilt during the season as
games finish, so a play from last night needs a fresh copy. Nothing here needs an account.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
FILES = {
    "play-by-play": "pbp/play_by_play_{season}.parquet",
    "charting": "ftn_charting/ftn_charting_{season}.parquet",
    "participation": "pbp_participation/pbp_participation_{season}.parquet",
}


def fetch(url: str, path: Path, timeout: float = 60.0) -> int | None:
    """Download to `path` (written whole or not at all). Returns the size, or None if the file isn't published."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return len(data)


def refresh(season: int, root: str | Path = "data/nflverse", kinds=tuple(FILES)) -> dict[str, int | None]:
    """Fetch this season's files. Returns {kind: size in bytes, or None when not published yet}."""
    out = {}
    for kind in kinds:
        rel = FILES[kind].format(season=season)
        out[kind] = fetch(f"{BASE}/{rel}", Path(root) / rel.split("/")[-1])
    return out


def describe(season: int, root: str | Path = "data/nflverse") -> str:
    """One line on what the play-by-play on disk covers."""
    import polars as pl

    path = Path(root) / f"play_by_play_{season}.parquet"
    if not path.exists():
        return f"no play-by-play for {season} on disk"
    df = pl.read_parquet(path, columns=["week", "game_id"])
    return f"{season}: {df.height} plays, {df['game_id'].n_unique()} games, through week {df['week'].max()}"
