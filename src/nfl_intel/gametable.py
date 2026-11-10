"""One table for a game: a row per player per play, from the runs' tracking CSVs and play logs.

Everything the pipeline works out ends up in two files per run: the tracking CSV (a row per
player per frame) and `<clip>.plays.json` (a dict per play). Neither is handy for a
question like "how deep did Coker line up on third downs" or "who was the best option most
often". This puts the two together, at the grain a football question is asked at:

    player rows   one per player per play: who he is and how we know, where he stood at the
                  snap, how fast he ran, how often he was the best throw
    player totals one per player per game: snaps, how often he was named and how, averages

    uv run nfl game-table outputs/live-02.csv outputs/live-03.csv --out outputs/game.parquet

The table is plain Polars; CSV and Parquet both work, by the extension of `--out`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from .schema import read_csv

FORMATION_LABEL = ("QB", "R?", "D?")                    # labels the formation gives, not a name
OFFENSE_ROLES = ("QB", "REC", "OL")
AFTER_SNAP = 5.0                                       # seconds of the play that count for speed and distance
LAG = 0.4                                              # seconds between positions when measuring speed
MAX_SPEED = 12.0                                       # yd/s; faster is a tracking glitch, not a man


def _is_formation_label(label: str | None) -> bool:
    return label is None or label in FORMATION_LABEL or (label[0] in "RD" and label[1:].isdigit())


def _speed_and_distance(t: np.ndarray, xy: np.ndarray, snap: float) -> tuple[float | None, float | None]:
    """Top speed (yd/s) and ground covered (yd) in the first seconds after the snap."""
    live = (t >= snap) & (t <= snap + AFTER_SNAP)
    if live.sum() < 3:
        return None, None
    t, xy = t[live], xy[live]
    speeds = []
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] - LAG)
        if j < i and t[i] - t[j] >= LAG * 0.7:
            speeds.append(float(np.linalg.norm(xy[i] - xy[j]) / (t[i] - t[j])))
    top = min(max(speeds), MAX_SPEED) if speeds else None
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(t)
    ok = dt > 0
    distance = float(np.minimum(steps[ok], MAX_SPEED * dt[ok]).sum()) if ok.any() else None
    return top, distance


def player_rows(csv_path: str | Path, plays_path: str | Path | None = None) -> list[dict]:
    """Rows for every labelled player on every play worked out in one run."""
    csv_path = Path(csv_path)
    plays_path = Path(plays_path) if plays_path is not None else csv_path.with_suffix("").with_suffix(".plays.json")
    plays = {p["shot"]: p for p in json.loads(Path(plays_path).read_text()) if p.get("matched")}
    tracks: dict = defaultdict(list)                   # (shot, label) -> [(t, x, y, role, team)]
    for f in read_csv(csv_path):
        if f.shot not in plays or not f.anchored:
            continue
        for p in f.players:
            if p.label is None or p.field_yd is None:
                continue
            tracks[(f.shot, p.label)].append((f.t, p.field_yd[0], p.field_yd[1], p.role))
    rows = []
    for (shot, label), obs in sorted(tracks.items()):
        play = plays[shot]
        obs.sort()
        t = np.array([o[0] for o in obs])
        xy = np.array([[o[1], o[2]] for o in obs])
        roles = [o[3] for o in obs if o[3]]
        role = max(set(roles), key=roles.count) if roles else None
        side = "offense" if role in OFFENSE_ROLES else "defense"
        team = play["offense"] if side == "offense" else play["defense"]
        numbers, looks = play.get("numbers", {}), play.get("looks", {})
        if label in numbers:
            name, number, known_by = label, numbers[label], "number"
        elif label in looks:
            name, number, known_by = label, looks[label], "look"
        elif _is_formation_label(label):
            name, number, known_by = None, None, "formation"
        else:
            name, number, known_by = label, None, "formation"
        snap = play.get("snap")
        los = 110.0 - float(play["yards_to_goal"]) if play.get("yards_to_goal") is not None else None
        at_snap = None
        if snap is not None:
            k = int(np.argmin(np.abs(t - snap)))
            if abs(t[k] - snap) <= 0.5:
                at_snap = xy[k]
        top, distance = _speed_and_distance(t, xy, snap) if snap is not None else (None, None)
        curve = play.get("value") or []
        best_moments = sum(1 for c in curve if c.get("best") == label)
        peak_as_best = max((c["value"] for c in curve if c.get("best") == label), default=None)
        rows.append({
            "game_id": play.get("game_id"), "play_id": play.get("play_id"), "shot": shot, "clip": csv_path.stem,
            "quarter": play.get("quarter"), "clock": play.get("clock"), "down": play.get("down"), "distance": play.get("distance"),
            "offense": play.get("offense"), "defense": play.get("defense"), "play_type": play.get("play_type"),
            "result": play.get("result"), "yards": play.get("yards"), "epa": play.get("epa"), "target": play.get("target"),
            "offense_look": play.get("offense_look"), "defense_look": play.get("defense_look"),
            "team": team, "side": side, "role": role, "label": label, "name": name, "number": number, "known_by": known_by,
            "frames": len(obs), "first_t": float(t[0]), "last_t": float(t[-1]),
            "snap_x": None if at_snap is None else round(float(at_snap[0]), 2),
            "snap_y": None if at_snap is None else round(float(at_snap[1]), 2),
            "depth_at_snap": None if at_snap is None or los is None else round(float(at_snap[0] - los), 2),
            "top_speed": None if top is None else round(top, 2),
            "distance_run": None if distance is None else round(distance, 1),
            "moments_best": best_moments, "peak_value_as_best": peak_as_best,
        })
    return rows


def player_table(csv_paths, plays_paths=None) -> pl.DataFrame:
    """Rows from several runs of the same game, as one frame."""
    csv_paths = [Path(p) for p in csv_paths]
    plays_paths = list(plays_paths) if plays_paths else [None] * len(csv_paths)
    rows = [r for c, p in zip(csv_paths, plays_paths) for r in player_rows(c, p)]
    schema = {"game_id": pl.Utf8, "play_id": pl.Int64, "shot": pl.Int64, "clip": pl.Utf8, "quarter": pl.Int64, "clock": pl.Utf8,
              "down": pl.Int64, "distance": pl.Int64, "offense": pl.Utf8, "defense": pl.Utf8, "play_type": pl.Utf8, "result": pl.Utf8,
              "yards": pl.Float64, "epa": pl.Float64, "target": pl.Utf8, "offense_look": pl.Utf8, "defense_look": pl.Utf8,
              "team": pl.Utf8, "side": pl.Utf8, "role": pl.Utf8, "label": pl.Utf8, "name": pl.Utf8, "number": pl.Int64, "known_by": pl.Utf8,
              "frames": pl.Int64, "first_t": pl.Float64, "last_t": pl.Float64, "snap_x": pl.Float64, "snap_y": pl.Float64,
              "depth_at_snap": pl.Float64, "top_speed": pl.Float64, "distance_run": pl.Float64, "moments_best": pl.Int64,
              "peak_value_as_best": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def player_totals(table: pl.DataFrame) -> pl.DataFrame:
    """One row per named player per game: how often he was seen and named, and his averages."""
    named = table.filter(pl.col("name").is_not_null())
    if named.is_empty():
        return pl.DataFrame()
    return (named.group_by("game_id", "team", "name")
            .agg(pl.col("number").drop_nulls().first().alias("number"),
                 pl.col("role").mode().first().alias("role"),
                 pl.len().alias("plays"),
                 (pl.col("known_by") == "number").sum().alias("by_number"),
                 (pl.col("known_by") == "look").sum().alias("by_look"),
                 pl.col("depth_at_snap").mean().round(2).alias("mean_depth_at_snap"),
                 pl.col("top_speed").max().alias("top_speed"),
                 pl.col("distance_run").mean().round(1).alias("mean_distance_run"),
                 pl.col("moments_best").sum().alias("moments_best"))
            .sort("team", "plays", descending=[False, True]))


def write(table: pl.DataFrame, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        table.write_parquet(out)
    else:
        table.write_csv(out)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="One table for a game: a row per player per play, from runs' CSVs and play logs.")
    ap.add_argument("csvs", nargs="+", help="tracking CSVs; each .plays.json sits beside its CSV")
    ap.add_argument("--out", required=True, help=".parquet or .csv; the player totals go beside it as <out>.totals.<ext>")
    a = ap.parse_args(argv)
    table = player_table(a.csvs)
    out = write(table, a.out)
    totals = player_totals(table)
    tot_path = Path(a.out).with_suffix(".totals" + Path(a.out).suffix)
    if not totals.is_empty():
        write(totals, tot_path)
    print(f"{len(table)} player-plays on {table['play_id'].n_unique()} plays -> {out}")
    if not totals.is_empty():
        print(f"{len(totals)} named players -> {tot_path}")
        with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=160):
            print(totals)


if __name__ == "__main__":
    main()
