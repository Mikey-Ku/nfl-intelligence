"""The game table: a row per player per play from a run's CSV and play log, and the totals."""

import json

import numpy as np

from nfl_intel.gametable import player_rows, player_table, player_totals, write
from nfl_intel.schema import FrameResult, TrackedPlayer, write_csv


def _run(tmp_path):
    """A 4-second play at 20 fps: the snap at 1.0 s, a receiver who runs 20 yards downfield in
    the two seconds after it, a quarterback who stands still, and one unnamed defender."""
    frames = []
    for i in range(80):
        t = i / 20.0
        gone = max(0.0, min(t - 1.0, 2.0)) * 10.0
        frames.append(FrameResult(frame=i, t=t, size=(100, 100), shot=2, anchored=True, players=[
            TrackedPlayer(1, 0, (0, 0, 10, 20), 0.9, (5, 20), (70.0 + gone, 10.0), role="REC", label="Golden"),
            TrackedPlayer(2, 0, (20, 0, 30, 20), 0.9, (25, 20), (65.0, 26.7), role="QB", label="QB"),
            TrackedPlayer(3, 1, (40, 0, 50, 20), 0.9, (45, 20), (78.0, 10.0), role="DEF", label="Terrell"),
            TrackedPlayer(4, 1, (60, 0, 70, 20), 0.9, (65, 20), (85.0, 26.7), role="DEF", label="D2")]))
    write_csv(tmp_path / "clip.csv", frames)
    play = {"shot": 2, "matched": True, "game_id": "2026_03_GB_ATL", "play_id": 7, "quarter": 4, "clock": "08:21", "down": 2, "distance": 7,
            "offense": "GB", "defense": "ATL", "play_type": "pass", "result": "Love to Golden, 15 yd", "yards": 15.0, "epa": 2.1,
            "target": "M.Golden", "offense_look": "Shotgun, 3 x 1, 1 back", "defense_look": "Single high", "yards_to_goal": 40.0,
            "snap": 1.0, "numbers": {"Golden": 80}, "looks": {"Terrell": 21},
            "value": [{"t": 0.5, "value": 6.0, "best": "Golden"}, {"t": 1.0, "value": 9.0, "best": "Golden"}, {"t": 1.5, "value": 7.0, "best": "R2"}]}
    (tmp_path / "clip.plays.json").write_text(json.dumps([play]))
    return tmp_path / "clip.csv"


def test_player_rows_say_who_where_and_how_fast(tmp_path):
    rows = {r["label"]: r for r in player_rows(_run(tmp_path))}
    assert set(rows) == {"Golden", "QB", "Terrell", "D2"}
    g = rows["Golden"]
    assert (g["name"], g["number"], g["known_by"], g["team"], g["side"]) == ("Golden", 80, "number", "GB", "offense")
    assert g["depth_at_snap"] == 0.0 and g["snap_y"] == 10.0            # on the line of scrimmage at the 30 (x = 70)
    assert 9.0 <= g["top_speed"] <= 10.5 and 19.0 <= g["distance_run"] <= 21.0
    assert g["moments_best"] == 2 and g["peak_value_as_best"] == 9.0
    assert rows["Terrell"]["known_by"] == "look" and rows["Terrell"]["team"] == "ATL" and rows["Terrell"]["depth_at_snap"] == 8.0
    assert rows["QB"]["name"] is None and rows["QB"]["known_by"] == "formation" and rows["QB"]["top_speed"] == 0.0
    assert rows["D2"]["name"] is None and rows["D2"]["result"] == "Love to Golden, 15 yd"


def test_table_and_totals_round_trip(tmp_path):
    csv = _run(tmp_path)
    table = player_table([csv])
    assert table.shape[0] == 4 and table["play_id"].to_list() == [7] * 4
    totals = player_totals(table)
    assert totals["name"].to_list() == ["Terrell", "Golden"]            # by team, then plays
    golden = totals.filter(totals["name"] == "Golden").row(0, named=True)
    assert golden["plays"] == 1 and golden["by_number"] == 1 and golden["by_look"] == 0 and golden["moments_best"] == 2
    out = write(table, tmp_path / "game.parquet")
    import polars as pl

    assert pl.read_parquet(out).shape == table.shape
    assert write(table, tmp_path / "game.csv").exists()
