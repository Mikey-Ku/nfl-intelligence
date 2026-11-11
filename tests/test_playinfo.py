"""Reading the play's name and finding it in a play-by-play table; no OCR, no files."""

import polars as pl

from nfl_intel.playinfo import PlayHeader, PlayRecord, match_play, parse_header


def test_parse_header_reads_the_lines_in_any_order():
    h = parse_header(["nflpro.example/film?season=2026&week=WEEK_3", "GB ON ATL 15", "junk", "Q4 08:21", "2ND & 7",
                      "(8:21) (Shotgun) J.Love pass short right to", "M.Golden for 15 yards, TOUCHDOWN."])
    assert (h.quarter, h.clock, h.down, h.distance) == (4, "08:21", 2, 7)
    assert (h.offense, h.side, h.yard) == ("GB", "ATL", 15)
    assert h.season == 2026 and h.week == 3
    assert h.description.endswith("M.Golden for 15 yards, TOUCHDOWN.")
    assert h.complete and h.yards_to_goal == 15


def test_yards_to_goal_depends_on_whose_half():
    assert PlayHeader(offense="GB", side="GB", yard=20).yards_to_goal == 80
    assert PlayHeader(offense="GB", side="ATL", yard=20).yards_to_goal == 20
    assert PlayHeader(offense="GB", side=None, yard=50).yards_to_goal == 50
    assert parse_header(["3RD & GOAL", "ATL ON GB 2", "OT 02:10"]).distance is None


def _table():
    return pl.DataFrame({
        "game_id": ["2026_03_ATL_GB"] * 3, "play_id": [100, 200, 300], "week": [3, 3, 3],
        "posteam": ["GB", "GB", "ATL"], "defteam": ["ATL", "ATL", "GB"], "qtr": [4, 4, 4],
        "time": ["08:21", "07:40", "08:21"], "down": [2, 1, 2], "ydstogo": [7, 10, 7],
        "yrdln": ["ATL 15", "GB 30", "GB 40"], "yardline_100": [15, 70, 40],
        "play_type": ["pass", "run", "pass"], "yards_gained": [15.0, 3.0, 0.0], "epa": [2.5, -0.1, -0.5],
        "touchdown": [1, 0, 0], "complete_pass": [1, 0, 0], "desc": ["Love to Golden", "run", "incomplete"],
        "passer_player_name": ["J.Love", None, "M.Penix"], "receiver_player_name": ["M.Golden", None, "D.London"],
        "rusher_player_name": [None, "J.Jacobs", None], "sack": [0, 0, 0], "interception": [0, 0, 0],
        "air_yards": [15.0, None, 9.0], "pass_location": ["right", None, "left"],
    })


def test_match_play_finds_the_row(tmp_path):
    h = PlayHeader(quarter=4, clock="08:21", down=2, distance=7, offense="GB", side="ATL", yard=15)
    rec = match_play(h, _table(), root=tmp_path)
    assert rec.matched and rec.play_id == 100 and rec.defense == "ATL"
    assert rec.is_pass and rec.yards_to_goal == 15 and rec.touchdown
    assert rec.outcome == "Love to Golden, 15 yd, touchdown"
    assert rec.summary == "Q4 08:21  2nd & 7  GB on ATL 15"


def test_outcomes_read_naturally(tmp_path):
    run = match_play(PlayHeader(quarter=4, clock="07:40", down=1, distance=10, offense="GB", side="GB", yard=30), _table(), root=tmp_path)
    assert run.outcome == "Jacobs, 3 yd" and not run.is_pass
    miss = match_play(PlayHeader(quarter=4, clock="08:21", down=2, distance=7, offense="ATL", side="GB", yard=40), _table(), root=tmp_path)
    assert miss.outcome == "Penix to London, incomplete"


def test_unmatched_play_still_carries_the_yard_line(tmp_path):
    h = PlayHeader(quarter=1, clock="15:00", down=1, distance=10, offense="GB", side="GB", yard=25)
    rec = match_play(h, _table(), root=tmp_path)
    assert not rec.matched and rec.yards_to_goal == 75 and rec.outcome == ""
    assert match_play(PlayHeader(), _table()).notes == ["could not read the play header"]


def test_charted_look():
    rec = PlayRecord(PlayHeader(), matched=True, play_type="pass",
                     charted={"qb_location": "S", "n_defense_box": 5, "n_pass_rushers": 4})
    assert rec.charted_look == "Shotgun, 5 in the box, 4 rush"


def test_the_filter_chips_stand_in_for_the_address_bar():
    h = parse_header(["180 plays matching your criteria", "Season: 2026", "Season Type: REG", "Week: 3", "Game: CAR at CLE",
                      "Q1 14:56", "1ST & 10", "CAR ON CAR 29", "(14:56) C.Hubbard right tackle to CAR 29 for no gain (M.Graham)."])
    assert (h.season, h.week, h.game) == (2026, 3, ("CAR", "CLE"))
    assert (h.quarter, h.clock, h.down, h.distance, h.offense, h.side, h.yard) == (1, "14:56", 1, 10, "CAR", "CAR", 29)
    assert h.yards_to_goal == 71


def test_the_newest_season_on_disk_is_assumed_when_the_screen_does_not_say(tmp_path):
    from nfl_intel.playinfo import _TABLES, latest_season

    assert latest_season(tmp_path) is None
    _table().write_parquet(tmp_path / "play_by_play_2025.parquet")
    _table().write_parquet(tmp_path / "play_by_play_2026.parquet")
    assert latest_season(tmp_path) == 2026
    _TABLES.clear()
    h = PlayHeader(quarter=4, clock="08:21", down=2, distance=7, offense="GB", side="ATL", yard=15, game=("ATL", "GB"))
    rec = match_play(h, root=tmp_path)
    assert rec.matched and h.season == 2026 and rec.play_id == 100
    _TABLES.clear()


def test_the_newer_header_is_one_line_without_the_offense():
    h = parse_header(["Game: JAX at BAL", "Week: 9", "1st & 10", "2ND & 6 BAL 17", "Q 1 12:44", "15:00",
                      "(12:44) (Shotgun) T.Lawrence pass short left to J.Meyers."])
    assert (h.quarter, h.clock, h.down, h.distance, h.side, h.yard) == (1, "12:44", 2, 6, "BAL", 17)
    assert h.offense is None and h.placed and not h.complete and h.game == ("JAX", "BAL")
    mid = parse_header(["1ST & 10 50", "Q3 09:00"])
    assert (mid.side, mid.yard, mid.down) == (None, 50, 1)
    goal = parse_header(["3RD & GOAL JAX 2", "Q4 01:10"])
    assert (goal.down, goal.distance, goal.side, goal.yard) == (3, None, "JAX", 2)


def test_the_play_by_play_says_whose_ball_it_is(tmp_path):
    h = PlayHeader(quarter=4, clock="08:21", down=2, distance=7, side="ATL", yard=15, game=("ATL", "GB"), week=3)
    rec = match_play(h, _table(), root=tmp_path)
    assert rec.matched and rec.play_id == 100 and h.offense == "GB" and rec.yards_to_goal == 15
    # two plays at that clock and no yard line to choose by: nobody is named
    vague = PlayHeader(quarter=4, clock="08:21", down=2, distance=7, side="NYJ", yard=1, game=("ATL", "GB"), week=3)
    assert not match_play(vague, _table(), root=tmp_path).matched and vague.offense is None
    # and without the game on screen there is nothing to go on
    assert not match_play(PlayHeader(quarter=4, clock="08:21", side="ATL", yard=15), _table(), root=tmp_path).matched


def test_boxes_on_one_line_are_one_row():
    from dataclasses import dataclass

    from nfl_intel.playinfo import text_rows

    @dataclass
    class Box:
        text: str
        box: tuple

    rows = text_rows([Box("05:12", (120, 40, 180, 60)), Box("2ND & 5 BAL", (20, 10, 200, 30)), Box("Q 1", (70, 41, 110, 61)), Box("43", (210, 11, 240, 31))])
    assert rows == ["2ND & 5 BAL 43", "Q 1 05:12"]
    assert parse_header(rows).key == (1, "05:12", 2, 5, None, "BAL", 43)
