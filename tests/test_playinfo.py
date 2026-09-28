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
