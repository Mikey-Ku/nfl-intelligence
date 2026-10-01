from nfl_intel.plays_check import _look, mark


def _play(**kw):
    base = {"play_type": "pass", "offense_look": "Shotgun, 3 x 1, 1 back", "defense_look": "Two high, 6+ in the box",
            "found": {"QB": 1, "REC": 5, "DL": 4, "DEF": 7}, "value": [{"t": 0.5}], "shotgun": True,
            "charting": {"qb_location": "S", "n_offense_backfield": 1, "n_defense_box": 6, "n_pass_rushers": 4}}
    base.update(kw)
    return base


def test_the_look_is_read_back_as_numbers():
    assert _look(_play()) == {"shotgun": True, "backs": 1, "box": 6, "rushers": 4}
    assert _look(_play(offense_look="Under centre, 2 x 1, empty", defense_look=""))["backs"] == 0
    assert _look(_play(offense_look="Not at the line yet"))["shotgun"] is None


def test_marks_against_the_charting():
    plays = [
        _play(),                                                                       # everything right
        _play(offense_look="Under centre, 2 x 2, 2 backs", defense_look="Single high, 8+ in the box",
              found={"REC": 4, "DL": 6, "DEF": 5}, value=None),                         # everything wrong, no quarterback
        _play(play_type="run", charting={}, shotgun=False, offense_look="Under centre, 1 x 1, 2 backs"),
        _play(play_type="punt"),
    ]
    c = mark(plays)
    assert c["plays"] == 3 and c["passes"] == 2
    assert c["quarterback named"] == 2 and c["passes valued"] == 1
    assert c["alignment marked"] == 3 and c["alignment right"] == 2       # the run falls back on the play-by-play's flag
    assert (c["backs marked"], c["backs right"], c["backs within one"]) == (2, 1, 2)
    assert (c["box marked"], c["box right"], c["box within one"]) == (2, 1, 1)
    assert (c["rush marked"], c["rush right"], c["rush within one"]) == (2, 1, 1)
