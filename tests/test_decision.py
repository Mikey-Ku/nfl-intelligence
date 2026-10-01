"""Grading the throw on a synthetic play: target found, throw time worked back, options ranked."""

import numpy as np
import pytest

from formations import LOS, film
from nfl_intel.analysis import AnalyzedTracker, PlayAnalyzer
from nfl_intel.decision import find_target, grade_throw, tracks, where
from test_analysis import FakeTracker, SeparationModel, _record


def _anchored(run=3.5):
    frames, snap_at = film(run=run)
    tracker = AnalyzedTracker(FakeTracker(frames), PlayAnalyzer(SeparationModel(), record=_record()))
    img = np.zeros((4, 4, 3), np.uint8)
    results = [tracker.process(img, f.frame, f.t) for f in frames]
    tracker.finish()
    return results, tracker.analyzer.snap_t


def test_paths_come_out_of_the_anchored_frames():
    results, snap = _anchored()
    paths = tracks(results, snap)
    assert {"QB", "R1", "R5", "D1"} <= set(paths) and paths["QB"]["role"] == "QB"
    r1 = paths["R1"]                                    # the widest receiver on the offense's left runs straight upfield
    at2 = where(r1, 2.0)
    assert at2[0] - LOS == pytest.approx(-1.0 + 6.5 * 2.0, abs=1.2) and at2[1] == pytest.approx(46.0, abs=0.5)
    assert where(r1, 30.0) is None


def test_target_found_by_where_the_ball_went_and_the_throw_worked_back():
    results, snap = _anchored()
    # 12 yards downfield to the right: only R5 (alone on that side) was ever there, about 2 seconds in
    grade = grade_throw(results, snap, LOS, SeparationModel(), air_yards=12.0, pass_location="right")
    assert grade.target == "R5" and grade.how == "by where the ball went"
    assert grade.arrived_at == pytest.approx(2.0, abs=0.3)
    assert 0.6 < grade.arrived_at - grade.thrown_at < 1.4          # the ball was in the air on the way there
    assert {o.label for o in grade.options} >= {"R1", "R5"} and 1 <= grade.rank <= len(grade.options)
    assert grade.chosen.label == "R5" and grade.left_on_the_field >= 0
    assert "option" in grade.sentence() and grade.as_dict()["target"] == "R5"


def test_two_receivers_who_fit_equally_are_not_guessed_between():
    results, snap = _anchored()
    paths = tracks(results, snap)
    qb_y = paths["QB"]["xy"][0, 1]
    # to the left there are two wide receivers running the same route at the same depth
    assert find_target(paths, LOS, qb_y, 12.0, "left") is None
    # unless one of them is known by name
    named = dict(paths)
    named["Golden"] = named.pop("R1")
    assert find_target(named, LOS, qb_y, 12.0, "left", name="Golden")[::2] == ("Golden", "by name")
    assert find_target(paths, LOS, qb_y, None, "left") is None


def test_no_quarterback_no_grade():
    results, snap = _anchored()
    for r in results:
        r.players = [p for p in r.players if p.role != "QB"]
    assert grade_throw(results, snap, LOS, SeparationModel(), air_yards=12.0, pass_location="right") is None


def test_the_grade_reaches_the_panel_and_the_play_log():
    frames, snap_at = film(run=7.0)
    record = _record()
    record.air_yards, record.pass_location, record.complete = 12.0, "right", False
    an = PlayAnalyzer(SeparationModel(), record=record)
    tracker = AnalyzedTracker(FakeTracker(frames), an)
    img = np.zeros((4, 4, 3), np.uint8)
    seen = []
    for f in frames:
        tracker.process(img, f.frame, f.t)
        seen.append(tracker.analysis.throw)
    assert seen[len(seen) // 2] is None and seen[-1] is not None     # graded once the ball must have arrived
    assert seen[-1].target == "R5"
    assert any(name == "Threw to" and value.startswith("R5, option") for name, value in an.rows())
    tracker.reset()
    log = an.plays[-1]
    assert log["throw"]["target"] == "R5" and log["throw"]["how"] == "by where the ball went" and log["shot"] == 0


def test_hold_times_and_firmness():
    from nfl_intel.decision import typical_hold

    assert typical_hold(-3.0) == 2.5 and typical_hold(7.0) == 2.3 and typical_hold(45.0) == 3.2
    results, snap = _anchored()
    grade = grade_throw(results, snap, LOS, SeparationModel(), air_yards=12.0, pass_location="right")
    assert grade.ranks[0] <= grade.rank <= grade.ranks[1]
    assert grade.firm == (grade.ranks[0] == grade.ranks[1])
    assert grade.as_dict()["firm"] == grade.firm and ("depending on the moment" in grade.sentence()) != grade.firm
