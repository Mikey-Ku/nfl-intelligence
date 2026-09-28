"""Anchoring, field-space tracking, the snap and role assignment, on synthetic formations."""

import cv2
import numpy as np
import pytest

from formations import BALL, LOS, SHOTGUN, slots
from nfl_intel.anchor import Anchor, anchor_play, find_end_zone, split_teams
from nfl_intel.field import FIELD_WIDTH
from nfl_intel.fieldtrack import FieldTracker, Slot, SnapDetector
from nfl_intel.roles import assign_roles, ball_spot, describe_look, sides


# ---------------------------------------------------------------- anchor

def test_split_teams_finds_the_line_between_them():
    x = np.array([38.0, 39.0, 40.0, 40.5, 41.0, 43.0, 43.5, 44.0, 47.0, 55.0])
    team = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    cut, low, share = split_teams(x, team)
    assert cut == pytest.approx(42.0) and low == 1 and share == 1.0
    assert split_teams(x[:4], team[:4]) is None                  # one team only


def test_anchor_by_end_zone_and_by_formation_agree():
    # offense (team 0) moving toward -x in the mapping; the goal line it attacks is 15 yards away at x = 25
    x = np.array([41.4, 41.3, 41.5, 44.5, 44.6, 41.0, 39.2, 39.1, 39.0, 36.0, 35.0, 28.0, 28.5, 37.0])
    team = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    by_zone = anchor_play(x, team, 15.0, end_zone=(25.0, -1))
    by_look = anchor_play(x, team, 15.0)
    for a, source in ((by_zone, "end zone"), (by_look, "formation")):
        assert a.source == source and a.direction == -1 and a.offense_team == 0
        assert a.los_rel == pytest.approx(40.0) and a.los_abs == 95.0
    # backed up on its own 15 instead: the same end zone is behind the offense
    backed = anchor_play(30.0 - (x - 40.0), team, 85.0, end_zone=(15.0, -1))
    assert backed.direction == 1 and backed.los_abs == 25.0 and backed.los_rel == pytest.approx(30.0)


def test_anchor_maps_to_the_real_field_without_mirroring():
    a = Anchor(los_rel=40.0, direction=-1, los_abs=95.0, offense_team=0, source="end zone", split_accuracy=1.0)
    xy = a.to_absolute([[40.0, 10.0], [30.0, 50.0]])
    assert xy[0] == pytest.approx([95.0, FIELD_WIDTH - 10.0])
    assert xy[1] == pytest.approx([105.0, FIELD_WIDTH - 50.0])
    h = a.matrix() @ np.array([30.0, 50.0, 1.0])
    assert h[:2] == pytest.approx(xy[1])
    assert np.linalg.det(a.matrix()[:2, :2]) > 0                 # a turn, not a mirror
    assert a.to_relative_x(102.0) == pytest.approx(33.0)
    assert [a.yard_name(x) for x in (45.0, 40.0, 30.0, 25.0, 20.0)] == ["20", "15", "5", "G", ""]


def test_anchor_refuses_a_muddled_split():
    x = np.array([38.0, 44.0, 40.0, 47.0, 41.0, 43.0, 39.5, 44.0, 42.0, 55.0])
    team = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 0])
    assert anchor_play(x, team, 15.0) is None


def test_find_end_zone_sees_the_painted_end_zone():
    scale = 12.0                                                  # a top-down picture, 12 px per yard, x from 0 to 50
    grass = (60, 140, 60)
    img = np.full((int(FIELD_WIDTH * scale), int(50 * scale), 3), grass, dtype=np.uint8)
    for x in range(10, 51, 5):
        cv2.line(img, (int(x * scale), 0), (int(x * scale), img.shape[0] - 1), (240, 240, 240), 2)
    cv2.rectangle(img, (0, 0), (int(10 * scale), img.shape[0] - 1), (40, 190, 230), -1)      # the end zone: the first 10 yards
    H = np.diag([1 / scale, 1 / scale, 1.0])
    grass_lab = cv2.cvtColor(np.uint8([[grass]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)
    found = find_end_zone(img, H, (0, 0, img.shape[1], img.shape[0]), grass_lab)
    assert found == (10.0, -1)
    img[:] = grass
    assert find_end_zone(img, H, (0, 0, img.shape[1], img.shape[0]), grass_lab) is None


# ---------------------------------------------------------------- tracking in yards

def test_field_tracker_keeps_identity_through_a_gap_and_a_new_track_id():
    tr = FieldTracker()
    ids = []
    for f in range(60):
        t = f / 20.0
        a = np.array([20.0 + 5.0 * t, 10.0])                      # runs at 5 yd/s
        b = np.array([30.0, 12.0])
        pts, teams, tids = [a, b], [0, 1], [7 if f < 30 else 99, 8]
        if 20 <= f < 26:                                          # the detector loses the runner for 0.3 s
            pts, teams, tids = [b], [1], [8]
        got = tr.update(t, np.array(pts), teams, tids)
        ids.append(got[0].id if len(pts) == 2 else None)
    assert {i for i in ids if i is not None} == {ids[0]}
    runner = next(s for s in tr.players if s.id == ids[0])
    assert runner.v[0] == pytest.approx(5.0, abs=0.4) and abs(runner.v[1]) < 0.3
    assert runner.track_id == 99 and len(tr.players) == 2


def test_field_tracker_does_not_swap_teams_when_players_cross():
    tr = FieldTracker()
    first = None
    for f in range(40):
        t = f / 20.0
        a = np.array([20.0 + 4.0 * t, 10.0 + 0.2])
        b = np.array([28.0 - 4.0 * t, 10.0 - 0.2])
        got = tr.update(t, np.array([a, b]), [0, 1], [1, 2])
        first = first or (got[0].id, got[1].id)
        assert (got[0].id, got[1].id) == first


def _still_then_go(detector, fps=20.0, snap_at=2.0, n=14):
    rng = np.random.default_rng(1)
    tr = FieldTracker()
    base = np.column_stack([np.linspace(30, 45, n), np.linspace(5, 48, n)])
    found = None
    for f in range(int(4.0 * fps)):
        t = f / fps
        s = max(0.0, t - snap_at)
        pts = base + np.array([4.5 * s, 0.0]) + rng.normal(0, 0.04, base.shape)
        tr.update(t, pts, [i % 2 for i in range(n)], list(range(n)))
        found = detector.update(t, tr.players)
        if found is not None:
            return found, t
    return None, None


def test_snap_detector_walks_back_to_the_snap():
    found, seen = _still_then_go(SnapDetector())
    assert found == pytest.approx(2.0, abs=0.2)
    assert seen - found < 0.8                                     # and doesn't take long to be sure


def test_snap_detector_ignores_one_man_in_motion():
    tr, det = FieldTracker(), SnapDetector()
    base = np.column_stack([np.linspace(30, 45, 14), np.linspace(5, 48, 14)])
    for f in range(60):
        t = f / 20.0
        pts = base.copy()
        pts[0, 1] += 5.0 * t                                      # a receiver in motion across the formation
        tr.update(t, pts, [i % 2 for i in range(14)], list(range(14)))
        assert det.update(t, tr.players) is None


# ---------------------------------------------------------------- roles

def _roles(drop=(), teams=None, formation=SHOTGUN):
    s, xy = slots(formation, drop, teams)
    roles = assign_roles(s, xy, 0, LOS)
    return {sl.id: roles[sl.id] for sl in s}, s, xy


def test_roles_in_a_full_shotgun_formation():
    roles, _, _ = _roles()
    assert roles[0] == "QB" and roles[1] == "REC"
    assert [roles[i] for i in range(2, 7)] == ["OL"] * 5
    assert [roles[i] for i in (7, 8, 9, 10)] == ["REC"] * 4
    assert [roles[i] for i in (11, 12, 13, 14)] == ["DL"] * 4
    assert [roles[i] for i in range(15, 22)] == ["DEF"] * 7


def test_roles_survive_linemen_lost_in_the_pile_and_a_wrong_jersey():
    # three linemen and a tackle undetected; the slot corner was put in the offense's colours
    roles, _, _ = _roles(drop=(3, 4, 5, 12), teams={18: 0})
    assert roles[0] == "QB" and roles[1] == "REC"
    assert roles[2] == "OL" and roles[6] == "OL"
    assert roles[18] == "DEF"                                     # he stands on the defense's side of the ball
    assert sum(r == "REC" for r in roles.values()) == 5


def test_quarterback_under_centre_with_backs_behind_him():
    i_form = [("QB", -2.3, BALL), ("RB", -7.0, BALL), ("RB", -4.5, BALL)] + [p for p in SHOTGUN[2:]]
    roles, _, _ = _roles(formation=i_form)
    assert roles[0] == "QB" and roles[1] == "REC" and roles[2] == "REC"


def test_officials_and_the_sideline_are_left_out():
    extra = SHOTGUN + [("REF", -12.0, 20.0), ("LJ", 0.0, 0.3), ("BJ", 38.0, 26.0)]
    roles, s, xy = _roles(formation=extra)
    assert roles[23] == "OFF" and roles[24] == "OFF"
    off, dfn = sides(s, xy, 0, LOS)
    assert 23 not in [s[i].id for i in off + dfn] and 24 not in [s[i].id for i in off + dfn]


def test_too_few_players_names_nobody():
    roles, _, _ = _roles(drop=tuple(range(3, 22)))
    assert set(roles.values()) == {"OFF"}


def test_ball_spot_stays_between_the_hashes():
    s, xy = slots()
    off, dfn = sides(s, xy, 0, LOS)
    assert ball_spot(xy, off, dfn, LOS) == pytest.approx(BALL, abs=1.5)
    wide = xy.copy()
    wide[:, 1] = np.clip(wide[:, 1] + 14.0, 0, FIELD_WIDTH)
    assert ball_spot(wide, off, dfn, LOS) == pytest.approx(29.75)


def test_describe_look_reads_like_a_broadcast():
    roles, s, xy = _roles()
    offense, defense = describe_look(xy, [roles[sl.id] for sl in s], LOS)
    assert offense == "Shotgun, 3 x 1, 1 back"
    assert defense.startswith("Two high") and "in the box" in defense
    assert describe_look(xy[:2], [None, None], LOS) == ("", "")


def test_the_play_by_play_settles_where_the_quarterback_is():
    # a back three yards deep behind the ball looks like a shotgun quarterback, unless we know better
    heavy = [("RB", -4.0, BALL), ("RB", -6.5, BALL)] + [p for p in SHOTGUN[2:]]
    s, xy = slots(heavy)
    assert assign_roles(s, xy, 0, LOS)[0] == "QB"
    told = assign_roles(s, xy, 0, LOS, shotgun=False)
    assert "QB" not in told.values() and told[0] == "REC"
    offense, _ = describe_look(xy, [told[sl.id] for sl in s], LOS, shotgun=False)
    assert offense.startswith("Under centre")
    few, fxy = slots(heavy, drop=(0, 1, 7, 8))
    roles = assign_roles(few, fxy, 0, LOS, shotgun=False)
    assert describe_look(fxy, [roles[sl.id] for sl in few], LOS, shotgun=False)[0] == "Under centre, 2 of 5 receivers found"
