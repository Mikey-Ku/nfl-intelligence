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
    assert backed.source == "end zone"                           # settled by the goal line, not left to the formation
    # the offense on its own 24 with its goal line 24 yards behind it, as the painted numbers give it
    deep = anchor_play(np.array([13.0, 12.8, 12.9, 9.5, 12.5, 13.1, 15.2, 15.0, 15.1, 15.3, 19.0, 21.0, 27.0, 28.0]), team, 76.0,
                       end_zone=(-10.0, -1), source="painted numbers")
    assert deep.source == "painted numbers" and deep.direction == 1 and deep.los_rel == pytest.approx(14.0) and deep.los_abs == 34.0


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
    # a back four yards deep behind the ball looks like a shotgun quarterback, unless we know better
    heavy = [("RB", -4.0, BALL), ("RB", -6.5, BALL)] + [p for p in SHOTGUN[2:]]
    s, xy = slots(heavy)
    assert assign_roles(s, xy, 0, LOS)[0] == "QB"
    told = assign_roles(s, xy, 0, LOS, shotgun=False)
    named = [i for i, r in told.items() if r == "QB"]
    assert told[0] == "REC" and len(named) == 1                   # the back stays a back
    assert abs(xy[named[0], 1] - BALL) <= 1.7 and xy[named[0], 0] > LOS - 2.0      # under centre: a man on the ball
    offense, _ = describe_look(xy, [told[sl.id] for sl in s], LOS, shotgun=False)
    assert offense.startswith("Under centre") and offense.endswith("2 backs")
    few, fxy = slots(heavy, drop=(0, 1, 7, 8))
    roles = assign_roles(few, fxy, 0, LOS, shotgun=False)
    assert describe_look(fxy, [roles[sl.id] for sl in few], LOS, shotgun=False)[0] == "Under centre, 2 of 5 receivers found"


def test_the_charted_hash_mark_picks_the_quarterback_from_the_back_beside_him():
    # no linemen found at all, quarterback on the right hash, the back a yard and a half inside him
    gun = [("QB", -5.0, 23.7), ("RB", -4.6, 26.0), ("WR", -1.0, 45.0), ("WR", -1.2, 38.0), ("WR", -1.0, 6.0), ("TE", -1.5, 18.5)] + SHOTGUN[15:]
    s, xy = slots(gun)
    assert assign_roles(s, xy, 0, LOS)[1] == "QB"                 # the crowd is gone, so the guess lands on the back
    told = assign_roles(s, xy, 0, LOS, shotgun=True, hash_mark="R")
    assert told[0] == "QB" and told[1] == "REC"
    under = [("QB", -1.6, 29.6), ("RB", -7.0, 29.5)] + gun[2:]
    s, xy = slots(under)
    assert assign_roles(s, xy, 0, LOS, shotgun=False, hash_mark="L")[0] == "QB"


def test_what_players_do_after_the_snap_corrects_their_roles():
    from nfl_intel.roles import update_roles

    roles = {1: "DEF", 2: "DL", 3: "OL", 4: "OL", 5: "DL", 6: "REC"}
    now = {1: np.array([LOS - 1.5, 20.0]),      # a linebacker who came through: rushing
           2: np.array([LOS + 4.0, 24.0]),      # walked up on the line, then dropped: covering
           3: np.array([LOS + 3.0, 31.0]),      # "lineman" running a route: he was a tight end
           4: np.array([LOS - 2.5, 26.0]),      # a lineman pass blocking
           5: np.array([LOS - 1.0, 27.0]),
           6: np.array([LOS + 8.0, 40.0])}
    changed = update_roles(roles, now, LOS, since_snap=1.0)
    assert roles == {1: "DL", 2: "DEF", 3: "REC", 4: "OL", 5: "DL", 6: "REC"} and sorted(changed) == [1, 2, 3]
    run = {3: "OL"}
    assert update_roles(run, now, LOS, since_snap=1.0, passing=False) == [] and run[3] == "OL"    # on a run linemen do go downfield


def test_roles_are_marked_against_tracked_plays():
    from nfl_intel.bdb import Play
    from nfl_intel.roles_check import score_roles

    def play(pid, qb_y, shotgun):
        # passer, a back beside him, three route runners, five coverage men; no linemen, as in the Big Data Bowl set
        depth = -5.0 if shotgun else -1.6
        start = np.array([[LOS + depth, qb_y], [LOS - 4.6, qb_y + 2.4], [LOS - 1.0, 45.0], [LOS - 1.2, 8.0], [LOS - 1.5, qb_y + 6.0],
                          [LOS + 5.0, 45.0], [LOS + 6.0, 8.0], [LOS + 0.9, qb_y + 6.5], [LOS + 12.0, 20.0], [LOS + 12.0, 33.0]])
        go = np.array([[-0.2, 0], [0.3, -0.3], [0.6, 0], [0.6, 0], [0.5, 0], [0.4, 0], [0.4, 0], [0.4, 0], [0.3, 0], [0.3, 0]])
        xy = np.stack([start + go * t for t in range(20)])
        return Play(game_id=1, play_id=pid, week=1, los=LOS, xy=xy, offense=np.array([True] * 5 + [False] * 5),
                    roles=["QB"] + ["REC"] * 4 + ["DEF"] * 5, names=[f"p{i}" for i in range(10)], positions=["QB", "RB", "WR", "WR", "TE"] + ["CB", "CB", "OLB", "FS", "SS"],
                    qb=0, target=2, ball_land=(LOS + 10.0, 45.0))

    plays = [play(1, 23.7, True), play(2, 29.6, True), play(3, 26.7, False)]
    hints = {(1, 1): (True, "R"), (1, 2): (True, "L"), (1, 3): (False, "M")}
    told = score_roles(plays, hints)
    assert told.plays == 3 and told.share("qb right", "plays") == 1.0
    assert told.counts["gun"] == 2 and told.counts["under"] == 1
    assert told.share("rec right", "rec") == 1.0
    assert told.share("def right", "def") == pytest.approx(12 / 15)          # the man on the line looks like a rusher...
    assert told.share("def right later", "def") == 1.0                       # ...until he drops into coverage
    alone = score_roles(plays, hints, use_hints=False)
    assert alone.counts["gun"] == 2 and alone.share("qb right under", "under") == 0.0     # nobody to stand behind
    assert len(told.lines()) == 7


def test_the_on_field_list_caps_the_receivers():
    # goal line: seven linemen, two tight ends and a back; a guard and a tackle look like receivers to the box
    jumbo = [("QB", -2.2, BALL), ("RB", -6.5, BALL)] + [("OL", -1.4, BALL + k * 1.6) for k in (-3, -2, -1, 0, 1, 2, 3)] \
        + [("TE", -1.5, BALL + 6.4), ("TE", -1.5, BALL - 6.4)] + SHOTGUN[11:]
    s, xy = slots(jumbo)
    alone = assign_roles(s, xy, 0, LOS, shotgun=False)
    assert sum(r == "OL" for r in alone.values()) == 5 and sum(r == "REC" for r in alone.values()) == 5
    told = assign_roles(s, xy, 0, LOS, shotgun=False, linemen=7, eligible=3)
    assert [told[i] for i in range(2, 9)] == ["OL"] * 7
    assert told[0] == "QB" and [told[i] for i in (1, 9, 10)] == ["REC"] * 3
    # linemen hidden in the pile: the cap still holds with what was found
    s, xy = slots(jumbo, drop=(4, 5, 6))
    told = assign_roles(s, xy, 0, LOS, shotgun=False, linemen=7, eligible=3)
    assert sum(r == "REC" for r in told.values()) == 3


def test_painted_numbers_anchor_a_play_without_the_play_by_play():
    # offense (team 0) going toward -x from the mapping's 40; the numbers put a goal line at 25, 15 yards on
    x = np.array([41.4, 41.3, 41.5, 44.5, 44.6, 41.0, 39.2, 39.1, 39.0, 36.0, 35.0, 28.0, 28.5, 37.0])
    team = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    a = anchor_play(x, team, None, end_zone=(25.0, -1), source="painted numbers")
    assert a.source == "painted numbers and formation" and a.direction == -1 and a.offense_team == 0
    assert a.los_abs == pytest.approx(95.0, abs=1.0) and a.los_rel == pytest.approx(40.0, abs=1.0)
    # the same picture with the goal line behind the offense: it has 85 yards to go
    b = anchor_play(x, team, None, end_zone=(55.0, 1), source="painted numbers")
    assert b.direction == -1 and b.los_abs == pytest.approx(25.0, abs=1.0)
    assert anchor_play(x, team, None) is None                     # no goal line, no play-by-play: nothing to go on
    # with the play-by-play, a goal line far out of the picture still pins the grid at midfield
    c = anchor_play(x, team, 45.0, end_zone=(-5.0, -1), source="painted numbers")
    assert c.source == "painted numbers" and c.los_rel == pytest.approx(40.0) and c.los_abs == 65.0


def test_an_offense_in_its_huddle_does_not_decide_the_direction():
    x = np.array([46.0, 46.5, 47.0, 46.2, 47.3, 45.8, 46.8, 39.2, 39.1, 39.0, 36.0, 35.0, 38.0, 38.5, 37.0])
    team = np.array([0] * 7 + [1] * 8)
    assert anchor_play(x, team, None, end_zone=(25.0, -1)) is None


def test_painted_numbers_are_parsed_and_vote_for_a_goal_line():
    from nfl_intel.fieldnumbers import FlatField, PaintedNumber, goal_line, parse_number, read_numbers

    assert parse_number("20-") == (20, 1) and parse_number("-10") == (10, -1) and parse_number("◄ 3O") == (30, -1)
    assert parse_number("50") == (50, 0) and parse_number("-40-") == (40, 0)
    assert parse_number("HEN BAY") is None and parse_number("60") is None and parse_number("12") is None

    def n(value, x, y, goal):
        return PaintedNumber(value, x, y, goal, 1.0)

    assert goal_line([n(20, 5.0, 13.0, 1), n(10, 15.0, 13.0, 1)]) == (25.0, 1)
    assert goal_line([n(10, 10.0, 40.0, 0)]) is None                          # one number, no arrow: either side of the field
    assert goal_line([n(30, 0.0, 13.0, 0), n(40, 10.0, 40.0, 0)]) == (-30.0, -1)     # two numbers agree without arrows
    assert goal_line([n(40, 10.0, 13.0, -1)]) == (-30.0, -1)
    assert goal_line([n(50, 20.0, 13.0, 0)]) == (70.0, 1)
    assert goal_line([]) is None

    class Text:
        def __init__(self, text, box, confidence=1.0):
            self.text, self.box, self.confidence = text, box, confidence

    scale = 24.0
    flat = FlatField(np.zeros((int(53.33 * scale), int(30 * scale), 3), np.uint8), scale, lo=0.0, hi=30.0, mirrored=False)
    seen = [Text("20-", (2.6 * scale, 12 * scale, 8.0 * scale, 14 * scale)),           # far row, read from the y = 0 sideline
            Text("-10", (12.0 * scale, 39 * scale, 17.4 * scale, 41 * scale)),         # near row, read from the other
            Text("GREEN BAY", (20 * scale, 13 * scale, 25 * scale, 40 * scale)),
            Text("30", (4 * scale, 25 * scale, 8 * scale, 27 * scale))]                # a jersey in the middle of the field
    got = read_numbers(flat, reader=lambda img: seen)
    assert [(g.value, g.x, g.goal) for g in got] == [(20, 5.0, 1), (10, 15.0, 1)]
    assert goal_line(got) == (25.0, 1)


def test_the_flat_field_is_never_a_mirror_image():
    from nfl_intel.fieldnumbers import flatten

    # a picture of 30 yards of field at 10 px a yard, with an "L" of paint whose foot points toward larger x in the picture
    img = np.full((534, 300, 3), (60, 120, 60), np.uint8)
    cv2.rectangle(img, (100, 100), (110, 200), (255, 255, 255), -1)
    cv2.rectangle(img, (100, 190), (160, 200), (255, 255, 255), -1)
    rect = (0, 0, 300, 534)
    straight = np.diag([0.1, 0.1, 1.0])
    turned = np.array([[-0.1, 0.0, 30.0], [0.0, -0.1, 53.4], [0.0, 0.0, 1.0]])       # the camera on the other sideline
    backwards = np.array([[-0.1, 0.0, 30.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])      # x counted the other way: a mirror if drawn naively
    for H, mirrored in ((straight, False), (turned, False), (backwards, True)):
        flat = flatten(img, H, rect, scale=10.0)
        assert flat.mirrored == mirrored and flat.img.shape[1] == 300
        paint = np.argwhere(flat.img[..., 0] > 200)
        stem_x = np.median(paint[:, 1])                       # most of the paint is the stem
        foot_x = paint[:, 1].max() if H is not turned else paint[:, 1].min()
        # the foot stays on the side of the stem it was painted on (a half turn swaps both, a mirror would swap one)
        assert (foot_x > stem_x) == (H is not turned)
        x_at_stem = flat.field_x(stem_x)
        assert x_at_stem == pytest.approx(10.5 if H is straight else 19.5, abs=0.6)
    boxes = np.array([[90.0, 90.0, 170.0, 210.0]])
    assert (flatten(img, straight, rect, boxes, scale=10.0).img[..., 0] > 200).sum() == 0      # players are painted out


def test_the_huddle_says_who_has_the_ball():
    from nfl_intel.anchor import find_huddle

    rng = np.random.default_rng(3)
    huddle = np.column_stack([47.0 + rng.normal(0, 0.8, 8), 26.0 + rng.normal(0, 0.8, 8)])
    defense = np.column_stack([np.linspace(30, 39, 9), np.linspace(10, 44, 9)])
    xy = np.vstack([huddle, defense])
    team = np.array([0] * 8 + [1] * 9)
    assert find_huddle(xy, team)[0] == 0 and find_huddle(xy, team)[1] == pytest.approx(47.0, abs=0.8)
    lined_up = np.column_stack([np.full(8, 41.3), np.linspace(18, 34, 8)])      # a line is wide, not a knot
    assert find_huddle(np.vstack([lined_up, defense]), team) is None

    # goal line, no play-by-play: both sides are bunched, so depth says nothing, but the huddle was seen
    x = np.array([41.4, 41.3, 41.5, 41.2, 41.6, 43.0, 47.5, 39.2, 39.1, 39.0, 38.9, 39.3, 37.0, 36.5, 36.0])
    t = np.array([0] * 7 + [1] * 8)
    assert anchor_play(x, t, None, end_zone=(33.0, -1)) is None
    a = anchor_play(x, t, None, end_zone=(33.0, -1), source="painted numbers", huddle=(0, 47.0))
    assert a.direction == -1 and a.offense_team == 0 and a.source == "painted numbers and huddle"
    assert a.los_abs == pytest.approx(103.0, abs=1.0)
    assert anchor_play(x, t, None, end_zone=(33.0, -1), huddle=(1, 47.0)) is None      # a "huddle" on the wrong side of the ball


def test_a_defender_hidden_behind_his_receiver_is_carried_along():
    tr = FieldTracker()
    rec_id = def_id = None
    for f in range(100):
        t = f / 20.0
        rec = np.array([20.0 + 6.0 * t, 10.0])
        dfn = rec + np.array([1.0, 0.6])                               # tight coverage, a yard off
        far = np.array([60.0, 40.0])
        merged = 30 <= f < 75                                          # one box for the two of them, for over two seconds
        pts = [rec, far] if merged else [rec, dfn, far]
        teams = [0, 1] if merged else [0, 1, 1]
        got = tr.update(t, np.array(pts), teams, list(range(len(pts))))
        if f == 20:
            rec_id, def_id = got[0].id, got[1].id
        if f == 60:
            hidden = next(s for s in tr.slots if s.id == def_id)       # still there, still a yard off his man
            assert hidden.partner == rec_id
            assert hidden.xy == pytest.approx(rec + np.array([1.0, 0.6]), abs=0.4)
        if f >= 80:
            assert (got[0].id, got[1].id) == (rec_id, def_id)          # and he is himself again when they part
    assert len(tr.players) == 3


def test_a_man_who_simply_leaves_is_not_kept():
    tr = FieldTracker()
    for f in range(60):
        t = f / 20.0
        pts = [np.array([20.0, 10.0]), np.array([40.0, 30.0])] if f < 20 else [np.array([20.0, 10.0])]
        tr.update(t, np.array(pts), [0, 1][: len(pts)], list(range(len(pts))))
    assert len(tr.slots) == 1                                          # nobody near him to hide behind: gone after a second


def test_officials_beyond_eleven_a_side_are_left_out():
    # the referee twelve yards behind the passer, the back judge twenty-five yards deep, a line judge on the sideline
    crowd = SHOTGUN + [("REF", -12.5, 26.0), ("BJ", 25.0, 27.0), ("LJ", 0.0, 0.5)]
    s, xy = slots(crowd)
    off, dfn = sides(s, xy, 0, LOS)
    assert len(off) == 11 and len(dfn) == 11
    assert 22 not in off and 23 not in dfn and 24 not in off + dfn
    roles = assign_roles(s, xy, 0, LOS)
    assert roles[22] == "OFF" and roles[23] == "OFF" and roles[24] == "OFF"
    assert sum(r in ("DEF", "DL") for r in roles.values()) == 11 and roles[0] == "QB"


def test_an_unbroken_image_track_keeps_its_slot_when_a_stale_slot_is_nearer():
    """Two slots end up beside one man (the second from a stray box). Without the image
    track to say which is his, the detection goes to whichever is nearer that frame."""
    tr = FieldTracker()
    for f in range(6):
        t = f / 20
        tr.update(t, np.array([[30.0, 20.0], [30.6, 20.0]]), [0, 0], [7, 8])       # both seen, and each has its own track
    mine = tr.slots[0].id
    for f in range(6, 30):
        t = f / 20
        got = tr.update(t, np.array([[30.45, 20.0]]), [0], [7])                    # one man now, nearer the stale slot
        assert got[0].id == mine
    assert FieldTracker(same_track=0.0)                                            # (the switch exists)


def test_an_unconfirmed_box_starts_a_player_only_clear_of_everyone():
    tr = FieldTracker()
    for f in range(5):
        tr.update(f / 20, np.array([[30.0, 20.0], [40.0, 25.0]]), [0, 1], [1, 2])
    n = len(tr.slots)
    got = tr.update(5 / 20, np.array([[30.0, 20.0], [40.0, 25.0], [30.3, 20.1], [50.0, 10.0]]), [0, 1, 0, 1], [1, 2, None, None],
                    confirmed=[True, True, False, False])
    assert got[2] is None and len(tr.slots) == n + 1                             # a second box on a man starts nothing
    assert got[3] is not None and got[3].born == 5 / 20                           # one well clear of everybody is a new man
    assert tr.update(5.5 / 20, np.array([[30.0, 20.0], [40.0, 25.0], [31.2, 20.0]]), [0, 1, 0], [1, 2, None],
                     confirmed=[True, True, False])[2] is None                    # beside a known player: nothing either
    tr.update(6 / 20, np.array([[40.0, 25.0], [30.1, 20.0]]), [1, 0], [2, None], confirmed=[True, False])
    assert tr.slots[0].seen == 6 / 20                                             # ...but it does show where a known man is


def test_a_second_track_on_the_same_man_takes_over_his_number():
    tr = FieldTracker()
    for f in range(8):
        tr.update(f / 20, np.array([[30.0 + 0.1 * f, 20.0]]), [0], [14])
    first = tr.slots[0].id
    tr.slots[0].role = "REC"
    got = tr.update(8 / 20, np.array([[30.8, 20.0], [31.0, 20.2]]), [0, 0], [14, 37])      # the image tracker starts a second track on him
    assert got[1].id != first and got[1].twin == first
    got = tr.update(9 / 20, np.array([[31.1, 20.2]]), [0], [37])                           # and drops the first
    assert got[0].id == first and got[0].role == "REC"
    for f in range(10, 20):
        assert tr.update(f / 20, np.array([[30.0 + 0.1 * f, 20.2]]), [0], [37])[0].id == first
    assert len(tr.players) == 1
