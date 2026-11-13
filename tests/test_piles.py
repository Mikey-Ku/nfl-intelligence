from types import SimpleNamespace

import numpy as np

from nfl_intel.analysis import PILE_JOIN, PILE_LEAVE, PlayAnalyzer
from nfl_intel.draw import annotate, pile_groups, pile_outlines
from nfl_intel.evaluate import evaluate
from nfl_intel.fieldtrack import Slot, chains, find_piles
from nfl_intel.schema import FrameResult, TrackedPlayer, read_csv, write_csv


def test_a_pile_is_four_or_more_from_both_teams_close_together_and_slow():
    scrum = np.array([(50.0, 26.0), (50.8, 26.5), (51.5, 25.8), (50.2, 27.6), (70.0, 10.0)])
    still = np.zeros(5)
    assert find_piles(scrum, still, [0, 1, 0, 1, 1]).tolist() == [True, True, True, True, False]
    assert not find_piles(scrum, still, [0, 0, 0, 0, 1]).any()                # receivers bunched at the snap are one team
    assert not find_piles(scrum, np.full(5, 6.0), [0, 1, 0, 1, 1]).any()      # a group running together is not a pile
    assert not find_piles(scrum[:3], still[:3], [0, 1, 0]).any()              # three is a block and a tackle, not a pile
    assert chains(np.array([(0.0, 0.0), (1.5, 0.0), (3.0, 0.0), (9.0, 0.0)]), 2.0).tolist() == [0, 0, 0, 3]


def _man(k, x, y, role, slot=None):
    return TrackedPlayer(k, 0 if role in ("QB", "REC", "OL") else 1, (10.0 * x, 100.0, 10.0 * x + 8.0, 130.0), 0.9, (10.0 * x + 4.0, 130.0), (x, y), role=role, slot=slot)


def test_players_join_a_pile_after_a_moment_in_it_and_leave_after_a_moment_clear():
    slots = [Slot(k, np.zeros(2), np.zeros(2), None, 0.0, 0.0, 0.0) for k in range(6)]
    me = SimpleNamespace(snap_t=10.0, tracker=SimpleNamespace(slots=slots), _pile={})

    def frame(t, spread):
        men = [_man(0, 44.0, 26.0, "QB"), _man(1, 50.0, 26.0, "OL"), _man(2, 50.8 + spread, 26.4, "DL"), _man(3, 51.4 + 2 * spread, 25.7, "OL"),
               _man(4, 50.3, 27.5 + 3 * spread, "DL"), _man(5, 50.5, 26.8, "QB")]
        out = SimpleNamespace(players=men, slot_ids=list(range(6)), unseen=[])
        PlayAnalyzer._mark_piles(me, out, t)
        return [p.pile for p in men]

    assert not any(frame(10.2, 0.0))                                          # too soon after the snap to call anything a pile
    for k in range(PILE_JOIN - 1):
        assert not any(frame(11.0 + 0.04 * k, 0.0))                           # in one, but only just
    assert frame(11.5, 0.0) == [False, True, True, True, True, False]         # the linemen are in; a quarterback is himself anywhere
    for k in range(PILE_LEAVE - 1):
        assert frame(12.0 + 0.04 * k, 3.0)[1:5] == [True] * 4                 # it has broken up, but only just
    assert not any(frame(12.5, 3.0))


def test_a_player_in_a_pile_is_found_but_not_named_when_scoring():
    def run(frame, slot, pile):
        return FrameResult(frame, frame / 10, (100, 100), [TrackedPlayer(1, 0, (0, 0, 1, 1), 0.9, (0.0, 0.0), (50.0, 20.0), slot=slot, pile=pile),
                                                           TrackedPlayer(2, 1, (0, 0, 1, 1), 0.9, (0.0, 0.0), (60.0, 30.0), slot=9)])

    truth = [FrameResult(k, k / 10, (100, 100), [TrackedPlayer(1, None, (0, 0, 1, 1), 1.0, (0.0, 0.0), (50.0, 20.0)),
                                                 TrackedPlayer(2, None, (0, 0, 1, 1), 1.0, (0.0, 0.0), (60.0, 30.0))]) for k in range(4)]
    named = evaluate([run(0, 5, False), run(1, 6, False), run(2, 5, False), run(3, 5, False)], truth)
    assert named.id_switches == 2 and named.named == 1.0
    piled = evaluate([run(0, 5, False), run(1, 6, True), run(2, 5, False), run(3, 5, False)], truth)      # the frame he was called 6, he was in a pile
    assert piled.id_switches == 0 and piled.recall == 1.0 and piled.named == 7 / 8
    assert "among the 88% named" in str(piled) and "named" not in str(named)


def test_the_pile_flag_is_saved_with_the_tracking(tmp_path):
    res = FrameResult(3, 0.1, (100, 100), [_man(1, 50.0, 26.0, "OL", slot=4), _man(2, 51.0, 26.0, "DL", slot=5)], anchored=True)
    res.players[0].pile = True
    write_csv(tmp_path / "run.csv", [res])
    back = read_csv(tmp_path / "run.csv")[0].players
    assert [p.pile for p in back] == [True, False] and back[0].slot == 4       # the tracker's own guess is still there to look at


def test_a_pile_is_drawn_as_one_outline_with_nobody_in_it_named():
    men = [_man(k, 50.0 + 0.8 * k, 26.0, "OL" if k % 2 else "DL", slot=k) for k in range(4)] + [_man(9, 70.0, 10.0, "REC", slot=9)]
    for p in men[:4]:
        p.pile = True
    assert [len(g) for g in pile_groups(men)] == [4]
    (rect, inside), = pile_outlines(men)
    assert len(inside) == 4 and rect[0] < 500.0 and rect[2] > 532.0           # one rectangle round all four boxes
    men[0].pile, men[1].pile, men[2].pile = False, False, False
    assert pile_groups(men) == [] and pile_outlines(men) == []                # one man left counted in a pile is a mark, not an area
    for p in men[:4]:
        p.pile = True
    frame = np.zeros((300, 800, 3), np.uint8)
    out = annotate(frame, FrameResult(0, 0.0, (800, 300), men))
    assert out[96:100, 496:540].max() > 200                                    # the outline is there
    assert (out[100, 512:516] == 0).all()                                      # and no box of his own on a man inside it
