"""The truth tool: the run's mapping refitted per frame, frames sampled, numbers carried, the CSV scored."""

import numpy as np

from nfl_intel.evaluate import evaluate
from nfl_intel.schema import FrameResult, TrackedPlayer, read_csv, write_csv
from nfl_intel.truth import (Click, TruthFrame, assign_id, default_out, frame_mapping, mappings_for, resume, sample_frames,
                             save, to_results, typed)

H = np.array([[0.05, 0.0, 10.0], [0.0, 0.08, 5.0], [0.0, 0.0, 1.0]])       # a plain pixel -> yards mapping


def _yd(px):
    v = H @ np.array([px[0], px[1], 1.0])
    return (float(v[0] / v[2]), float(v[1] / v[2]))


def _run(n_frames=6, fps=20.0, players=8, lose_mapping_on=()):
    frames = []
    for i in range(n_frames):
        ps = []
        for j in range(players):
            px = (100.0 + 60 * j + 4 * i, 400.0 + 30 * (j % 3))
            ps.append(TrackedPlayer(j + 1, j % 2, (px[0] - 10, px[1] - 30, px[0] + 10, px[1]), 0.9, px,
                                    None if i in lose_mapping_on else _yd(px)))
        frames.append(FrameResult(i, i / fps, (1000, 600), ps, shot=2, anchored=True))
    return frames


def test_frame_mapping_recovers_the_runs_homography():
    m = frame_mapping(_run()[0])
    assert m is not None
    px = np.array([[333.0, 444.0]])
    assert np.allclose(m.to_field(px)[0], _yd((333.0, 444.0)), atol=1e-6)


def test_a_frame_with_too_few_mapped_players_borrows_a_neighbour():
    run = _run(lose_mapping_on=(2,))
    maps = mappings_for(run)
    assert maps[2] is not None and np.allclose(maps[2].H, maps[1].H)
    assert frame_mapping(run[2]) is None


def test_sample_frames_every_half_second_in_one_shot():
    run = _run(n_frames=40)
    run[0].shot = 1                                        # a frame from another play
    picked = sample_frames(run, 0.5, shot=2)
    assert [r.frame for r in picked] == [1, 11, 21, 31]
    assert [r.frame for r in sample_frames(run, 0.5, start=1.0)] == [20, 30]


def test_assign_id_carries_the_nearest_previous_number_and_never_doubles_up():
    previous = [Click((0, 0), (10.0, 10.0), 1), Click((0, 0), (20.0, 10.0), 2)]
    assert assign_id((10.5, 10.2), previous, set(), reach=3.0) == 1
    assert assign_id((10.5, 10.2), previous, {1}, reach=3.0) == 3        # 1 already used on this frame: a new man
    assert assign_id((40.0, 40.0), previous, set(), reach=3.0) == 3      # nobody near
    assert assign_id(None, previous, {3}, reach=3.0) == 4                # no mapping: still a fresh number


def test_truth_csv_round_trips_and_scores_the_run(tmp_path):
    run = _run()
    truth = []
    for r in run:
        tf = TruthFrame(r.frame, r.t, r.shot, r.anchored)
        for p in r.players:
            px = (p.foot_px[0] + 3.0, p.foot_px[1])                    # the labeller clicks 3 px off
            tf.clicks.append(Click(px, _yd(px), p.track_id))
        truth.append(tf)
    out = tmp_path / "truth.csv"
    write_csv(out, to_results(truth))
    back = read_csv(out)
    assert len(back) == 6 and all(p.field_yd is not None for r in back for p in r.players)
    score = evaluate(run, back)
    assert score.unit == "yd" and score.recall == 1.0 and score.id_switches == 0
    assert abs(score.position_error - 0.15) < 1e-6                       # 3 px at 0.05 yd/px


def test_default_out_names_the_clip_and_shot():
    assert default_out("data/clips/live-02.mp4", 5).as_posix() == "data/truth/live-02.shot5.csv"
    assert default_out("x/live-02.mp4", None).as_posix() == "data/truth/live-02.csv"


def test_a_saved_play_is_picked_up_where_it_stopped(tmp_path):
    out = tmp_path / "truth" / "clip.shot2.csv"
    truth = [TruthFrame(i * 10, i * 0.5, 2, True) for i in range(4)]
    assert save(out, truth) == (0, 0) and not out.exists()                  # nothing clicked: no file
    truth[0].clicks = [Click((100.0, 400.0), (15.0, 37.0), 1), Click((160.0, 430.0), (18.0, 39.4), 2)]
    truth[1].clicks = [Click((104.0, 400.0), (15.2, 37.0), 1)]
    assert save(out, truth) == (2, 3)
    again = [TruthFrame(i * 10, i * 0.5, 2, True) for i in range(4)]
    assert resume(again, read_csv(out)) == 2                                # the first frame with no clicks
    assert [(c.pid, c.px, c.yd) for c in again[0].clicks] == [(1, (100.0, 400.0), (15.0, 37.0)), (2, (160.0, 430.0), (18.0, 39.4))]
    assert len(again[1].clicks) == 1 and not again[2].clicks
    for tf in again:
        tf.clicks = tf.clicks or [Click((1.0, 1.0), None, 9)]
    assert resume(again, []) == 3                                           # all done: stay on the last frame


def test_a_number_is_typed_in_the_window():
    entry = ""
    for key in (ord("1"), ord("x"), ord("7"), 8, ord("2")):                 # 1, a stray letter, 7, backspace, 2
        entry, number = typed(entry, key)
        assert number is None
    assert entry == "12"
    assert typed(entry, 13) == (None, 12)                                   # Enter sets it
    assert typed("12", 27) == (None, None) and typed("", 13) == (None, None)   # Esc, or Enter on nothing, leaves it
    assert typed("123", ord("4")) == ("123", None)
