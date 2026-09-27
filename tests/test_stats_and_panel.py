import math

import pytest

from nfl_intel.schema import FrameResult, TrackedPlayer
from nfl_intel.stats import LiveStats


def _frame(i, ids, fps=20.0, field=False, teams=None):
    players = []
    for k, tid in enumerate(ids):
        team = None if teams is None else teams[k % len(teams)]
        xy = (100.0 + 40 * k, 300.0)
        players.append(TrackedPlayer(tid, team, (xy[0] - 10, 250, xy[0] + 10, 300), 0.9, xy, (30.0 + k, 20.0) if field else None))
    return FrameResult(i, i / fps, (1280, 720), players)


def test_counts_fps_and_teams():
    stats = LiveStats()
    for i in range(40):
        snap = stats.update(_frame(i, range(22), teams=[0, 1]))
    assert snap.on_field == 22 and snap.active_ids == 22
    assert snap.team_counts == (11, 11, 0)
    assert math.isclose(snap.fps, 20.0, rel_tol=0.05)
    assert snap.play == 1 and snap.play_clock == pytest.approx(39 / 20)
    assert snap.reacquired == 0 and snap.lost == 0
    assert snap.value is None and not snap.calibrated
    assert len(snap.history) == 40 and snap.history[-1] == (0.0, 22)


def test_late_pickups_and_losses():
    stats = LiveStats(settle=1.0, lost_after=0.5)
    for i in range(30):                                   # 1.5 s with ids 0..9
        stats.update(_frame(i, range(10)))
    for i in range(30, 60):                               # id 9 swaps to 99, id 0 disappears
        snap = stats.update(_frame(i, [*range(1, 9), 99]))
    assert snap.reacquired == 1          # 99 appeared after the play settled
    assert snap.lost == 2                # 0 and 9 are gone


def test_play_segmentation_resets_counters():
    stats = LiveStats(gap_reset=1.0)
    i = 0
    for _ in range(20):
        stats.update(_frame(i, range(12))); i += 1
    for _ in range(30):                                   # 1.5 s of nothing: replay, menu, black
        snap = stats.update(_frame(i, [])); i += 1
    assert snap.play == 1 and snap.play_clock == 0.0
    for _ in range(10):
        snap = stats.update(_frame(i, range(100, 112))); i += 1
    assert snap.play == 2
    assert snap.reacquired == 0 and snap.lost == 0       # new play, fresh counters
    assert snap.play_clock == pytest.approx(9 / 20)


def test_value_delta_and_swings_need_calibration():
    values = iter([0.50] * 10 + [0.50, 0.46, 0.40, 0.34, 0.30] + [0.30] * 10 + [0.35, 0.45, 0.55, 0.62] + [0.62] * 10)
    stats = LiveStats(value_fn=lambda r: next(values), metrics_fn=lambda r: [("In the box", "7")])
    snap = stats.update(_frame(0, range(12), field=False))
    assert snap.value is None and snap.rows == []
    stats = LiveStats(value_fn=lambda r: next(values), metrics_fn=lambda r: [("In the box", "7")])
    for i in range(39):
        snap = stats.update(_frame(i, range(12), field=True))
    assert snap.calibrated and snap.rows == [("In the box", "7")]
    assert snap.value == pytest.approx(0.62)
    assert snap.value_delta == pytest.approx(0.12)
    assert len(snap.swings) == 2
    drop, rise = snap.swings
    assert drop.delta < -0.15 and rise.delta > 0.25 and drop.t < rise.t


def test_panel_renders_both_states():
    from PySide6.QtGui import QGuiApplication

    from nfl_intel.panel import qimage_to_bgr, render_panel

    app = QGuiApplication.instance() or QGuiApplication([])
    stats = LiveStats()
    for i in range(30):
        snap = stats.update(_frame(i, range(22), teams=[0, 1]))
    off = qimage_to_bgr(render_panel(snap, 300, 560, dpr=2.0))
    assert off.shape == (1120, 600, 3)
    assert off.std() > 5                                  # something was drawn

    vals = iter([0.5 + 0.01 * k for k in range(40)])
    stats = LiveStats(value_fn=lambda r: next(vals), metrics_fn=lambda r: [("Formation", "Shotgun trips"), ("In the box", "6")])
    for i in range(40):
        snap = stats.update(_frame(i, range(22), field=True, teams=[0, 1]))
    on = qimage_to_bgr(render_panel(snap, 300, 560, dpr=2.0))
    assert on.shape == off.shape
    assert (on.astype(int) - off.astype(int)).any()       # the value card changed
    del app
