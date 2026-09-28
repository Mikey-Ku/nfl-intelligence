"""The whole chain on a synthetic play: anchor, snap, roles, value, and what the panel is told."""

import numpy as np
import pytest

from formations import film
from nfl_intel.analysis import LIVE_FOR, AnalyzedTracker, PlayAnalyzer, live_stats
from nfl_intel.capture import VideoSource
from nfl_intel.panel import render_panel
from nfl_intel.playinfo import PlayHeader, PlayRecord
from nfl_intel.schema import read_csv, write_csv
from nfl_intel.topdown import FieldSlice, TopDownWriter, draw_topdown, title_for
from nfl_intel.value import FEATURES, PassValueModel


class SeparationModel(PassValueModel):
    """Stand-in model: catch chance grows with separation, yards with depth."""

    def __init__(self):
        super().__init__(None, None, {"baseline": [5.0] * 60})

    def score(self, X):
        sep, depth = X[:, FEATURES.index("sep1")], X[:, FEATURES.index("depth")]
        return np.clip(0.3 + 0.1 * sep, 0.05, 0.95), np.clip(depth + 3.0, 0.0, None)


def _record(play_type="pass"):
    header = PlayHeader(quarter=4, clock="08:21", down=2, distance=7, offense="GB", side="ATL", yard=15)
    return PlayRecord(header, matched=True, defense="ATL", yards_to_goal=15, play_type=play_type, yards_gained=15.0,
                      touchdown=True, complete=True, passer="J.Love", receiver="M.Golden",
                      charted={"qb_location": "S", "n_defense_box": 6, "n_pass_rushers": 4})


class FakeTracker:
    """Stands in for PlayTracker: hands out prepared frames."""

    def __init__(self, frames):
        self.frames = {f.frame: f for f in frames}
        self.field = None
        self.registrar = None
        self.shot = 0

    def process(self, frame, idx, t):
        return self.frames[idx]

    def reset(self):
        pass


def _run(frames, analyzer):
    img = np.zeros((4, 4, 3), np.uint8)
    return [analyzer.update(img, f) for f in frames]


def test_pass_play_is_anchored_snapped_and_scored():
    frames, snap_at = film(direction=-1, los_rel=40.0)
    an = PlayAnalyzer(SeparationModel(), record=_record())
    out = _run(frames, an)
    a = an.anchor
    assert a is not None and a.direction == -1 and a.offense_team == 0
    assert a.los_abs == 95.0 and a.los_rel == pytest.approx(40.0)
    assert an.snap_t == pytest.approx(snap_at, abs=0.25)
    roles = list(an.roles.values())
    assert roles.count("QB") == 1 and roles.count("REC") == 5 and roles.count("DL") == 4
    label = {s.track_id: an.labels.get(s.id) for s in an.tracker.slots}
    assert label[0] == "QB" and label[8] == "R1" and label[10] == "R5"        # numbered from the offense's left
    assert an.look[0] == "Shotgun, 3 x 1, 1 back"

    before = [o for o, f in zip(out, frames) if f.t < snap_at - 0.1]
    assert all(o.value is None for o in before)
    live = [o for o in out if o.live]
    assert len(live) > 30 and all(o.value is not None for o in live)
    assert live[-1].value > live[0].value                          # the routes get deeper
    assert live[-1].best.label in {"R1", "R2", "R5"}
    # positions come out on the real field: the quarterback behind the 15, in the middle
    qb = next(p for p in out[-1].players if p.role == "QB")
    assert 85.0 < qb.field_yd[0] < 95.0 and 24.0 < qb.field_yd[1] < 28.0
    assert an.explain(0.5, 1.5) != ""


def test_the_panel_is_told_what_is_going_on(qt_app):
    frames, snap_at = film()
    an = PlayAnalyzer(SeparationModel(), record=_record())
    stats = live_stats(an, an.value_model)
    img = np.zeros((4, 4, 3), np.uint8)
    notes, snaps = [], []
    for f in frames:
        an.update(img, f)
        snaps.append(stats.update(f))
        notes.append(snaps[-1].value_note)
    assert "line of scrimmage" in notes[0]
    assert any("Waiting for the snap" in n for n in notes)
    last = snaps[-1]
    assert last.value is not None and last.value_note == "" and last.value_unit == "yd"
    assert last.delta_note == "vs a typical play" and last.value_delta == pytest.approx(last.value - 5.0)
    assert last.clock_text.startswith("snap +")
    assert last.team_names[:2] == ("GB offense", "ATL defense")
    names = [name for name, _ in last.rows]
    assert names[:3] == ["Play", "Best throw", "Most open"] and "Offense" in names
    assert render_panel(last, 300, 560, dpr=1.0).width() == 300


def test_a_run_play_gets_roles_but_no_pass_value():
    frames, _ = film()
    an = PlayAnalyzer(SeparationModel(), record=_record("run"))
    out = _run(frames, an)
    assert an.snap_t is not None and all(o.value is None for o in out)
    assert an.value_note().startswith("Run play")


def test_after_the_play_the_panel_says_what_happened():
    frames, snap_at = film(run=LIVE_FOR + 1.0)
    an = PlayAnalyzer(SeparationModel(), record=_record())
    out = _run(frames, an)
    assert not out[-1].live and out[-1].value is None
    rows = dict(an.rows())
    assert rows["Result"] == "Love to Golden, 15 yd, touchdown"
    assert rows["Charted"] == "Shotgun, 6 in the box, 4 rush"
    assert "Best throw" not in rows


def test_unmapped_frames_say_so():
    frames, _ = film()
    for f in frames:
        for p in f.players:
            p.field_yd = None
    an = PlayAnalyzer(SeparationModel(), record=_record())
    _run(frames[:5], an)
    assert "not mapped" in an.value_note() and an.anchor is None
    assert PlayAnalyzer(None, record=_record()).has_model is False


def test_analyzed_tracker_puts_the_whole_shot_on_the_real_field(tmp_path):
    frames, snap_at = film(direction=-1, los_rel=40.0)
    tracker = AnalyzedTracker(FakeTracker(frames), PlayAnalyzer(SeparationModel(), record=_record()))
    img = np.zeros((4, 4, 3), np.uint8)
    results = [tracker.process(img, f.frame, f.t) for f in frames]
    assert not results[0].anchored and results[-1].anchored       # the first frames came before the anchor
    tracker.finish()
    assert all(r.anchored for r in results)
    for r in (results[0], results[-1]):
        qb = next(p for p in r.players if p.track_id == 0)
        assert qb.role == "QB" and qb.label == "QB" and 85.0 < qb.field_yd[0] < 95.0
    corner = next(p for p in results[0].players if p.track_id == 17)
    assert corner.role == "DEF" and corner.field_yd[0] == pytest.approx(100.0, abs=0.5)

    path = tmp_path / "play.csv"
    write_csv(path, results)
    back = read_csv(path)
    assert back[0].anchored and back[0].players[0].role == "QB"


def test_topdown_draws_and_writes(tmp_path, qt_app):
    frames, _ = film()
    an = PlayAnalyzer(SeparationModel(), record=_record())
    stats = live_stats(an, an.value_model)
    writer = TopDownWriter(tmp_path, "clip", fps=20.0, height=360, panel_width=300)
    img = np.zeros((4, 4, 3), np.uint8)
    for f in frames:
        a = an.update(img, f)
        writer.add(f, a, stats.update(f))
    writer.close(hold=0.2)
    assert len(writer.paths) == 1 and writer.paths[0].stat().st_size > 10_000
    view = FieldSlice(95.0, height=360, to_go=7)
    assert (view.x0, view.x1) == (80.0, 120.0)
    assert view.P((95.0, 0.0)) == (view.px(95.0), 360)
    picture = draw_topdown(view, a)
    assert picture.shape == (360, view.width, 3) and (picture != view.base).any()
    assert title_for(a).startswith("Q4 08:21  2nd & 7  GB on ATL 15") and "from the snap" in title_for(a)


def test_video_source_uses_the_real_capture_times(tmp_path):
    import cv2

    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(5):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    assert [round(t, 3) for _, t, _ in VideoSource(str(path))] == [0.0, 0.033, 0.067, 0.1, 0.133]
    path.with_suffix(".times.csv").write_text("frame,t\n0,10.00\n1,10.05\n2,10.10\n3,10.15\n4,10.20\n")
    src = VideoSource(str(path))
    assert [round(t, 2) for _, t, _ in src] == [0.0, 0.05, 0.1, 0.15, 0.2]
    assert src.fps == pytest.approx(20.0)
