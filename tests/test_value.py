"""Value model tests on synthetic plays; no Big Data Bowl files needed."""

import numpy as np
import pytest

from nfl_intel.bdb import HZ, Play
from nfl_intel.demo import decision, explain, make_stats, score_plays, short_name, to_results
from nfl_intel.value import FEATURES, Kinematics, PassValueModel, play_value, receiver_features, velocities
from nfl_intel.value_train import evaluate, fit


def _play(frames=30, open_at=15):
    """QB, two receivers, their two defenders, a linebacker and a deep safety.
    Receiver 1 starts covered and breaks open at `open_at`."""
    xy = np.zeros((frames, 7, 2))
    for t in range(frames):
        s = t / HZ
        xy[t, 0] = (25.0 - min(s, 1.0) * 3, 26.0)                 # QB drops back
        xy[t, 1] = (30.0 + 6.0 * s, 40.0)                          # receiver on a go route
        xy[t, 2] = (30.0 + 3.0 * s, 12.0)                          # second receiver
        gap = 0.8 if t < open_at else 0.8 + (t - open_at) * 0.35
        xy[t, 3] = (30.0 + 6.0 * s - gap, 40.0)                    # his defender, falling behind
        xy[t, 4] = (30.0 + 3.0 * s + 1.0, 12.5)                    # tight on receiver 2 all play
        xy[t, 5] = (35.0, 26.0)                                    # linebacker
        xy[t, 6] = (75.0, 26.0)                                    # deep safety
    return Play(
        game_id=1, play_id=7, week=16, los=30.0, xy=xy,
        offense=np.array([True, True, True, False, False, False, False]),
        roles=["QB", "REC", "REC", "DEF", "DEF", "DEF", "DEF"],
        names=["Pat Passer", "Go Route Jr.", "Slot Man", "Corner One", "Corner Two", "Mike Backer", "Deep Safety"],
        positions=["QB", "WR", "WR", "CB", "CB", "MLB", "FS"],
        qb=0, target=1, ball_land=(55.0, 40.0), complete=True, yards=25.0, desc="synthetic",
    )


class SeparationModel(PassValueModel):
    """Stand-in model: catch chance grows with separation, yards with depth."""

    def __init__(self):
        super().__init__(None, None)

    def score(self, X):
        X = np.atleast_2d(X)
        return np.clip(0.2 + 0.15 * X[:, 0], 0.05, 0.95), np.clip(4.0 + X[:, 4], 0.0, 60.0)


def test_velocities_from_positions():
    xy = np.zeros((10, 1, 2))
    xy[:, 0, 0] = np.arange(10) * 0.5          # 0.5 yd per frame at 10 Hz = 5 yd/s
    v = velocities(xy, HZ)
    assert v[0, 0].tolist() == [0.0, 0.0]
    assert v[6, 0, 0] == pytest.approx(5.0)


def test_receiver_features_geometry():
    play = _play()
    v = velocities(play.xy, HZ)
    f = dict(zip(FEATURES, receiver_features(play.xy[20], v[20], 1, 0, play.defenders, play.los, 2.0)))
    assert f["sep1"] == pytest.approx(0.8 + 5 * 0.35)
    assert f["depth"] == pytest.approx(12.0)
    assert f["sideline"] == pytest.approx(160 / 3 - 40.0)
    assert f["speed"] == pytest.approx(6.0) and f["vx"] == pytest.approx(6.0)
    assert f["closing"] < 0                       # the defender is falling behind
    assert f["leverage"] > 0.5                    # and trails on the quarterback's side
    assert f["n_def_3"] == 1 and f["t"] == 2.0
    assert len(f) == len(FEATURES)


def test_value_rises_when_receiver_breaks_open():
    sp = score_plays([_play()], SeparationModel())[0]
    assert len(sp.values) == 30
    assert sp.values[-1] > sp.values[10] + 3
    assert sp.best[-1].index == 1 and sp.best[-1].label == "Route"
    assert "opens up" in explain(sp, 1.5, 2.5)
    assert decision(sp) == ("Threw to", f"Route  option 1 of 2, {sp.best[-1].expected_yards:.1f} yd")


def test_panel_stats_from_a_scored_play():
    sp = score_plays([_play()], SeparationModel())[0]
    stats = make_stats(sp)
    for res in to_results(sp):
        snap = stats.update(res)
    assert snap.calibrated and snap.value_unit == "yd"
    assert snap.value == pytest.approx(sp.values[-1])
    assert snap.value_delta == pytest.approx(sp.values[-1] - sp.values[0])
    assert snap.swings and snap.swings[-1].delta > 0 and "Route" in snap.swings[-1].label
    assert [name for name, _ in snap.rows] == ["Best throw", "Most open", "Tightest coverage", "Threw to"]


def test_play_value_picks_the_best_expected_yards():
    from nfl_intel.value import ReceiverScore

    a = ReceiverScore(1, 0.9, 5.0, 4.5, 3.0, "short")
    b = ReceiverScore(2, 0.4, 30.0, 12.0, 1.0, "deep")
    assert play_value([a, b]) == (12.0, b)
    assert play_value([]) == (0.0, None)


def test_kinematics_smooths_and_differences():
    k = Kinematics(lag=0.4, smooth=1.0)
    for i in range(10):
        p, v = k.update(3, i / 10, (i * 0.5, 20.0))
    assert v[0] == pytest.approx(5.0) and v[1] == pytest.approx(0.0)
    k.forget(set())
    assert k.update(3, 1.0, (9.0, 20.0))[1].tolist() == [0.0, 0.0]


def test_short_name_drops_suffixes():
    assert short_name("Velus Jones Jr.") == "Jones"
    assert short_name("Cole Kmet") == "Kmet"


def test_fit_and_evaluate_learn_separation(tmp_path):
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(0, 1, (n, len(FEATURES)))
    X[:, 0] = rng.uniform(0, 8, n)                                # separation
    X[:, 4] = rng.uniform(-2, 30, n)                              # depth
    caught = (rng.uniform(0, 1, n) < np.clip(0.25 + 0.09 * X[:, 0], 0, 0.95)).astype(int)
    yards = np.clip(X[:, 4] + 4 + rng.normal(0, 2, n), 0, None)
    model = fit(X[:1000], caught[:1000], yards[:1000])
    r = evaluate(model, X[1000:], caught[1000:], yards[1000:])
    assert r["auc"] > 0.65
    assert r["log_loss"] < r["log_loss_baseline"]
    assert r["yards_mae"] < r["yards_mae_baseline"]
    path = tmp_path / "m.pkl"
    model.save(path)
    again = PassValueModel.load(path)
    assert np.allclose(again.score(X[:5])[0], model.score(X[:5])[0])
