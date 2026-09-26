from nfl_intel.replay import render_replay
from nfl_intel.schema import FrameResult, TrackedPlayer, read_csv, write_csv


def _results(calibrated: bool):
    out = []
    for f in range(5):
        players = [
            TrackedPlayer(1, 0, (10 + f, 20, 40 + f, 100), 0.9, (25 + f, 100), (30 + f, 20.0) if calibrated else None),
            TrackedPlayer(2, None, (200, 20, 240, 110), 0.8, (220, 110), (60.0, 26.7) if calibrated else None),
        ]
        out.append(FrameResult(f, f / 30, (1280, 720), players))
    return out


def test_csv_round_trip(tmp_path):
    path = tmp_path / "play.csv"
    assert write_csv(path, _results(True)) == 10
    back = read_csv(path)
    assert [r.frame for r in back] == [0, 1, 2, 3, 4]
    p = back[3].players[0]
    assert p.track_id == 1 and p.team == 0
    assert p.field_yd == (33.0, 20.0)
    assert back[3].players[1].team is None


def test_replay_renders_calibrated_and_uncalibrated(tmp_path):
    for calibrated in (True, False):
        csv_path = tmp_path / f"play_{calibrated}.csv"
        write_csv(csv_path, _results(calibrated))
        out = render_replay(csv_path, tmp_path / f"replay_{calibrated}.mp4", fps=10, scale=4)
        assert out.exists() and out.stat().st_size > 0
