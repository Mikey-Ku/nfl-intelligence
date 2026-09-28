import urllib.error

import polars as pl
import pytest

from nfl_intel import nflverse


def test_refresh_writes_what_is_published_and_skips_what_is_not(tmp_path, monkeypatch):
    class Reply:
        def __init__(self, data): self.data = data
        def read(self): return self.data
        def __enter__(self): return self
        def __exit__(self, *a): return False

    asked = []

    def fake(url, timeout=0):
        asked.append(url)
        if "participation" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return Reply(b"parquet bytes")

    monkeypatch.setattr(nflverse.urllib.request, "urlopen", fake)
    got = nflverse.refresh(2026, tmp_path)
    assert got == {"play-by-play": 13, "charting": 13, "participation": None}
    assert (tmp_path / "play_by_play_2026.parquet").read_bytes() == b"parquet bytes"
    assert not (tmp_path / "pbp_participation_2026.parquet").exists()
    assert not list(tmp_path.glob("*.part"))
    assert asked[0] == "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.parquet"


def test_other_download_errors_are_not_swallowed(tmp_path, monkeypatch):
    def fake(url, timeout=0):
        raise urllib.error.HTTPError(url, 500, "Server Error", None, None)

    monkeypatch.setattr(nflverse.urllib.request, "urlopen", fake)
    with pytest.raises(urllib.error.HTTPError):
        nflverse.refresh(2026, tmp_path, kinds=("charting",))


def test_describe_says_what_is_on_disk(tmp_path):
    assert nflverse.describe(2026, tmp_path) == "no play-by-play for 2026 on disk"
    pl.DataFrame({"week": [1, 1, 3], "game_id": ["a", "a", "b"]}).write_parquet(tmp_path / "play_by_play_2026.parquet")
    assert nflverse.describe(2026, tmp_path) == "2026: 3 plays, 2 games, through week 3"
