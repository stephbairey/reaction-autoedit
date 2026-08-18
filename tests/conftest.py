from __future__ import annotations

import pytest

from reaction_autoedit import ffmpeg

requires_ffmpeg = pytest.mark.skipif(not ffmpeg.available(), reason="ffmpeg/ffprobe not found")


@pytest.fixture(scope="session")
def fixture_videos(tmp_path_factory):
    """Short synthetic composites (both presets), generated once per session."""
    if not ffmpeg.available():
        pytest.skip("ffmpeg not available")
    from reaction_autoedit.ingest.fixture import PRESETS, make_fixture

    d = tmp_path_factory.mktemp("fixtures")
    out = {}
    for p in PRESETS:
        v, j = make_fixture(d, p, duration=12.0)
        out[p] = (v, j)
    return out
