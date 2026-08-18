import json

import pytest
from pydantic import ValidationError

from reaction_autoedit.edl import EDL, Segment, starter_edl


def test_segment_requires_out_after_in():
    with pytest.raises(ValidationError):
        Segment(id="x", **{"in": 5.0}, out=5.0)


def test_edl_roundtrip(tmp_path):
    e = starter_edl("src.mp4", 600.0)
    p = tmp_path / "edl.json"
    e.save(p)
    raw = json.loads(p.read_text())
    assert "in" in raw["segments"][0]           # alias used on disk
    e2 = EDL.load(p)
    assert e2.duration == pytest.approx(e.duration)
    assert [s.id for s in e2.segments] == [s.id for s in e.segments]


def test_offsets_and_duration():
    e = EDL(source="s.mp4", segments=[
        Segment(id="a", **{"in": 10}, out=15),
        Segment(id="b", **{"in": 20}, out=22.5),
    ])
    assert e.offsets() == [0.0, 5.0]
    assert e.duration == 7.5


def test_validate_rules_warnings():
    e = EDL(source="s.mp4", segments=[
        Segment(id="a", **{"in": 10}, out=20, layout="movie-large"),   # > clip cap
        Segment(id="b", **{"in": 5}, out=6),                           # not chronological
        Segment(id="b", **{"in": 30}, out=30.1),                       # dup id, very short
    ])
    warns = e.validate_rules(clip_cap_s=7.0, source_duration=25.0)
    joined = "\n".join(warns)
    assert "exceeds clip cap" in joined
    assert "not chronological" in joined
    assert "duplicate segment ids" in joined
    assert "exceeds source duration" in joined
    assert "very short" in joined


def test_starter_edl_short_source_is_valid():
    e = starter_edl("s.mp4", 30.0)
    assert e.validate_rules(clip_cap_s=7.0, source_duration=30.0) == []
