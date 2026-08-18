import json

from reaction_autoedit import ffmpeg
from reaction_autoedit.assemble.render import render
from reaction_autoedit.assemble.templates import make_endcard, make_lower_third
from reaction_autoedit.config import ReactorConfig, Branding
from reaction_autoedit.edl import EDL, Endcard, Overlay, Segment
from reaction_autoedit.models import Geometry
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_render_preview_smoke(fixture_videos, tmp_path):
    video, truth_path = fixture_videos["sbs"]
    geom = Geometry.model_validate(json.loads(truth_path.read_text()))
    ec = make_endcard(tmp_path / "endcard.png")
    lt = make_lower_third(tmp_path / "lt.png")
    reactor = ReactorConfig(branding=Branding(endcard=str(ec), lower_third=str(lt), endcard_duration=1.0))
    edl = EDL(source=str(video), segments=[
        Segment(id="a", **{"in": 1.0}, out=3.0, layout="movie-large", chapter="One"),
        Segment(id="b", **{"in": 3.0}, out=4.5, layout="reactor-large", transition="xfade"),
        Segment(id="c", **{"in": 6.0}, out=7.0, layout="movie-large"),
    ], overlays=[Overlay(at=0.5, dur=1.0)], endcard=Endcard(dur=1.0))
    out = tmp_path / "out.mp4"
    res = render(edl, geom, out=out, reactor=reactor, preview=True, jobs=2)
    info = ffmpeg.probe(out)
    assert info.width == 854 and info.height == 480
    assert abs(info.duration - (4.5 + 1.0)) < 0.35
    assert res.chapters.read_text().splitlines()[0].startswith("0:00")
    assert (tmp_path / "out.edl.json").exists()
    assert (tmp_path / "out.description.txt").exists()

    # second run reuses cached intermediates (no new encodes) and still produces the file
    out.unlink()
    render(edl, geom, out=out, reactor=reactor, preview=True, jobs=2)
    assert out.exists()
