import json

import pytest

from reaction_autoedit.ingest.layout import detect_layout
from reaction_autoedit.models import Geometry
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
@pytest.mark.parametrize("preset", ["sbs", "pip"])
def test_detect_layout_matches_fixture_truth(fixture_videos, preset, tmp_path):
    video, truth_path = fixture_videos[preset]
    truth = Geometry.model_validate(json.loads(truth_path.read_text()))
    g = detect_layout(video, n_frames=30, debug_image=tmp_path / f"{preset}.png")
    assert g.movie_inner.iou(truth.movie_inner) >= 0.9, g
    assert g.face.iou(truth.face) >= 0.85, g
    assert g.face.shape == truth.face.shape
    assert g.confidence >= 0.5
    assert (tmp_path / f"{preset}.png").exists()


@requires_ffmpeg
def test_template_overrides_on_disagreement(fixture_videos):
    video, truth_path = fixture_videos["sbs"]
    truth = Geometry.model_validate(json.loads(truth_path.read_text()))
    bogus = truth.model_copy(deep=True)
    bogus.face.x, bogus.face.w = 900, 300  # clearly wrong template
    g = detect_layout(video, n_frames=20, template=bogus)
    # detector is confident and disagrees → template wins by design (per-reactor templates are ground truth)
    assert g.source == "template"
    assert any("disagreed" in n or "low confidence" in n for n in g.notes)
