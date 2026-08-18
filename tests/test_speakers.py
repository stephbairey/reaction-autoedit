import numpy as np

from reaction_autoedit.analysis.speakers import Enrollment, calibrate, label_segments, reactor_spans


def _tl(sims, hop=0.5):
    return [{"t": 0.8 + i * hop, "sim": s, "db": -20.0 if s is not None else -60.0} for i, s in enumerate(sims)]


def test_calibrate_finds_bimodal_valley():
    rng = np.random.default_rng(0)
    sims = list(rng.normal(0.58, 0.04, 400)) + list(rng.normal(0.86, 0.04, 400))
    enrol = Enrollment(embedding=np.zeros(256), sims=rng.normal(0.9, 0.03, 100), sample="x")
    calibrate(enrol, _tl(sims))
    assert 0.66 <= enrol.threshold <= 0.80
    assert "otsu" in enrol.extra["calibration"]


def test_label_segments_rules():
    # 0..10s reactor-like, 10..20s film-like, 20..30s mixed, 30..40s silent
    sims = [0.9] * 20 + [0.5] * 20 + [0.9, 0.5] * 10 + [None] * 20
    tl = _tl(sims)
    segs = [{"id": 0, "start": 1, "end": 9, "text": "a"}, {"id": 1, "start": 11, "end": 19, "text": "b"},
            {"id": 2, "start": 21, "end": 29, "text": "c"}, {"id": 3, "start": 32, "end": 39, "text": "d"}]
    out = label_segments(segs, tl, threshold=0.7, margin=0.04)
    assert [s["speaker"] for s in out] == ["REACTOR", "FILM", "MIXED", "UNKNOWN"]


def test_reactor_spans_merge():
    sims = [0.9] * 6 + [0.5] * 1 + [0.9] * 6 + [0.5] * 10 + [0.9] * 1
    spans = reactor_spans(_tl(sims), threshold=0.7, min_dur=2.0, gap=1.0)
    assert len(spans) == 1            # small gap bridged, lone window too short
    assert spans[0]["t0"] == 0.0 and spans[0]["t1"] > 6.0
