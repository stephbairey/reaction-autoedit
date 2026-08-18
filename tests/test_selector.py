import numpy as np

from reaction_autoedit.select.selector import Analysis, SelectParams, select


def _synthetic(duration=3600.0):
    segs = []
    # intro monologue 0-60 s
    for i in range(6):
        segs.append({"id": len(segs), "start": i * 10.0, "end": i * 10.0 + 8, "text": "hello patreon", "speaker": "REACTOR"})
    # film 60..3400 with dialogue every 8 s; reactor comment every 90 s
    t = 60.0
    while t < 3400:
        segs.append({"id": len(segs), "start": t, "end": t + 4, "text": "some film dialogue words here", "speaker": "FILM"})
        if int(t) % 90 == 0:
            segs.append({"id": len(segs), "start": t + 4.5, "end": t + 7, "text": "oh wow", "speaker": "REACTOR"})
        t += 8
    for i in range(4):  # outro
        segs.append({"id": len(segs), "start": 3420 + i * 10.0, "end": 3428 + i * 10.0, "text": "final thoughts", "speaker": "REACTOR"})
    peaks = [{"t": tt, "t0": tt - 3, "t1": tt + 5, "score": 3.0 - k * 0.1, "kind": "laugh", "text": "haha"}
             for k, tt in enumerate(range(200, 3400, 150))]
    music = [{"t0": 1000, "t1": 1060, "kind": "song", "level": 0.8, "song_conf": 0.5},
             {"t0": 2000, "t1": 2100, "kind": "score", "level": 0.6, "song_conf": 0.1}]
    cuts = np.arange(62, 3400, 11.0)
    dead = [{"t0": 2500, "t1": 2540, "dur": 40}]
    return Analysis(duration=duration, segments=segs, peaks=peaks, music=music, cuts=cuts, dead=dead,
                    film_start=60.0, film_end=3400.0)


def test_select_hits_runtime_and_rules():
    an = _synthetic()
    p = SelectParams(runtime_target_s=15 * 60, clip_cap_s=7.0, tolerance_s=45)
    edl = select(an, p, source="x.mp4")
    assert abs(edl.duration - 15 * 60) <= 120
    ins = [s.in_ for s in edl.segments]
    assert ins == sorted(ins)                                     # chronological
    assert all(s.dur <= 7.0 + 1e-6 for s in edl.segments if s.layout == "movie-large")
    assert edl.segments[0].kind == "intro" and edl.segments[-1].kind == "outro"
    assert any(s.kind == "cta" for s in edl.segments)             # withhold-the-climax
    for s in edl.segments:                                        # no movie-large inside the song span (≤0.5 s brush ok)
        if s.layout == "movie-large":
            assert max(0.0, min(s.out, 1060) - max(s.in_, 1000)) <= 0.5
    assert edl.validate_rules(clip_cap_s=7.0, source_duration=3600) == []


def test_select_without_withhold_has_no_cta():
    an = _synthetic()
    edl = select(an, SelectParams(runtime_target_s=10 * 60, withhold_climax=False), source="x.mp4")
    assert not any(s.kind == "cta" for s in edl.segments)
