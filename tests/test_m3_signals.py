import json

import numpy as np

from reaction_autoedit.analysis.audio_tags import AudioTags
from reaction_autoedit.analysis.music import detect_music
from reaction_autoedit.analysis.video_signals import detect_cuts


def test_detect_cuts_finds_spikes():
    rng = np.random.default_rng(1)
    change = rng.normal(3, 0.8, 600).clip(0)
    for i in (100, 250, 251, 420):   # 251 is within min-gap of 250
        change[i] = 40
    cuts = detect_cuts(change, fps=5.0, t0=0.0)
    assert [round(c) for c in cuts] == [20, 50, 84]


def test_music_spans_and_kinds(tmp_path):
    t = [2.5 + 2.5 * i for i in range(40)]
    music = [0.0] * 8 + [0.55] * 10 + [0.0] * 6 + [0.7] * 10 + [0.0] * 6
    singing = [0.0] * 8 + [0.05] * 10 + [0.0] * 6 + [0.5] * 10 + [0.0] * 6
    classes = ["Music", "Singing"]
    tags = {"win_s": 5.0, "hop_s": 2.5, "t": t, "classes": classes,
            "probs": {"Music": music, "Singing": singing}}
    tp = tmp_path / "audio_tags.json"
    tp.write_text(json.dumps(tags))
    out = detect_music(tp, tmp_path / "music.json")
    d = json.loads(out.read_text())
    kinds = [s["kind"] for s in d["spans"]]
    assert kinds == ["score", "song"]  # span 1: level .55 & singing .05 → score; span 2: level .7 ≥ .65 for 25 s → song
    at = AudioTags(tp)
    assert at.at("Music", np.array([30.0]))[0] > 0.5
    assert at.at("Music", np.array([5.0]))[0] == 0.0
