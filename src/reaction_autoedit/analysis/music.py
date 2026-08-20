"""Music tiering from the AudioSet tags (see audio_tags.py).

Risk tiers (from the brief): label-owned *songs* (needle drops, vocals) are the block-happy layer →
``kind = "song"``; orchestral *score* mostly yields revenue-share claims → ``kind = "score"``.

Spans: windows with Music ≥ ``music_thr`` (smoothed over 3 windows) merged with gaps ≤ ``gap_s``.
A span is a *song* when the max over its windows of the song classes (Singing, Vocal music, Pop,
Rock, Hip hop, R&B, Country, Choir) ≥ ``song_thr`` (LOW by design: dialogue over a needle drop
dilutes vocal probabilities badly), or when the span is long and music-dominant
(level ≥ ``level_song_thr`` for ≥ ``level_song_min_dur`` s); otherwise *score*. ``level`` is the mean Music
probability (how dominant the music is in the mix; selection avoids high-level spans harder).

Output ``music.json``::

    {"spans": [{"t0": 100.0, "t1": 130.5, "kind": "song", "level": 0.71, "song_conf": 0.42}]}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio_tags import SONG_CLASSES, AudioTags


def detect_music(tags_path: str | Path, out: str | Path, *, music_thr: float = 0.45, song_thr: float = 0.06,
                 level_song_thr: float = 0.65, level_song_min_dur: float = 15.0,
                 gap_s: float = 6.0, min_dur: float = 4.0, force: bool = False) -> Path:
    out = Path(out)
    if out.exists() and not force:
        return out
    tg = AudioTags(tags_path)
    t = tg.t
    music = tg.probs.get("Music", np.zeros(len(t), dtype=np.float32)).astype(np.float64)
    if len(music) >= 3:
        music = np.convolve(music, np.ones(3) / 3, mode="same")
    song = tg.group_max(SONG_CLASSES, t).astype(np.float64)
    on = music >= music_thr
    spans: list[dict] = []
    cur: dict | None = None
    half = tg.hop_s / 2
    for i, flag in enumerate(on):
        if flag:
            a, b = float(t[i] - half), float(t[i] + half)
            if cur and a - cur["t1"] <= gap_s:
                cur["t1"] = b
                cur["idx"].append(i)
            else:
                cur = {"t0": a, "t1": b, "idx": [i]}
                spans.append(cur)
    result = []
    for s in spans:
        if s["t1"] - s["t0"] < min_dur:
            continue
        idx = np.array(s["idx"])
        level = float(music[idx].mean())
        sc = float(song[idx].max())
        dur = s["t1"] - s["t0"]
        # vocals leak weakly through dialogue in PANNs 5 s windows, so the song bar is LOW; and a
        # long, loud music passage is treated as song-risk even without detected vocals
        # (conservative in the copyright-safe direction — see brief's music tiering).
        is_song = sc >= song_thr or (level >= level_song_thr and dur >= level_song_min_dur)
        result.append({"t0": round(s["t0"], 2), "t1": round(s["t1"], 2), "kind": "song" if is_song else "score",
                       "level": round(level, 3), "song_conf": round(sc, 3)})
    data = {"music_thr": music_thr, "song_thr": song_thr, "spans": result,
            "totals": {"song_s": round(sum(r["t1"] - r["t0"] for r in result if r["kind"] == "song"), 1),
                       "score_s": round(sum(r["t1"] - r["t0"] for r in result if r["kind"] == "score"), 1)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return out


class MusicSpans:
    def __init__(self, path: str | Path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.spans = d["spans"]

    def kind_at(self, t: float) -> str | None:
        for s in self.spans:
            if s["t0"] <= t <= s["t1"]:
                return s["kind"]
        return None

    def overlap(self, a: float, b: float) -> dict[str, float]:
        """Seconds of song / score inside [a, b]."""
        out = {"song": 0.0, "score": 0.0}
        for s in self.spans:
            o = min(b, s["t1"]) - max(a, s["t0"])
            if o > 0:
                out[s["kind"]] += o
        return out
