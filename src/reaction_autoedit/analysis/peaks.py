"""Reaction peak detection — where are his biggest moments?

Works on the 0.5 s speaker timeline (speakers.json) and fuses:

* **vocal energy** of REACTOR windows (his loudness relative to his own baseline)
* **face motion** (mouth/jaw region, from video signals) — laughing, jaw-drops, head movement
* **AudioSet tags** (audio_tags.json): laughter / shout / gasp probabilities, weighted by how
  reactor-like the window sounds (so the *film's* laughter and screams don't score)
* a small bonus for exclamations in REACTOR transcript segments ("oh my god", "no way", …)

The combined signal is smoothed (~2.5 s) and local maxima ≥ ``min_gap_s`` apart become peaks;
each peak gets an extent (where the signal stays above 40 % of the peak, bounded 2–14 s), a
``kind`` (laugh | shout | gasp | talk | visual) and the REACTOR transcript text inside it.

Output ``peaks.json``::

    {"peaks": [{"t": 3013.5, "t0": 3010.0, "t1": 3018.5, "score": 3.4, "kind": "laugh",
                "components": {...}, "text": "Is he gonna throw up?"}], "n": 42}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .audio_tags import LAUGH_CLASSES, SHOUT_CLASSES, AudioTags
from .face_motion import FaceMotion

EXCLAIM = re.compile(r"\b(oh my god|oh my gosh|holy|no way|what the|are you kidding|dude|bro|oh no|oh man|"
                     r"jesus|damn|what\?|wow|yo|nooo+|whoa|come on|let'?s go|yes+!)\b", re.I)


def _z(x: np.ndarray, mask: np.ndarray | None = None, clip: float = 3.0) -> np.ndarray:
    ref = x[mask] if mask is not None and mask.any() else x
    ref = ref[~np.isnan(ref)]
    if ref.size < 5:
        return np.zeros_like(x)
    z = (x - np.nanmean(ref)) / (np.nanstd(ref) + 1e-6)
    return np.clip(np.nan_to_num(z), -clip, clip)


def detect_peaks(
    speakers_path: str | Path,
    out: str | Path,
    *,
    face_motion: FaceMotion | None = None,
    tags: AudioTags | None = None,
    transcript_segments: list[dict] | None = None,
    min_gap_s: float = 12.0,
    max_peaks: int | None = None,
    force: bool = False,
) -> Path:
    out = Path(out)
    if out.exists() and not force:
        return out
    d = json.loads(Path(speakers_path).read_text(encoding="utf-8"))
    tl = d["timeline"]
    key, thr, margin = d.get("score_key", "sim"), float(d["threshold"]), float(d.get("margin", 0.02))
    t = np.array([w["t"] for w in tl], dtype=np.float64)
    db = np.array([w["db"] for w in tl], dtype=np.float64)
    sc = np.array([np.nan if w.get(key) is None else w[key] for w in tl], dtype=np.float64)
    voiced = ~np.isnan(sc)
    reactor_soft = np.where(voiced, 1 / (1 + np.exp(-(np.nan_to_num(sc) - thr) / max(margin, 1e-3))), 0.0)
    reactor = voiced & (np.nan_to_num(sc, nan=-9) >= thr)

    # vocal energy relative to his own baseline (only counts on reactor windows)
    energy = np.zeros(len(t))
    if reactor.sum() >= 5:
        base = np.median(db[reactor])
        spread = np.std(db[reactor]) + 1e-6
        energy = np.where(reactor, np.clip((db - base) / spread, -1, 3), 0.0)
        energy = np.clip(energy + 0.5, 0, None) * reactor  # any reactor speech counts a bit

    motion = np.zeros(len(t))
    if face_motion is not None:
        m = face_motion.per_window(t)
        motion = np.clip(_z(m), 0, None)

    laugh = shout = gasp = np.zeros(len(t))
    if tags is not None:
        laugh = tags.group_max(LAUGH_CLASSES, t) * (0.35 + 0.65 * reactor_soft)
        shout = tags.group_max(SHOUT_CLASSES, t) * (0.35 + 0.65 * reactor_soft)
        gasp = tags.at("Gasp", t) * (0.35 + 0.65 * reactor_soft)

    excl = np.zeros(len(t))
    react_text: list[tuple[float, float, str]] = []
    if transcript_segments:
        for s in transcript_segments:
            if s.get("speaker") in ("REACTOR", "MIXED") and s.get("text"):
                react_text.append((s["start"], s["end"], s["text"]))
                if EXCLAIM.search(s["text"]):
                    excl[(t >= s["start"] - 0.5) & (t <= s["end"] + 0.5)] = 1.0 if s.get("speaker") == "REACTOR" else 0.5

    # startle: two flavours.
    # (a) a loud non-reactor stretch followed by his motion/speech within ~3 s (crashes, scare chords)
    # (b) a sudden JUMP in his face motion (gunshots and jump-scares are transients that RMS windows
    #     miss entirely — but his flinch is unmistakable in the motion derivative)
    startle = np.zeros(len(t))
    if len(t) > 10:
        loud_thr = np.percentile(db[db > -80], 88) if (db > -80).any() else 0.0
        film_loud = ((db >= loud_thr) & ~reactor).astype(float)
        k6 = int(3.0 / 0.5)
        trail = np.convolve(film_loud, np.ones(k6), mode="full")[: len(t)]  # loudness in the last 3 s
        response = np.maximum(motion / 2.0, np.clip(energy, 0, 1))
        startle = np.clip(trail, 0, 1) * response
        jump = np.zeros(len(t))
        jump[1:] = np.clip(np.diff(motion), 0, None)          # positive motion-z acceleration
        startle = startle + np.clip(jump - 0.8, 0, 3.0)       # only sharp jumps count

    comp = {"energy": 1.0 * energy, "motion": 0.8 * motion, "laugh": 2.5 * laugh, "shout": 2.5 * shout,
            "gasp": 1.5 * gasp, "exclaim": 0.6 * excl, "startle": 1.2 * startle}
    raw = sum(comp.values())
    k = int(round(2.5 / 0.5))
    kern = np.bartlett(2 * k + 1)
    kern /= kern.sum()
    sm = np.convolve(raw, kern, mode="same")

    # local maxima with min gap
    order = np.argsort(-sm)
    picked: list[int] = []
    gap = int(round(min_gap_s / 0.5))
    floor = max(float(np.percentile(sm, 78)), 0.5)
    for i in order:
        if sm[i] < floor:
            break
        if all(abs(i - j) >= gap for j in picked):
            picked.append(int(i))
        if max_peaks and len(picked) >= max_peaks:
            break
    picked.sort()

    peaks = []
    for i in picked:
        lvl = 0.4 * sm[i]
        a = i
        while a > 0 and sm[a - 1] >= lvl and (i - a) < 14 * 2:
            a -= 1
        b = i
        while b < len(sm) - 1 and sm[b + 1] >= lvl and (b - i) < 14 * 2:
            b += 1
        t0, t1 = float(t[a] - 0.8), float(t[b] + 0.8)
        if t1 - t0 < 2.0:
            t0, t1 = t[i] - 1.0, t[i] + 1.0
        seg = slice(a, b + 1)
        cs = {n: round(float(v[seg].mean()), 3) for n, v in comp.items()}
        kind_scores = {"laugh": cs["laugh"], "shout": cs["shout"], "gasp": cs["gasp"],
                       "talk": cs["energy"] + cs["exclaim"], "visual": cs["motion"], "startle": cs["startle"]}
        kind = max(kind_scores, key=kind_scores.get)
        text = " ".join(x for (s0, s1, x) in react_text if s0 <= t1 and s1 >= t0)
        peaks.append({"t": round(float(t[i]), 2), "t0": round(t0, 2), "t1": round(t1, 2),
                      "score": round(float(sm[i]), 3), "kind": kind, "components": cs, "text": text[:240]})
    data = {"n": len(peaks), "floor": round(floor, 3), "min_gap_s": min_gap_s, "peaks": peaks,
            "signal": {"hop_s": 0.5, "t0": float(t[0]) if len(t) else 0.0, "values": [round(float(v), 3) for v in sm]}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data) + "\n", encoding="utf-8")
    return out
