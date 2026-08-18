"""AudioSet tagging of the mixed track with PANNs (CNN14) — one model, several uses:

* music tiering (Music / Singing / Vocal music / genre tags → song vs score), Stage 2 music.py
* reaction kinds (Laughter, Giggle, Screaming, Shout, Gasp, Crying), Stage 3 peaks
* speech presence sanity signal

Windows of ``WIN_S`` seconds every ``HOP_S`` seconds; probabilities for a fixed subset of classes.
Output ``audio_tags.json``::

    {"win_s": 5.0, "hop_s": 2.5, "t": [2.5, 5.0, ...], "classes": ["Speech", ...],
     "probs": {"Speech": [..], "Music": [..], ...}}

The CNN14 checkpoint (~330 MB) is downloaded once by panns_inference into ``~/panns_data``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import SR, load_wav

WIN_S = 5.0
HOP_S = 2.5
PANNS_SR = 32000

CLASSES = [
    "Speech", "Narration, monologue", "Conversation", "Music", "Singing", "Vocal music",
    "Background music", "Theme music", "Soundtrack music", "Pop music", "Rock music", "Hip hop music",
    "Rhythm and blues", "Country", "Electronic music", "Orchestra", "Choir",
    "Laughter", "Giggle", "Chuckle, chortle", "Belly laugh", "Screaming", "Shout", "Yell", "Gasp",
    "Crying, sobbing", "Whimper", "Applause", "Cheering", "Whistling", "Silence",
    "Explosion", "Gunshot, gunfire", "Vehicle", "Rain", "Wind",
]

SONG_CLASSES = ["Singing", "Vocal music", "Pop music", "Rock music", "Hip hop music", "Rhythm and blues", "Country", "Choir"]
SCORE_CLASSES = ["Soundtrack music", "Theme music", "Background music", "Orchestra"]
LAUGH_CLASSES = ["Laughter", "Giggle", "Chuckle, chortle", "Belly laugh"]
SHOUT_CLASSES = ["Screaming", "Shout", "Yell"]


def compute_audio_tags(
    wav_path: str | Path,
    out: str | Path,
    *,
    t0: float | None = None,
    t1: float | None = None,
    device: str = "cpu",
    force: bool = False,
    progress: Callable[[float, str], None] | None = None,
) -> Path:
    out = Path(out)
    if out.exists() and not force:
        return out
    import librosa
    from panns_inference import AudioTagging, labels as panns_labels

    wav, sr = load_wav(wav_path, t0, t1)
    assert sr == SR
    wav32 = librosa.resample(wav, orig_sr=SR, target_sr=PANNS_SR).astype(np.float32)
    win, hop = int(WIN_S * PANNS_SR), int(HOP_S * PANNS_SR)
    starts = np.arange(0, max(1, len(wav32) - win + 1), hop)
    idx = [panns_labels.index(c) for c in CLASSES]
    tagger = AudioTagging(checkpoint_path=None, device=device)
    probs = np.zeros((len(starts), len(CLASSES)), dtype=np.float32)
    B = 32
    for i in range(0, len(starts), B):
        batch = np.stack([wav32[s:s + win] for s in starts[i:i + B]])
        clip, _ = tagger.inference(batch)
        probs[i:i + len(batch)] = np.asarray(clip)[:, idx]
        if progress:
            progress(min(1.0, (i + B) / len(starts)), f"audio tags {min(len(starts), i+B)}/{len(starts)} windows")
    centres = (t0 or 0.0) + (starts + win / 2) / PANNS_SR
    data = {"win_s": WIN_S, "hop_s": HOP_S, "range": [t0, t1] if (t0 is not None or t1 is not None) else None,
            "t": [round(float(c), 2) for c in centres], "classes": CLASSES,
            "probs": {c: [round(float(v), 3) for v in probs[:, j]] for j, c in enumerate(CLASSES)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data) + "\n", encoding="utf-8")
    return out


class AudioTags:
    def __init__(self, path: str | Path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.t = np.asarray(d["t"], dtype=np.float64)
        self.probs = {c: np.asarray(v, dtype=np.float32) for c, v in d["probs"].items()}
        self.win_s, self.hop_s = float(d["win_s"]), float(d["hop_s"])

    def at(self, cls: str, times: np.ndarray) -> np.ndarray:
        """Nearest-window probability of a class at arbitrary times."""
        p = self.probs.get(cls)
        if p is None or self.t.size == 0:
            return np.zeros(len(times), dtype=np.float32)
        i = np.clip(np.searchsorted(self.t, times), 0, len(self.t) - 1)
        j = np.clip(i - 1, 0, len(self.t) - 1)
        pick = np.where(np.abs(self.t[i] - times) <= np.abs(self.t[j] - times), i, j)
        return p[pick]

    def group_max(self, classes: list[str], times: np.ndarray) -> np.ndarray:
        arrs = [self.at(c, times) for c in classes if c in self.probs]
        return np.max(np.stack(arrs), axis=0) if arrs else np.zeros(len(times), dtype=np.float32)
