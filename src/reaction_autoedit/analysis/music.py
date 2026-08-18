"""Music detection on the film audio (tiered risk). [M3]

librosa features (spectral flatness, harmonic ratio, tempo stability, onset regularity) → music vs
speech per 1 s window; smoothing; spans of music get ``kind = "song"`` when a vocal-presence
heuristic fires (harmonic energy in the vocal band with pitch continuity while no REACTOR speech is
tagged), else ``"score"``. Songs are the block-happy layer and are excluded by selection; score is
minimised but allowed inside the clip cap.
Output analysis/music.json: [{"t0": 100.0, "t1": 130.5, "kind": "song", "conf": 0.7}]
"""


def detect_music(*args, **kwargs):  # pragma: no cover - M3
    raise NotImplementedError("Stage 2 music detection lands in Milestone 3")
