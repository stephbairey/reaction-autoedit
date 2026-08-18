"""Reaction peak detection. [M3]

Signals (all cheap):
  * vocal energy (RMS) spikes restricted to REACTOR-tagged spans (from speakers.json)
  * laughter / gasp / shout heuristics (spectral flatness + pitch variance) — a small classifier later
  * facial expression deltas: sample the facecam region at ~1 fps, landmark/expression model
    (or a plain frame-difference energy fallback), z-score the change signal
Output analysis/peaks.json: [{"t": 1234.5, "dur": 4.0, "score": 0.83, "kind": "laugh"}]
"""


def detect_peaks(*args, **kwargs):  # pragma: no cover - M3
    raise NotImplementedError("Stage 2 peak detection lands in Milestone 3")
