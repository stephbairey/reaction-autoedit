"""Dead-air map: long stretches where nothing happens — quiet audio, still reactor, no REACTOR speech.

Selection uses these as compression candidates (they are the first thing to cut when tightening
runtime) and mid-roll placement avoids the middle of them (a break in a lull is fine, but a break
inside a 40 s silent stare reads oddly).

Output ``deadair.json``::  {"spans": [{"t0": 100.0, "t1": 131.5, "dur": 31.5}], "total_s": 812.0}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .face_motion import FaceMotion


def detect_dead_air(speakers_path: str | Path, out: str | Path, *, face_motion: FaceMotion | None = None,
                    min_dur: float = 15.0, bridge_s: float = 1.5, force: bool = False) -> Path:
    out = Path(out)
    if out.exists() and not force:
        return out
    d = json.loads(Path(speakers_path).read_text(encoding="utf-8"))
    tl = d["timeline"]
    key, thr = d.get("score_key", "sim"), float(d["threshold"])
    t = np.array([w["t"] for w in tl], dtype=np.float64)
    db = np.array([w["db"] for w in tl], dtype=np.float64)
    sc = np.array([np.nan if w.get(key) is None else w[key] for w in tl], dtype=np.float64)
    reactor = np.nan_to_num(sc, nan=-9) >= thr
    quiet = db < (np.median(db) - 6.0)
    still = np.ones(len(t), dtype=bool)
    if face_motion is not None:
        m = face_motion.per_window(t)
        still = np.nan_to_num(m, nan=0.0) < 0.6 * np.nanmedian(m)
    dead = quiet & still & ~reactor
    spans: list[dict] = []
    cur = None
    for i, flag in enumerate(dead):
        a, b = float(t[i] - 0.25), float(t[i] + 0.25)
        if flag:
            if cur and a - cur["t1"] <= bridge_s:
                cur["t1"] = b
            else:
                cur = {"t0": a, "t1": b}
                spans.append(cur)
    result = [{"t0": round(s["t0"], 2), "t1": round(s["t1"], 2), "dur": round(s["t1"] - s["t0"], 1)}
              for s in spans if s["t1"] - s["t0"] >= min_dur]
    data = {"min_dur": min_dur, "spans": result, "total_s": round(sum(r["dur"] for r in result), 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return out
