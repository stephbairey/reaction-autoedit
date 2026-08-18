"""One cheap decode pass over the composite → several per-frame video signals.

The source is decoded once at ``fps`` (default 5) to a small greyscale thumbnail (320 px wide) and
we compute, per frame:

* **face motion** — mean abs frame difference in the reactor's mouth/jaw region → ``face_motion.json``
  (same schema as face_motion.py, which this supersedes for the full pipeline)
* **movie change** — mean abs frame difference over the movie's active picture → scene cuts
  (``scenes.json``: adaptive threshold on the change signal, minimum gap between cuts)
* **movie brightness** — mean luma of the movie region (kept in scenes.json; useful for fades)

Everything is model-free and CPU-trivial; the decode is the only real cost (~10-15 min for a
90-min 1080p60 file on a laptop CPU).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from .. import ffmpeg
from ..models import Geometry, Rect

THUMB_W = 320


def _scaled(r: Rect, sx: float, sy: float, W: int, H: int) -> tuple[int, int, int, int]:
    x0 = int(max(0, min(W - 1, round(r.x * sx))))
    y0 = int(max(0, min(H - 1, round(r.y * sy))))
    x1 = int(max(x0 + 1, min(W, round((r.x + r.w) * sx))))
    y1 = int(max(y0 + 1, min(H, round((r.y + r.h) * sy))))
    return x0, y0, x1, y1


def compute_video_signals(
    src: str | Path,
    geom: Geometry,
    face_motion_out: str | Path,
    scenes_out: str | Path,
    *,
    fps: float = 5.0,
    t0: float | None = None,
    t1: float | None = None,
    duration: float | None = None,
    force: bool = False,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[Path, Path]:
    face_motion_out, scenes_out = Path(face_motion_out), Path(scenes_out)
    if face_motion_out.exists() and scenes_out.exists() and not force:
        return face_motion_out, scenes_out
    fw, fh = geom.frame.w, geom.frame.h
    W = THUMB_W
    H = max(2, int(round(fh * W / fw)))
    H -= H % 2
    sx, sy = W / fw, H / fh
    f = geom.face
    mouth = Rect(x=f.x + int(f.w * 0.25), y=f.y + int(f.h * 0.25), w=int(f.w * 0.5), h=int(f.h * 0.30))
    mx0, my0, mx1, my1 = _scaled(mouth, sx, sy, W, H)
    vx0, vy0, vx1, vy1 = _scaled(geom.movie_inner, sx, sy, W, H)

    args = [ffmpeg.ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if t0:
        args += ["-ss", f"{t0:.3f}"]
    args += ["-i", str(src)]
    if t1 is not None:
        args += ["-t", f"{t1 - (t0 or 0.0):.3f}"]
    args += ["-vf", f"fps={fps},scale={W}:{H}:flags=area,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-an", "-"]
    frame_bytes = W * H
    total_frames = ((t1 or duration or 0.0) - (t0 or 0.0)) * fps if (t1 or duration) else None
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 64)
    assert proc.stdout is not None
    face_vals: list[float] = []
    movie_vals: list[float] = []
    lum_vals: list[float] = []
    prev_face = prev_movie = None
    i = 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        fr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W).astype(np.float32)
        face = fr[my0:my1, mx0:mx1]
        mov = fr[vy0:vy1, vx0:vx1]
        face_vals.append(0.0 if prev_face is None else float(np.mean(np.abs(face - prev_face))))
        movie_vals.append(0.0 if prev_movie is None else float(np.mean(np.abs(mov - prev_movie))))
        lum_vals.append(float(mov.mean()))
        prev_face, prev_movie = face, mov
        i += 1
        if progress and total_frames and i % 500 == 0:
            progress(min(1.0, i / total_frames), f"video signals {i/fps/60:.1f}/{total_frames/fps/60:.1f} min")
    proc.wait()
    if not face_vals:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise ffmpeg.FFmpegError(f"video decode failed: {err[-2000:]}")

    off = t0 or 0.0
    face_motion_out.parent.mkdir(parents=True, exist_ok=True)
    face_motion_out.write_text(json.dumps({"fps": fps, "t0": off, "region": mouth.model_dump(),
                                           "values": [round(v, 2) for v in face_vals]}) + "\n", encoding="utf-8")
    cuts = detect_cuts(np.asarray(movie_vals), fps, off)
    scenes_out.write_text(json.dumps({"fps": fps, "t0": off, "cuts": cuts,
                                      "movie_change": [round(v, 2) for v in movie_vals],
                                      "movie_luma": [round(v, 1) for v in lum_vals]}) + "\n", encoding="utf-8")
    return face_motion_out, scenes_out


def detect_cuts(change: np.ndarray, fps: float, t0: float = 0.0, *, abs_thr: float = 14.0,
                rel_k: float = 3.0, min_gap_s: float = 0.6, window_s: float = 20.0) -> list[float]:
    """Scene cuts = spikes in the movie change signal well above the local level.
    Rolling median/MAD over ``window_s`` gives the local baseline; a frame is a cut if
    change > max(abs_thr, median + rel_k * MAD*1.4826 + 4) and it is the local maximum."""
    n = len(change)
    if n < 3:
        return []
    w = max(3, int(window_s * fps))
    cuts: list[float] = []
    last = -1e9
    # rolling stats via cumulative approach on a padded array (approximate: use uniform windows)
    med = np.zeros(n)
    mad = np.zeros(n)
    half = w // 2
    for i in range(0, n, half):
        a, b = max(0, i - half), min(n, i + half)
        seg = change[a:b]
        m = float(np.median(seg))
        d = float(np.median(np.abs(seg - m))) * 1.4826
        med[i:i + half] = m
        mad[i:i + half] = d
    thr = np.maximum(abs_thr, med + rel_k * mad + 4.0)
    for i in range(1, n - 1):
        v = change[i]
        if v >= thr[i] and v >= change[i - 1] and v >= change[i + 1]:
            t = t0 + i / fps
            if t - last >= min_gap_s:
                cuts.append(round(t, 2))
                last = t
    return cuts


class Scenes:
    def __init__(self, path: str | Path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.cuts = np.asarray(d["cuts"], dtype=np.float64)
        self.fps = float(d["fps"])
        self.t0 = float(d["t0"])
        self.change = np.asarray(d.get("movie_change", []), dtype=np.float32)

    def nearest_cut(self, t: float, max_dist: float = 1.5) -> float | None:
        if self.cuts.size == 0:
            return None
        i = int(np.argmin(np.abs(self.cuts - t)))
        return float(self.cuts[i]) if abs(self.cuts[i] - t) <= max_dist else None

    def cuts_between(self, a: float, b: float) -> list[float]:
        return [float(c) for c in self.cuts if a <= c <= b]
