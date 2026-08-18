"""Facecam motion signal: how much the reactor's mouth/jaw region changes frame to frame.

Cheap, model-free, and surprisingly informative: it separates "he is talking / laughing / reacting"
from "he is sitting still watching" and is fused with the audio speaker score (Stage 2) and used for
reaction peaks (Stage 3). One sequential ffmpeg decode of the source at ``fps`` (default 5), cropped
to the lower-middle of the detected face region and downscaled to a thumbnail; the per-frame mean
absolute difference is the signal.

Output ``face_motion.json``::

    {"fps": 5, "t0": 0.0, "region": {"x":..,"y":..,"w":..,"h":..}, "values": [0.0, 3.2, ...]}

``values[i]`` is the motion between frame i-1 and i, at time ``t0 + i / fps``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from .. import ffmpeg
from ..models import Geometry, Rect

THUMB_W, THUMB_H = 96, 64


def mouth_region(geom: Geometry) -> Rect:
    f = geom.face
    return Rect(x=f.x + int(f.w * 0.25), y=f.y + int(f.h * 0.25), w=int(f.w * 0.5), h=int(f.h * 0.30)).even()


def compute_face_motion(
    src: str | Path,
    geom: Geometry,
    out: str | Path,
    *,
    fps: float = 5.0,
    t0: float | None = None,
    t1: float | None = None,
    duration: float | None = None,
    force: bool = False,
    progress: Callable[[float, str], None] | None = None,
) -> Path:
    out = Path(out)
    if out.exists() and not force:
        return out
    r = mouth_region(geom)
    args = [ffmpeg.ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if t0:
        args += ["-ss", f"{t0:.3f}"]
    args += ["-i", str(src)]
    if t1 is not None:
        args += ["-t", f"{t1 - (t0 or 0.0):.3f}"]
    args += ["-vf", f"{r.ffmpeg_crop()},fps={fps},scale={THUMB_W}:{THUMB_H}:flags=area,format=gray",
             "-f", "rawvideo", "-pix_fmt", "gray", "-an", "-"]
    frame_bytes = THUMB_W * THUMB_H
    values: list[float] = []
    prev: np.ndarray | None = None
    total_frames = ((t1 or duration or 0.0) - (t0 or 0.0)) * fps if (t1 or duration) else None
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 64)
    assert proc.stdout is not None
    i = 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        cur = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        values.append(0.0 if prev is None else float(np.mean(np.abs(cur - prev))))
        prev = cur
        i += 1
        if progress and total_frames and i % 500 == 0:
            progress(min(1.0, i / total_frames), f"face motion {i/fps/60:.1f}/{total_frames/fps/60:.1f} min")
    proc.wait()
    if proc.returncode not in (0, None) and not values:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise ffmpeg.FFmpegError(f"face motion decode failed: {err[-2000:]}")
    data = {"fps": fps, "t0": t0 or 0.0, "region": r.model_dump(), "values": [round(v, 2) for v in values]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data) + "\n", encoding="utf-8")
    return out


class FaceMotion:
    """Query helper over a face_motion.json."""

    def __init__(self, path: str | Path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.fps = float(d["fps"])
        self.t0 = float(d["t0"])
        self.values = np.asarray(d["values"], dtype=np.float32)
        self.times = self.t0 + np.arange(len(self.values)) / self.fps

    def covers(self, a: float, b: float) -> bool:
        return self.times.size > 0 and self.times[0] <= a + 1.0 and self.times[-1] >= b - 1.0

    def window_mean(self, centre: float, half: float = 0.8) -> float:
        i0 = int(np.searchsorted(self.times, centre - half))
        i1 = int(np.searchsorted(self.times, centre + half))
        seg = self.values[i0:i1]
        return float(seg.mean()) if seg.size else float("nan")

    def per_window(self, centres: np.ndarray, half: float = 0.8) -> np.ndarray:
        return np.array([self.window_mean(c, half) for c in centres], dtype=np.float32)
