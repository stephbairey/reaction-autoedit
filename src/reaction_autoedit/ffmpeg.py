"""Thin wrappers around the ffmpeg / ffprobe binaries.

Binary lookup order: ``FFMPEG_BIN`` / ``FFPROBE_BIN`` env vars, then PATH, then ``~/.local/bin``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from .models import AudioInfo, ProbeInfo


class FFmpegError(RuntimeError):
    pass


def _find(name: str, env: str) -> str:
    cand = os.environ.get(env)
    if cand and Path(cand).exists():
        return cand
    found = shutil.which(name)
    if found:
        return found
    local = Path.home() / ".local" / "bin" / name
    if local.exists():
        return str(local)
    for ext in (".exe",):
        found = shutil.which(name + ext)
        if found:
            return found
    raise FFmpegError(
        f"{name} not found. Install ffmpeg (see README) or set {env} to the binary path."
    )


@lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    return _find("ffmpeg", "FFMPEG_BIN")


@lru_cache(maxsize=1)
def ffprobe_bin() -> str:
    return _find("ffprobe", "FFPROBE_BIN")


def available() -> bool:
    try:
        ffmpeg_bin()
        ffprobe_bin()
        return True
    except FFmpegError:
        return False


def run(args: list[str], *, check: bool = True, quiet: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run ffmpeg with the given args (without the leading binary). Raises FFmpegError on failure."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin"]
    if quiet:
        cmd += ["-loglevel", "error"]
    cmd += args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}):\n  {' '.join(cmd)}\n{proc.stderr[-4000:]}")
    return proc


def probe(path: str | Path) -> ProbeInfo:
    path = str(path)
    cmd = [
        ffprobe_bin(), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    fmt = data.get("format", {})
    vstreams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    astreams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not vstreams:
        raise FFmpegError(f"no video stream in {path}")
    v = vstreams[0]
    fps_str = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(fps_str))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(fmt.get("duration") or v.get("duration") or 0.0)
    audio = None
    if astreams:
        a = astreams[0]
        audio = AudioInfo(
            codec=a.get("codec_name", "?"),
            sample_rate=int(a.get("sample_rate") or 0),
            channels=int(a.get("channels") or 0),
        )
    br = fmt.get("bit_rate")
    return ProbeInfo(
        path=path,
        duration=duration,
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        video_codec=v.get("codec_name", "?"),
        bit_rate=int(br) if br else None,
        audio=audio,
    )


@lru_cache(maxsize=1)
def encoders() -> set[str]:
    proc = subprocess.run([ffmpeg_bin(), "-hide_banner", "-encoders"], capture_output=True, text=True)
    names: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] and parts[0][0] in "VAS" and len(parts[0]) == 6:
            names.add(parts[1])
    return names


def encoder_works(name: str) -> bool:
    """Try a tiny test encode; hardware encoders can be listed but unusable."""
    if name not in encoders():
        return False
    try:
        run([
            "-f", "lavfi", "-i", "color=c=black:s=256x256:r=30", "-t", "0.2",
            "-c:v", name, "-f", "null", "-",
        ], timeout=30)
        return True
    except (FFmpegError, subprocess.TimeoutExpired):
        return False


def extract_frame(src: str | Path, t: float, out: str | Path, width: int | None = None) -> None:
    vf = [f"scale={width}:-2"] if width else []
    args = ["-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1"]
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-y", str(out)]
    run(args)
