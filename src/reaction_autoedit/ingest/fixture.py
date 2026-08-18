"""Synthetic composite fixtures with known geometry, so tests run without real footage.

Presets:
- ``sbs``  — mirrors the real sample: reactor cam fills the left column, movie in a 16:9 box bottom-right.
- ``pip``  — movie letterboxed across the full frame, circular facecam PiP bottom-right (overlapping the movie).
"""

from __future__ import annotations

from pathlib import Path

from .. import ffmpeg
from ..config import dump_json
from ..models import FaceRegion, FrameInfo, Geometry, Rect

PRESETS = ("sbs", "pip")


def fixture_geometry(preset: str, w: int = 1920, h: int = 1080, fps: float = 30.0) -> Geometry:
    if preset == "sbs":
        movie = Rect(x=732, y=424, w=1140, h=640)
        # 2.39:1 picture letterboxed inside the 16:9 box
        ih = int(round(1140 / 2.39)) // 2 * 2
        inner = Rect(x=732, y=424 + (640 - ih) // 2, w=1140, h=ih)
        face = FaceRegion(x=0, y=232, w=708, h=848, shape="rect")
    elif preset == "pip":
        movie = Rect(x=0, y=0, w=1920, h=1080)
        inner = Rect(x=0, y=140, w=1920, h=800)
        face = FaceRegion(x=1500, y=680, w=360, h=360, shape="circle")
    else:
        raise ValueError(f"unknown preset {preset!r}; choose from {PRESETS}")
    return Geometry(frame=FrameInfo(w=w, h=h, fps=fps), movie=movie, movie_inner=inner, face=face,
                    confidence=1.0, source="fixture")


def make_fixture(out_dir: str | Path, preset: str = "sbs", duration: float = 60.0, fps: float = 30.0,
                 w: int = 1920, h: int = 1080) -> tuple[Path, Path]:
    """Render ``<out_dir>/fixture_<preset>.mp4`` and its truth geometry JSON. Returns (video, json)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = fixture_geometry(preset, w, h, fps)
    m, i, f = g.movie, g.movie_inner, g.face
    video = out_dir / f"fixture_{preset}.mp4"
    truth = out_dir / f"fixture_{preset}.geometry.json"

    parts = [
        f"color=c=0x1e1e22:s={w}x{h}:r={fps}:d={duration}[bg]",
        # movie box: black bars + moving picture inside
        f"color=c=black:s={m.w}x{m.h}:r={fps}:d={duration}[mbox]",
        f"testsrc2=s={i.w}x{i.h}:r={fps}:d={duration},scroll=h=0.02:v=0.01[pic]",
        f"[mbox][pic]overlay={i.x - m.x}:{i.y - m.y}:shortest=1[movie]",
        f"[bg][movie]overlay={m.x}:{m.y}:shortest=1[v1]",
    ]
    # reactor cam: classic testsrc, tinted, slowly scrolling so the whole region has motion
    face_src = f"testsrc=s={f.w}x{f.h}:r={fps}:d={duration},hue=s=0.4,scroll=v=0.015"
    if f.shape == "circle":
        r = f.w / 2
        parts.append(
            face_src + ",format=yuva420p,"
            f"geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='if(lte(hypot(X-{r},Y-{r}),{r*0.985}),255,0)'[cam]"
        )
    else:
        parts.append(face_src + "[cam]")
    parts.append(f"[v1][cam]overlay={f.x}:{f.y}:shortest=1,format=yuv420p[vout]")
    # audio: quiet tone + noise so audio filters have something to chew on
    parts.append(f"sine=frequency=220:sample_rate=48000:d={duration},volume=0.2[a1]")
    parts.append(f"anoisesrc=color=pink:sample_rate=48000:d={duration},volume=0.05[a2]")
    parts.append("[a1][a2]amix=inputs=2:duration=shortest[aout]")

    ffmpeg.run([
        "-filter_complex", ";".join(parts),
        "-map", "[vout]", "-map", "[aout]",
        "-t", f"{duration}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-y", str(video),
    ], timeout=600)
    dump_json(g, truth)
    return video, truth
