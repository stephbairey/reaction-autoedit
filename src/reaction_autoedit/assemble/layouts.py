"""ffmpeg filtergraph builders for the two output layouts.

Both layouts are pure crop/scale/overlay operations on the composite frame (input ``[0:v]``):

- ``movie-large``  : movie active picture fills the frame (letter/pillar-boxed), facecam as small PiP.
- ``reactor-large``: facecam blown up and centred over a blurred+darkened movie background
                     (optionally with the movie as a small PiP).

Every builder returns a filtergraph string that consumes ``[0:v]`` and produces ``[vout]``.
"""

from __future__ import annotations

from ..config import PipStyle, ReactorLargeStyle, RenderTarget
from ..models import Geometry, Rect


def _even(v: float) -> int:
    n = int(round(v))
    return n - (n % 2)


def _pip_size(face: Rect, target: RenderTarget, style: PipStyle) -> tuple[int, int]:
    pw = target.w * style.width_frac
    ph = pw / max(face.aspect, 1e-6)
    if ph > 0.45 * target.h:
        ph = 0.45 * target.h
        pw = ph * face.aspect
    return _even(pw), _even(ph)


def _corner_xy(corner: str, w_expr: str, h_expr: str, target: RenderTarget, margin: int) -> tuple[str, str]:
    m = margin
    W, H = target.w, target.h
    x = f"{m}" if corner.endswith("left") else f"{W}-{w_expr}-{m}"
    y = f"{m}" if corner.startswith("top") else f"{H}-{h_expr}-{m}"
    return x, y


def _circle_mask() -> str:
    """Make the current stream's alpha a centred circle (for circular facecams)."""
    return ("format=yuva420p,"
            "geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':"
            "a='if(lte(hypot(X-W/2+0.5,Y-H/2+0.5),min(W,H)/2),255,0)'")


def movie_large(geom: Geometry, target: RenderTarget, pip: PipStyle) -> str:
    W, H = target.w, target.h
    inner, face = geom.movie_inner.even(), geom.face.even()
    pw, ph = _pip_size(face, target, pip)
    circle = pip.circle if pip.circle is not None else (geom.face.shape == "circle")
    parts = [
        "[0:v]split=2[m0][f0]",
        f"[m0]{inner.ffmpeg_crop()},scale={W}:{H}:force_original_aspect_ratio=decrease:flags=bicubic,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[bg]",
    ]
    fchain = f"[f0]{face.ffmpeg_crop()},scale={pw}:{ph}:flags=bicubic"
    if circle:
        fchain += "," + _circle_mask()
    if pip.border_px > 0 and not circle:
        b = pip.border_px
        fchain += f",pad=iw+{2*b}:ih+{2*b}:{b}:{b}:color=white"
    parts.append(fchain + "[pip]")
    x, y = _corner_xy(pip.corner, "overlay_w", "overlay_h", target, pip.margin_px)
    parts.append(f"[bg][pip]overlay={x}:{y}:format=auto,format=yuv420p[vout]")
    return ";".join(parts)


def reactor_large(geom: Geometry, target: RenderTarget, style: ReactorLargeStyle, pip: PipStyle) -> str:
    W, H = target.w, target.h
    inner, face = geom.movie_inner.even(), geom.face.even()
    scale_f = W / 1920.0
    n_splits = 3 if style.movie_pip else 2
    labels = "[m0][f0]" + ("[m1]" if style.movie_pip else "")
    parts = [f"[0:v]split={n_splits}{labels}"]

    # background
    if style.background == "blur":
        r = max(2, int(style.blur_strength * scale_f))
        bg = (f"[m0]{inner.ffmpeg_crop()},scale={W}:{H}:force_original_aspect_ratio=increase:flags=bilinear,"
              f"crop={W}:{H},boxblur=luma_radius={r}:luma_power=2:chroma_radius={max(1, r // 2)}:chroma_power=2")
        if style.darken > 0:
            bg += f",lutyuv=y='val*{1.0 - style.darken:.3f}'"
        if style.vignette:
            bg += ",vignette=angle=PI/4.5"
        parts.append(bg + ",setsar=1[bg]")
    else:
        parts.append(f"[m0]{inner.ffmpeg_crop()},scale={W}:{H},drawbox=c=black:t=fill,setsar=1[bg]")

    # face: scale to face_height_frac of H, but never wider than 0.9 W
    fh = _even(H * style.face_height_frac)
    fw = _even(fh * face.aspect)
    if fw > 0.9 * W:
        fw = _even(0.9 * W)
        fh = _even(fw / max(face.aspect, 1e-6))
    fchain = f"[f0]{face.ffmpeg_crop()},scale={fw}:{fh}:flags=lanczos"
    if style.sharpen > 0:
        fchain += f",unsharp=5:5:{style.sharpen:.2f}:5:5:0"
    if geom.face.shape == "circle":
        fchain += "," + _circle_mask()
    parts.append(fchain + "[face]")
    parts.append(f"[bg][face]overlay=(W-w)/2:(H-h)/2:format=auto[v1]")

    if style.movie_pip:
        pw = _even(W * pip.width_frac)
        ph = _even(pw / max(inner.aspect, 1e-6))
        parts.append(f"[m1]{inner.ffmpeg_crop()},scale={pw}:{ph}:flags=bicubic[mpip]")
        x, y = _corner_xy(pip.corner, "overlay_w", "overlay_h", target, pip.margin_px)
        parts.append(f"[v1][mpip]overlay={x}:{y}:format=auto,format=yuv420p[vout]")
    else:
        parts.append("[v1]format=yuv420p[vout]")
    return ";".join(parts)


def build(layout: str, geom: Geometry, target: RenderTarget, pip: PipStyle, rl: ReactorLargeStyle) -> str:
    if layout == "movie-large":
        return movie_large(geom, target, pip)
    if layout == "reactor-large":
        return reactor_large(geom, target, rl, pip)
    raise ValueError(f"unknown layout {layout!r}")
