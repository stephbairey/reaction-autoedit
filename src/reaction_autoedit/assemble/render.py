"""Stage 4/5 — render an EDL to a video.

Strategy: every segment is encoded to its own intermediate clip (identical codec parameters), then the
clips are joined losslessly with ffmpeg's concat demuxer. This is simple, parallel (thread pool), and
resumable (intermediates are content-addressed by a hash of everything that affects them).

Transitions: ``cut`` is a hard cut. ``xfade`` is approximated as a short dip-to-black on both sides
of the join (true cross-dissolves need overlapping media which the concat approach doesn't give;
revisit if the rhythm needs it).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from .. import ffmpeg
from ..compute import ComputeProfile, detect
from ..config import ReactorConfig, RenderTarget, TitleConfig
from ..edl import EDL, Overlay, Segment
from ..models import Geometry
from . import layouts, package

def _file_sig(path: str | Path) -> str:
    """Cheap content signature (size+mtime) so cached clips of images re-encode when the file changes."""
    try:
        st = Path(path).stat()
        return f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return "missing"


XFADE_DUR = 0.25
AUDIO_FADE = 0.04
AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

console = Console()


@dataclass
class RenderResult:
    output: Path
    edl_copy: Path
    chapters: Path | None
    description: Path | None
    duration: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Plan:
    seg: Segment
    offset: float                     # output-timeline start
    overlays: list[tuple[Overlay, float]]  # (overlay, start relative to segment)
    fade_in: bool
    fade_out: bool


def _plan(edl: EDL) -> list[_Plan]:
    plans: list[_Plan] = []
    offs = edl.offsets()
    n = len(edl.segments)
    for i, (s, off) in enumerate(zip(edl.segments, offs)):
        ovs = []
        for o in edl.overlays:
            rel = o.at - off
            if rel < s.dur and rel + o.dur > 0:
                ovs.append((o, rel))
        fade_in = s.transition == "xfade" and i > 0
        fade_out = i + 1 < n and edl.segments[i + 1].transition == "xfade"
        plans.append(_Plan(seg=s, offset=off, overlays=ovs, fade_in=fade_in, fade_out=fade_out))
    return plans


def _seg_hash(p: _Plan, geom: Geometry, target: RenderTarget, reactor: ReactorConfig,
              profile: ComputeProfile, preview: bool, source: str) -> str:
    key = {
        "seg": p.seg.model_dump(by_alias=True, exclude={"score", "note", "tags", "chapter", "kind"}),
        "fade": [p.fade_in, p.fade_out],
        "ov": [(o.model_dump(), rel, _file_sig(o.template or reactor.branding.lower_third or "")) for o, rel in p.overlays],
        "geom": geom.model_dump(exclude={"confidence", "notes", "source"}),
        "target": target.model_dump(),
        "style": {"pip": reactor.pip.model_dump(), "rl": reactor.reactor_large.model_dump(),
                  "lt": reactor.branding.lower_third},
        "enc": profile.encoder_args(preview),
        "src": source,
        "v": 3,
    }
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _segment_cmd(p: _Plan, src: str, out: Path, geom: Geometry, target: RenderTarget,
                 reactor: ReactorConfig, profile: ComputeProfile, preview: bool) -> list[str]:
    s = p.seg
    dur = s.dur
    args = ["-ss", f"{s.in_:.3f}", "-t", f"{dur:.3f}", "-i", src]
    graph = layouts.build(s.layout, geom, target, reactor.pip, reactor.reactor_large)
    vlabel = "[vout]"
    extra_inputs = 0

    # lower-third overlays intersecting this segment
    lt_path = reactor.branding.lower_third
    for k, (o, rel) in enumerate(p.overlays):
        tpl = o.template or lt_path
        if not tpl or not Path(tpl).exists():
            continue
        extra_inputs += 1
        args += ["-loop", "1", "-framerate", f"{target.fps}", "-i", tpl]
        a, b = max(0.0, rel), min(dur, rel + o.dur)
        graph += (f";[{extra_inputs}:v]scale={target.w}:-2:flags=bicubic,format=yuva420p[lt{k}]"
                  f";{vlabel}[lt{k}]overlay=0:H-h:enable='between(t,{a:.3f},{b:.3f})':format=auto[vlt{k}]")
        vlabel = f"[vlt{k}]"

    # dip-to-black approximations for xfade
    fades = []
    if p.fade_in:
        fades.append(f"fade=t=in:st=0:d={XFADE_DUR}")
    if p.fade_out:
        fades.append(f"fade=t=out:st={max(0.0, dur - XFADE_DUR):.3f}:d={XFADE_DUR}")
    tail = ",".join(fades + [f"fps={target.fps}", "format=yuv420p"])
    graph += f";{vlabel}{tail}[vfinal]"

    afade = (f"afade=t=in:st=0:d={AUDIO_FADE},"
             f"afade=t=out:st={max(0.0, dur - AUDIO_FADE):.3f}:d={AUDIO_FADE},"
             f"aresample=48000")
    graph += f";[0:a]{afade}[afinal]"

    args += ["-filter_complex", graph, "-map", "[vfinal]", "-map", "[afinal]"]
    args += profile.encoder_args(preview) + ["-pix_fmt", "yuv420p", "-g", str(int(target.fps * 2))]
    args += AUDIO_ARGS
    args += ["-shortest", "-avoid_negative_ts", "make_zero", "-y", str(out)]
    return args


def _card_cmd(template: str, dur: float, out: Path, target: RenderTarget,
              profile: ComputeProfile, preview: bool, fade_out: bool = True) -> list[str]:
    fades = f"fade=t=in:st=0:d=0.45"
    if fade_out:
        fades += f",fade=t=out:st={max(0.0, dur - 0.45):.2f}:d=0.45"
    graph = (f"[0:v]scale={target.w}:{target.h}:force_original_aspect_ratio=decrease,"
             f"pad={target.w}:{target.h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
             f"{fades},fps={target.fps},format=yuv420p[v]")
    return (["-loop", "1", "-framerate", f"{target.fps}", "-t", f"{dur:.3f}", "-i", template,
             "-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
             "-filter_complex", graph, "-map", "[v]", "-map", "1:a"]
            + profile.encoder_args(preview) + ["-pix_fmt", "yuv420p", "-g", str(int(target.fps * 2))]
            + AUDIO_ARGS + ["-shortest", "-y", str(out)])


def render(
    edl: EDL,
    geom: Geometry,
    *,
    out: Path,
    reactor: ReactorConfig | None = None,
    title: TitleConfig | None = None,
    preview: bool = False,
    jobs: int | None = None,
    force: bool = False,
    keep_tmp: bool = True,
    profile: ComputeProfile | None = None,
    source_duration: float | None = None,
) -> RenderResult:
    reactor = reactor or ReactorConfig()
    title = title or TitleConfig()
    profile = profile or detect()
    target = RenderTarget.preview() if preview else edl.target
    src = edl.source
    if not Path(src).exists():
        raise FileNotFoundError(f"EDL source not found: {src}")
    warnings = edl.validate_rules(clip_cap_s=title.clip_cap_s, source_duration=source_duration)
    for w in warnings:
        console.print(f"[yellow]warn:[/] {w}")
    if not edl.segments:
        raise ValueError("EDL has no segments")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (".tmp_" + out.stem)
    tmp.mkdir(exist_ok=True)

    plans = _plan(edl)
    jobs = jobs or profile.render_jobs
    tasks: list[tuple[int, Path, list[str]]] = []
    clip_paths: list[Path] = []
    for i, p in enumerate(plans):
        h = _seg_hash(p, geom, target, reactor, profile, preview, src)
        clip = tmp / f"seg_{i:04d}_{p.seg.id}_{h}.mp4"
        clip_paths.append(clip)
        if force or not clip.exists():
            tasks.append((i, clip, _segment_cmd(p, src, clip, geom, target, reactor, profile, preview)))

    # endcard
    endcard_clip: Path | None = None
    if edl.endcard is not None:
        tpl = edl.endcard.template or reactor.branding.endcard
        if tpl and Path(tpl).exists():
            key = hashlib.sha1(f"{tpl}{_file_sig(tpl)}{edl.endcard.dur}{target}{profile.encoder_args(preview)}".encode()).hexdigest()[:10]
            endcard_clip = tmp / f"endcard_{key}.mp4"
            if force or not endcard_clip.exists():
                tasks.append((len(plans), endcard_clip, _card_cmd(tpl, edl.endcard.dur, endcard_clip, target, profile, preview, fade_out=False)))
        else:
            warnings.append(f"endcard template missing ({tpl}); skipped")
            console.print(f"[yellow]warn:[/] endcard template missing ({tpl}); skipped")

    # full-frame cards (e.g. movie title card after the intro)
    card_inserts: list[tuple[int, Path]] = []      # (index into clip_paths to insert BEFORE, clip)
    for ci, card in enumerate(edl.cards):
        if not Path(card.template).exists():
            warnings.append(f"card template missing ({card.template}); skipped")
            console.print(f"[yellow]warn:[/] card template missing ({card.template}); skipped")
            continue
        pos = 0
        if card.before_id is not None:
            pos = next((i for i, pl in enumerate(plans) if pl.seg.id == card.before_id), 0)
        key = hashlib.sha1(f"{card.template}{_file_sig(card.template)}{card.dur}{target}{profile.encoder_args(preview)}".encode()).hexdigest()[:10]
        clip = tmp / f"card_{ci}_{key}.mp4"
        card_inserts.append((pos, clip))
        if force or not clip.exists():
            tasks.append((len(plans) + 1 + ci, clip, _card_cmd(card.template, card.dur, clip, target, profile, preview)))

    console.print(f"rendering {len(plans)} segments ({len(tasks)} to encode, {len(plans) - len([t for t in tasks if t[0] < len(plans)])} cached) "
                  f"→ {target.w}x{target.h}@{target.fps:g} via {profile.video_encoder}, {jobs} jobs")
    failures: list[str] = []
    with Progress(TextColumn("[bold]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), console=console) as prog:
        t = prog.add_task("encode", total=len(tasks))
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            futs = {ex.submit(_encode_one, cmd, clip): (i, clip) for i, clip, cmd in tasks}
            for f in as_completed(futs):
                i, clip = futs[f]
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    failures.append(f"segment {i}: {e}")
                prog.advance(t)
    if failures:
        raise RuntimeError("render failed:\n" + "\n".join(failures[:5]))

    # concat
    lst = tmp / "concat.txt"
    seq: list[Path] = list(clip_paths)
    for pos, clip in sorted(card_inserts, key=lambda x: -x[0]):
        if clip.exists():
            seq.insert(pos, clip)
    if endcard_clip:
        seq.append(endcard_clip)
    with lst.open("w", encoding="utf-8") as fh:
        for c in seq:
            fh.write(f"file '{c.resolve().as_posix()}'\n")
    ffmpeg.run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", "-y", str(out)])

    # sidecars
    edl_copy = out.with_suffix(".edl.json")
    edl.save(edl_copy)
    chapters = package.write_chapters(edl, out.with_suffix(".chapters.txt"))
    desc = package.write_description(edl, reactor, title, out.with_suffix(".description.txt"))
    package.write_midrolls(edl, out.with_suffix(".midrolls.txt"))
    total = edl.duration + (edl.endcard.dur if endcard_clip and edl.endcard else 0.0) + sum(c.dur for c in edl.cards)
    if not keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    console.print(f"[green]done:[/] {out} ({total/60:.1f} min)")
    return RenderResult(output=out, edl_copy=edl_copy, chapters=chapters, description=desc, duration=total, warnings=warnings)


def _encode_one(cmd: list[str], clip: Path) -> None:
    part = clip.with_suffix(".part.mp4")
    cmd = cmd[:-1] + [str(part)]  # replace output path
    ffmpeg.run(cmd)
    part.replace(clip)
