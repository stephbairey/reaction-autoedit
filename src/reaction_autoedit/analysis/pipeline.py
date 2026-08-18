"""Stage 2 orchestration: run the analysers for a project, with per-artifact caching.

Full-track artifacts live in ``work/<name>/analysis/``; a ``--range t0 t1`` run (fast iteration
during development) lives in ``analysis/r<t0>-<t1>/`` so it never masquerades as the full result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..compute import ComputeProfile, detect
from ..project import Project
from . import audio, transcribe as _tr

STEPS = ("transcribe", "video", "speakers", "tags", "music", "peaks", "deadair")
ALIASES = {"facemotion": "video"}


def analysis_dir(proj: Project, t0: float | None, t1: float | None) -> Path:
    if t0 is None and t1 is None:
        return proj.analysis_dir
    return proj.analysis_dir / f"r{int(t0 or 0)}-{int(t1) if t1 is not None else 'end'}"


def voice_samples(proj: Project, extra: list[Path] | None = None) -> list[Path]:
    paths: list[Path] = []
    vs = proj.reactor().voice_sample
    if vs:
        paths.append(Path(vs))
    paths += extra or []
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"voice sample(s) not found: {missing}")
    if not paths:
        raise RuntimeError("no voice sample: set `voice_sample` in the reactor config or pass --voice")
    return paths


def run(
    proj: Project,
    *,
    steps: tuple[str, ...] = STEPS,
    t0: float | None = None,
    t1: float | None = None,
    force: bool = False,
    model: str | None = None,
    profile: ComputeProfile | None = None,
    voice: list[Path] | None = None,
    backend: str = "auto",
    fusion_weight: float = 0.0,
    chunk_min: float = 10.0,
    log: Callable[[str], None] = print,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Path]:
    steps = tuple(ALIASES.get(x, x) for x in steps)
    profile = profile or detect()
    adir = analysis_dir(proj, t0, t1)
    adir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    wav = proj.analysis_dir / "audio16k.wav"
    if not wav.exists():
        log("extracting audio → analysis/audio16k.wav")
    out["audio"] = audio.extract_audio(proj.source, wav)

    if "transcribe" in steps:
        tp = adir / "transcript.json"
        if tp.exists() and not force:
            log(f"transcript cached: {tp}")
        else:
            log(f"transcribing with whisper {model or profile.whisper_model} on {profile.device} …")
            _tr.transcribe(wav, tp, t0=t0, t1=t1, model=model, force=force, profile=profile, progress=progress)
        out["transcript"] = tp
        proj.mark("transcribe", path=str(tp), range=[t0, t1])

    if "video" in steps:
        from .video_signals import compute_video_signals

        if proj.state.geometry is None:
            log("video signals skipped: no geometry yet (run `rae detect-layout`)")
        else:
            fm, sc = adir / "face_motion.json", adir / "scenes.json"
            if fm.exists() and sc.exists() and not force:
                log(f"video signals cached: {fm}, {sc}")
            else:
                log("video signals (face motion + scene cuts, one 5 fps decode) …")
                compute_video_signals(proj.source, proj.state.geometry, fm, sc, t0=t0, t1=t1,
                                      duration=proj.state.probe.duration if proj.state.probe else None,
                                      force=force, progress=progress)
            out["face_motion"], out["scenes"] = fm, sc
            proj.mark("video", face_motion=str(fm), scenes=str(sc), range=[t0, t1])

    if "speakers" in steps:
        from .face_motion import FaceMotion
        from .speakers import run_speakers

        tp = adir / "transcript.json"
        transcript = _tr.load_transcript(tp) if tp.exists() else {"segments": []}
        if not tp.exists():
            log("  (no transcript yet — timeline/spans only; segment labels will be empty)")
        sp = adir / "speakers.json"
        if sp.exists() and not force:
            log(f"speakers cached: {sp}")
        else:
            samples = voice_samples(proj, voice)
            log(f"speaker attribution (backend={backend}, enrol from {[s.name for s in samples]}) …")
            fm_path = adir / "face_motion.json"
            if not fm_path.exists() and (proj.analysis_dir / "face_motion.json").exists():
                fm_path = proj.analysis_dir / "face_motion.json"      # full-track motion covers any range
            fm = FaceMotion(fm_path) if fm_path.exists() else None
            if fm is None:
                log("  (no face_motion.json — audio-only scoring; run the facemotion step for better accuracy)")
            run_speakers(wav, transcript, sp, sample_paths=samples, t0=t0, t1=t1,
                         device=profile.device, force=force, face_motion=fm, backend=backend,
                         fusion_weight=fusion_weight, labels_path=proj.analysis_dir / "labels.json", chunk_min=chunk_min, progress=progress)
        out["speakers"] = sp
        proj.mark("speakers", path=str(sp), range=[t0, t1])

    if "tags" in steps:
        from .audio_tags import compute_audio_tags

        tg = adir / "audio_tags.json"
        if tg.exists() and not force:
            log(f"audio tags cached: {tg}")
        else:
            log("audio tags (PANNs CNN14: music / singing / laughter / shout / gasp …) …")
            compute_audio_tags(wav, tg, t0=t0, t1=t1, device=profile.device, force=force, progress=progress)
        out["audio_tags"] = tg
        proj.mark("tags", path=str(tg), range=[t0, t1])

    if "music" in steps:
        from .music import detect_music

        tg = adir / "audio_tags.json"
        if not tg.exists():
            log("music skipped: needs audio_tags.json (run the tags step)")
        else:
            mu = adir / "music.json"
            if mu.exists() and not force:
                log(f"music cached: {mu}")
            else:
                detect_music(tg, mu, force=force)
            out["music"] = mu
            proj.mark("music", path=str(mu), range=[t0, t1])

    if "peaks" in steps or "deadair" in steps:
        from .audio_tags import AudioTags
        from .face_motion import FaceMotion

        sp = adir / "speakers.json"
        if not sp.exists():
            log("peaks/deadair skipped: needs speakers.json")
        else:
            fm_path = adir / "face_motion.json"
            if not fm_path.exists() and (proj.analysis_dir / "face_motion.json").exists():
                fm_path = proj.analysis_dir / "face_motion.json"
            fm = FaceMotion(fm_path) if fm_path.exists() else None
            if "peaks" in steps:
                from .peaks import detect_peaks

                tg = adir / "audio_tags.json"
                tags = AudioTags(tg) if tg.exists() else None
                segs = None
                try:
                    segs = json.loads(sp.read_text(encoding="utf-8")).get("segments")
                except Exception:  # noqa: BLE001
                    segs = None
                pk = adir / "peaks.json"
                if pk.exists() and not force:
                    log(f"peaks cached: {pk}")
                else:
                    detect_peaks(sp, pk, face_motion=fm, tags=tags, transcript_segments=segs, force=force)
                out["peaks"] = pk
                proj.mark("peaks", path=str(pk), range=[t0, t1])
            if "deadair" in steps:
                from .deadair import detect_dead_air

                da = adir / "deadair.json"
                if da.exists() and not force:
                    log(f"dead air cached: {da}")
                else:
                    detect_dead_air(sp, da, face_motion=fm, force=force)
                out["deadair"] = da
                proj.mark("deadair", path=str(da), range=[t0, t1])
    return out
