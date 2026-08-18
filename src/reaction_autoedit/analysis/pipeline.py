"""Stage 2 orchestration: run the analysers for a project, with per-artifact caching.

Full-track artifacts live in ``work/<name>/analysis/``; a ``--range t0 t1`` run (fast iteration
during development) lives in ``analysis/r<t0>-<t1>/`` so it never masquerades as the full result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..compute import ComputeProfile, detect
from ..project import Project
from . import audio, transcribe as _tr

STEPS = ("transcribe", "facemotion", "speakers")


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
    log: Callable[[str], None] = print,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Path]:
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

    if "facemotion" in steps:
        from .face_motion import compute_face_motion

        if proj.state.geometry is None:
            log("facemotion skipped: no geometry yet (run `rae detect-layout`)")
        else:
            fm = adir / "face_motion.json"
            if fm.exists() and not force:
                log(f"face motion cached: {fm}")
            else:
                log("face motion (mouth-region frame differences @5 fps) …")
                compute_face_motion(proj.source, proj.state.geometry, fm, t0=t0, t1=t1,
                                    duration=proj.state.probe.duration if proj.state.probe else None,
                                    force=force, progress=progress)
            out["face_motion"] = fm
            proj.mark("facemotion", path=str(fm), range=[t0, t1])

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
                         fusion_weight=fusion_weight, labels_path=proj.analysis_dir / "labels.json", progress=progress)
        out["speakers"] = sp
        proj.mark("speakers", path=str(sp), range=[t0, t1])
    return out
