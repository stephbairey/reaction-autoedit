"""GUI API routes — thin wrappers over the existing engine modules. No pipeline logic here."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import compute, ffmpeg
from ..config import ReactorConfig, TitleConfig, dump_json, load_reactor, load_title
from ..edl import EDL
from ..project import DEFAULT_ROOT, Project
from .jobs import Job, JobManager

router = APIRouter(prefix="/api")
jobs = JobManager(max_workers=1)


def _proj(name: str) -> Project:
    try:
        return Project.load(name, DEFAULT_ROOT)
    except FileNotFoundError:
        raise HTTPException(404, f"no project '{name}'")


# ------------------------------------------------------------------ system / lookup
@router.get("/system")
def system():
    p = compute.detect()
    return {"device": p.device, "gpu": p.gpu_name, "cpu_count": p.cpu_count,
            "encoder": p.video_encoder, "whisper": f"{p.whisper_model}/{p.whisper_compute_type}",
            "ffmpeg": ffmpeg.available(), "notes": p.notes,
            "keys": {"anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
                     "tvdb": bool(os.environ.get("TVDB_API_KEY")),
                     "youtube": bool(os.environ.get("YOUTUBE_API_KEY"))}}


@router.get("/lookup")
def lookup(title: str, year: int | None = None, force: bool = False):
    from ..preflight.outcomes import OutcomeStore
    from ..preflight.survey import SurveyError, survey

    slug = "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")
    cache = Path("work/_lookup") / f"{slug}{'-' + str(year) if year else ''}.json"
    try:
        sv = survey(title, year, cache, force=force)
    except SurveyError as e:
        raise HTTPException(503, str(e))
    own = [e for e in OutcomeStore.default().entries if title.lower() in e.get("title", "").lower()]
    return {"survey": sv, "own": own}


# ------------------------------------------------------------------ projects
@router.get("/projects")
def projects():
    out = []
    root = DEFAULT_ROOT
    if root.exists():
        for pdir in sorted(root.iterdir()):
            f = pdir / "project.json"
            if not f.exists():
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            probe = d.get("probe") or {}
            renders = sorted((pdir / "renders").glob("*.mp4")) if (pdir / "renders").exists() else []
            out.append({"name": d["name"], "source": d.get("source"),
                        "duration_min": round((probe.get("duration") or 0) / 60, 1),
                        "stages": {k: v.get("at") for k, v in (d.get("stages") or {}).items()},
                        "film_bounds": d.get("film_bounds"),
                        "has_edl": (pdir / "edl.json").exists(),
                        "renders": [r.name for r in renders]})
    return out


class NewProject(BaseModel):
    name: str
    source: str
    reactor_config: str | None = "configs/reactors/example.json"
    title_config: str | None = None
    title: str | None = None
    year: int | None = None
    studio: str | None = None


@router.post("/projects")
def create_project(body: NewProject):
    if not Path(body.source).exists():
        raise HTTPException(400, f"source not found: {body.source}")
    tc_path = body.title_config
    if tc_path is None and body.title:
        tc = TitleConfig(title=body.title, year=body.year, studio=body.studio)
        tc_path = f"configs/titles/{''.join(c if c.isalnum() else '-' for c in body.title.lower()).strip('-')}.json"
        dump_json(tc, tc_path)
    try:
        proj = Project.create(body.name, body.source, root=DEFAULT_ROOT,
                              reactor_config=body.reactor_config, title_config=tc_path)
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except ffmpeg.FFmpegError as e:
        raise HTTPException(400, f"could not read the recording: {e}")
    return {"name": proj.state.name, "probe": proj.state.probe.model_dump()}


@router.get("/projects/{name}")
def project_detail(name: str):
    proj = _proj(name)
    d = json.loads(proj.path.read_text(encoding="utf-8"))
    adir = proj.analysis_dir
    arts = {p.stem: f"/media/{name}/analysis/{p.name}" for p in adir.glob("*.json")} if adir.exists() else {}
    renders = []
    for r in sorted(proj.renders_dir.glob("*.mp4")):
        renders.append({"name": r.name, "url": f"/media/{name}/renders/{r.name}",
                        "size_mb": round(r.stat().st_size / 1e6, 1),
                        "sidecars": {sc.suffix.lstrip("."): f"/media/{name}/renders/{sc.name}"
                                     for sc in proj.renders_dir.glob(r.stem + ".*") if sc.suffix != ".mp4"}})
    return {"state": d, "artifacts": arts, "renders": renders,
            "layout_debug": f"/media/{name}/layout_debug.png" if (proj.root / "layout_debug.png").exists() else None,
            "edl": f"/media/{name}/edl.json" if proj.edl_path.exists() else None,
            "narrative": json.loads((adir / "narrative.json").read_text(encoding="utf-8")) if (adir / "narrative.json").exists() else None,
            "preflight": json.loads((adir / "preflight.json").read_text(encoding="utf-8")) if (adir / "preflight.json").exists() else None}


# ------------------------------------------------------------------ stages (jobs)
class StageReq(BaseModel):
    options: dict = {}


@router.post("/projects/{name}/run/{stage}")
def run_stage(name: str, stage: str, body: StageReq | None = None):
    proj = _proj(name)
    opts = (body.options if body else {}) or {}

    def task(job: Job):
        from ..analysis import pipeline

        log, prog = jobs.log_cb(job), jobs.progress_cb(job)
        if stage == "detect":
            from ..ingest.layout import detect_layout

            tmpl = proj.reactor().layout_template
            geom = detect_layout(proj.source, n_frames=int(opts.get("frames", 120)), template=tmpl,
                                 debug_image=proj.root / "layout_debug.png")
            proj.state.geometry = geom
            proj.mark("layout", confidence=geom.confidence, source=geom.source)
            return geom.model_dump()
        if stage == "analyze":
            profile = compute.detect()
            steps = tuple(opts.get("steps", ",".join(pipeline.STEPS)).split(","))
            try:
                pipeline.voice_samples(proj)
            except (FileNotFoundError, RuntimeError):
                log("no voice sample — skipping speaker attribution")
                steps = tuple(x for x in steps if x != "speakers")
            outs = pipeline.run(proj, steps=steps, force=bool(opts.get("force")), profile=profile,
                                voice=[], log=log, progress=prog)
            return {k: str(v) for k, v in outs.items()}
        if stage == "narrative":
            from ..select.narrative import build_plan, ground_plan

            tc = proj.title()
            sp = proj.analysis_dir / "speakers.json"
            tp = sp if sp.exists() else proj.analysis_dir / "transcript.json"
            if not tp.exists():
                raise RuntimeError("run analyze first")
            log("N1: plot → beat sheet…")
            plan = build_plan(tc.title, tc.year, proj.analysis_dir / "narrative_plan.json",
                              force=bool(opts.get("force")))
            log("N2: aligning to the recording…")
            segs = json.loads(tp.read_text(encoding="utf-8"))["segments"]
            fb = tuple(proj.state.film_bounds) if proj.state.film_bounds else None
            out = ground_plan(plan, segs, proj.analysis_dir / "narrative.json", film_bounds=fb,
                              force=bool(opts.get("force")))
            d = json.loads(out.read_text(encoding="utf-8"))
            proj.mark("narrative", path=str(out), beats=d["n_beats"], key_lines=d["n_key_lines"])
            return {"beats": d["n_beats"], "key_lines": d["n_key_lines"], "moments": d.get("n_moments", 0)}
        if stage == "select":
            from ..select.selector import Analysis, SelectParams, select as do_select

            tc = proj.title()
            params = SelectParams(
                runtime_target_s=float(opts.get("runtime_min", tc.runtime_target_min)) * 60.0,
                clip_cap_s=float(opts.get("clip_cap", tc.clip_cap_s)),
                withhold_climax=bool(opts.get("withhold", tc.withhold_climax)),
                trim_intro=bool(opts.get("trim_intro", tc.trim_intro)),
                trim_outro=bool(opts.get("trim_outro", tc.trim_outro)),
                silence_cut_s=opts.get("silence_cut", tc.silence_cut_s),
                movie_frac=float(opts.get("movie_frac", tc.movie_frac)),
                layout_min_s=tc.layout_min_s,
            )
            an = Analysis.load(proj.analysis_dir, proj.state.probe.duration)
            fb = tuple(proj.state.film_bounds) if proj.state.film_bounds else None
            edl = do_select(an, params, source=str(proj.source), reactor=proj.reactor(), title=tc, film_bounds=fb)
            edl.save(proj.edl_path)
            proj.mark("select", path=str(proj.edl_path), duration=edl.duration, segments=len(edl.segments))
            return {"segments": len(edl.segments), "duration_min": round(edl.duration / 60, 1), "meta": edl.meta}
        if stage == "render":
            from ..assemble.render import render as do_render
            from ..config import RenderTarget

            preview = bool(opts.get("preview", True))
            e = EDL.load(proj.edl_path)
            if not Path(e.source).exists():
                e.source = str(proj.source)
            if not preview:
                e.target = RenderTarget.preset(str(opts.get("resolution", proj.reactor().branding.resolution)))
            geom = proj.require_geometry()
            tag = "preview" if preview else "final"
            dest = proj.renders_dir / f"{name}_{tag}.mp4"
            res = do_render(e, geom, out=dest, reactor=proj.reactor(), title=proj.title(),
                            preview=preview, force=bool(opts.get("force")),
                            source_duration=proj.state.probe.duration)
            proj.mark("render_preview" if preview else "render", output=str(res.output), duration=res.duration)
            return {"output": res.output.name, "duration_min": round(res.duration / 60, 1),
                    "warnings": res.warnings[:10]}
        if stage == "card":
            from ..assemble.templates import make_title_card

            tc = proj.title()
            rc = proj.reactor()
            url = tc.clearlogo_url
            if url is None and os.environ.get("TVDB_API_KEY"):
                from ..cli import _tvdb_clearlogo

                url = _tvdb_clearlogo(tc.title, tc.year)
            logo = None
            if url:
                from urllib.request import Request, urlopen

                logo = proj.root / "assets" / "clearlogo.png"
                if not logo.exists():
                    logo.parent.mkdir(parents=True, exist_ok=True)
                    with urlopen(Request(url, headers={"User-Agent": "reaction-autoedit/0.1"}), timeout=30) as r:
                        logo.write_bytes(r.read())
            base = rc.branding.title_card_base
            dest = proj.root / "assets" / "title_card.png"
            make_title_card(dest, logo=logo, base=base, title=tc.title, subtitle="" if base else "abridged reaction")
            return {"card": f"/media/{name}/assets/title_card.png"}
        raise RuntimeError(f"unknown stage {stage}")

    try:
        job = jobs.submit(stage, name, task)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return job.as_dict()


@router.get("/jobs")
def list_jobs(project: str | None = None):
    return jobs.list(project)


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    return j.as_dict()


# ------------------------------------------------------------------ film bounds / edl / outcomes
class BoundsReq(BaseModel):
    start: float
    end: float
    from_preview: bool = True


@router.post("/projects/{name}/film-bounds")
def set_bounds(name: str, body: BoundsReq):
    proj = _proj(name)
    a, b = body.start, body.end
    if body.from_preview:
        from ..cli import _preview_to_source

        a, b = _preview_to_source(proj, a), _preview_to_source(proj, b)
    if not (0 <= a < b):
        raise HTTPException(400, f"bad bounds {a:.1f}..{b:.1f}")
    proj.state.film_bounds = [round(a, 2), round(b, 2)]
    proj.save()
    return {"film_bounds": proj.state.film_bounds}


@router.get("/projects/{name}/edl")
def get_edl(name: str):
    proj = _proj(name)
    if not proj.edl_path.exists():
        raise HTTPException(404, "no EDL yet — run select")
    e = EDL.load(proj.edl_path)
    offs = e.offsets()
    head = 0.0
    ob = proj.reactor().branding.opening_bumper
    if ob and Path(ob).exists():
        head = ffmpeg.probe(ob).duration
    return {"source": e.source, "duration": e.duration,
            "warnings": e.validate_rules(clip_cap_s=proj.title().clip_cap_s,
                                         source_duration=proj.state.probe.duration),
            "head_offset": head,
            "segments": [{**s.model_dump(by_alias=True), "at": round(off + head, 2)}
                         for s, off in zip(e.segments, offs)]}


class EdlPatch(BaseModel):
    drop: list[str] = []
    flip: list[str] = []                 # movie-large <-> reactor-large
    retime: dict[str, list[float]] = {}  # id -> [in, out]


@router.post("/projects/{name}/edl")
def patch_edl(name: str, body: EdlPatch):
    proj = _proj(name)
    e = EDL.load(proj.edl_path)
    drop = set(body.drop)
    out = []
    for s in e.segments:
        if s.id in drop:
            continue
        if s.id in body.flip and s.layout in ("movie-large", "reactor-large"):
            s = s.model_copy(update={"layout": "reactor-large" if s.layout == "movie-large" else "movie-large",
                                     "note": s.note + " (flipped in review)"})
        if s.id in body.retime:
            a, b = body.retime[s.id]
            if b > a >= 0:
                s = s.model_copy(update={"in_": round(a, 2), "out": round(b, 2), "note": s.note + " (retimed)"})
        out.append(s)
    e.segments = out
    e.save(proj.edl_path)
    return {"segments": len(out), "duration_min": round(e.duration / 60, 1)}


class OutcomeReq(BaseModel):
    outcome: str


@router.post("/projects/{name}/outcome")
def log_outcome_api(name: str, body: OutcomeReq):
    from ..preflight.outcomes import OutcomeStore

    proj = _proj(name)
    store = OutcomeStore.default()
    try:
        store.record(title=proj.title().title, studio=proj.title().studio, outcome=body.outcome, project=name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"recorded": body.outcome, "studio_table": {k: dict(v) for k, v in store.by_studio().items()}}


# ------------------------------------------------------------------ CTA / branding asset styling
class StyleReq(BaseModel):
    kind: str                       # lower_third | endcard
    text: str = "FULL REACTION ON PATREON"
    sub: str = ""
    out: str | None = None          # default: templates/<kind>.png


@router.post("/style")
def style(body: StyleReq):
    from ..assemble.templates import make_endcard, make_lower_third

    out = body.out or f"templates/{body.kind}.png"
    if body.kind == "lower_third":
        make_lower_third(out, text=body.text, sub=body.sub)
    elif body.kind == "endcard":
        make_endcard(out, patreon_url=body.sub or "patreon.com", display_name=body.text or "the reactor")
    else:
        raise HTTPException(400, "kind must be lower_third|endcard")
    return {"written": out, "url": "/asset?path=" + out}


@router.get("/asset")
def asset(path: str):
    from fastapi.responses import FileResponse

    p = Path(path)
    if not p.exists() or p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise HTTPException(404, "no such asset")
    return FileResponse(p)


# ------------------------------------------------------------------ settings
_CONFIGS = {"reactor": (ReactorConfig, load_reactor), "title": (TitleConfig, load_title)}


@router.get("/settings/{kind}")
def get_settings(kind: str, path: str):
    if kind not in _CONFIGS:
        raise HTTPException(404, "kind must be reactor|title")
    model, loader = _CONFIGS[kind]
    cfg = loader(path if Path(path).exists() else None)
    return {"path": path, "values": cfg.model_dump(), "schema": model.model_json_schema()}


class SettingsBody(BaseModel):
    path: str
    values: dict


@router.post("/settings/{kind}")
def save_settings(kind: str, body: SettingsBody):
    if kind not in _CONFIGS:
        raise HTTPException(404, "kind must be reactor|title")
    model, _ = _CONFIGS[kind]
    try:
        cfg = model.model_validate(body.values)
    except Exception as e:  # noqa: BLE001 — pydantic error → 400 with detail
        raise HTTPException(400, str(e))
    dump_json(cfg, body.path)
    return {"saved": body.path}
