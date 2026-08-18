"""``rae`` — reaction-autoedit command line.

M1: probe, init, detect-layout, set-layout, edl-init, edl-check, render, make-fixture, make-templates, compute.
M2: analyze (transcribe + speakers), transcript, speaker-check, speaker-review.
Later-stage commands (select, preflight) are stubs that document the intended interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, compute, ffmpeg
from .config import ReactorConfig, TitleConfig, dump_json, load_reactor
from .edl import EDL, starter_edl
from .models import Geometry
from .project import DEFAULT_ROOT, Project

app = typer.Typer(help="reaction-autoedit: abridged co-watch cuts from movie-reaction composites.",
                  no_args_is_help=True, add_completion=False)
console = Console()


def _version(value: bool):
    if value:
        console.print(f"rae {__version__}")
        raise typer.Exit()


@app.callback()
def _main(version: bool = typer.Option(False, "--version", callback=_version, is_eager=True)):
    pass


# --------------------------------------------------------------------------- inspection
@app.command()
def probe(path: Path):
    """Print resolution / fps / duration / audio info for a media file."""
    info = ffmpeg.probe(path)
    console.print_json(info.model_dump_json())


@app.command("compute")
def compute_info(no_gpu: bool = typer.Option(False, "--no-gpu", help="ignore any GPU")):
    """Show the detected compute profile (device, encoder, defaults)."""
    p = compute.detect(prefer_gpu=not no_gpu)
    t = Table(show_header=False)
    t.add_row("device", p.device + (f" ({p.gpu_name})" if p.gpu_name else ""))
    t.add_row("cpu cores", str(p.cpu_count))
    t.add_row("video encoder", p.video_encoder)
    t.add_row("render jobs", str(p.render_jobs))
    t.add_row("whisper default", f"{p.whisper_model} / {p.whisper_compute_type}")
    t.add_row("ffmpeg", ffmpeg.ffmpeg_bin() if ffmpeg.available() else "[red]not found[/]")
    for n in p.notes:
        t.add_row("note", n)
    console.print(t)


# --------------------------------------------------------------------------- project
@app.command()
def init(
    name: str,
    input: Path = typer.Option(..., "--input", "-i", exists=True, help="composite recording (mp4/mkv)"),
    reactor: Optional[Path] = typer.Option(None, help="configs/reactors/*.json"),
    title: Optional[Path] = typer.Option(None, help="configs/titles/*.json"),
    root: Path = typer.Option(DEFAULT_ROOT, help="work root"),
    overwrite: bool = False,
):
    """Create work/<name>/project.json for a recording (probes the file)."""
    proj = Project.create(name, input, root=root, reactor_config=str(reactor) if reactor else None,
                          title_config=str(title) if title else None, overwrite=overwrite)
    pi = proj.state.probe
    console.print(f"[green]created[/] {proj.path}")
    console.print(f"  {pi.width}x{pi.height} @ {pi.fps:g} fps, {pi.duration/60:.1f} min, "
                  f"{pi.video_codec}/{pi.audio.codec if pi.audio else 'no audio'}")


@app.command("detect-layout")
def detect_layout_cmd(
    name: str,
    frames: int = typer.Option(120, help="frames to sample across the recording"),
    template: Optional[Path] = typer.Option(None, help="geometry JSON to use as fallback/override (defaults to reactor config's layout_template)"),
    force_template: bool = typer.Option(False, help="skip detection; use the template as-is"),
    debug_image: bool = typer.Option(True, help="write work/<name>/layout_debug.png"),
    root: Path = typer.Option(DEFAULT_ROOT),
):
    """Stage 1: detect movie / facecam regions and store them in project.json."""
    from .ingest.layout import detect_layout

    proj = Project.load(name, root)
    tmpl: Geometry | None = None
    if template:
        tmpl = Geometry.model_validate(json.loads(template.read_text(encoding="utf-8")))
    elif proj.reactor().layout_template is not None:
        tmpl = proj.reactor().layout_template
    dbg = proj.root / "layout_debug.png" if debug_image else None
    if force_template:
        if tmpl is None:
            raise typer.BadParameter("--force-template needs a template")
        geom = tmpl.model_copy(deep=True)
        geom.source = "template"
    else:
        with console.status(f"sampling {frames} frames…"):
            geom = detect_layout(proj.source, n_frames=frames, template=tmpl, debug_image=dbg)
    proj.state.geometry = geom
    proj.mark("layout", confidence=geom.confidence, source=geom.source)
    _print_geometry(geom)
    if dbg:
        console.print(f"debug image: {dbg}")


def _print_geometry(g: Geometry):
    t = Table(title=f"geometry (conf {g.confidence:.2f}, {g.source})")
    t.add_column("region"); t.add_column("x"); t.add_column("y"); t.add_column("w"); t.add_column("h"); t.add_column("extra")
    t.add_row("movie", str(g.movie.x), str(g.movie.y), str(g.movie.w), str(g.movie.h), "")
    t.add_row("movie_inner", str(g.movie_inner.x), str(g.movie_inner.y), str(g.movie_inner.w), str(g.movie_inner.h), f"aspect {g.movie_inner.aspect:.2f}")
    t.add_row("face", str(g.face.x), str(g.face.y), str(g.face.w), str(g.face.h), g.face.shape)
    console.print(t)
    for n in g.notes:
        console.print(f"  [dim]note:[/] {n}")


@app.command("set-layout")
def set_layout(
    name: str,
    movie: str = typer.Option(..., help="x,y,w,h of movie active picture"),
    face: str = typer.Option(..., help="x,y,w,h of facecam"),
    circle: bool = typer.Option(False, help="facecam is circular"),
    root: Path = typer.Option(DEFAULT_ROOT),
):
    """Manually set geometry (when auto-detect gets it wrong)."""
    from .models import FaceRegion, FrameInfo, Rect

    proj = Project.load(name, root)
    pi = proj.state.probe
    mx, my, mw, mh = (int(v) for v in movie.split(","))
    fx, fy, fw, fh = (int(v) for v in face.split(","))
    r = Rect(x=mx, y=my, w=mw, h=mh)
    geom = Geometry(frame=FrameInfo(w=pi.width, h=pi.height, fps=pi.fps), movie=r, movie_inner=r,
                    face=FaceRegion(x=fx, y=fy, w=fw, h=fh, shape="circle" if circle else "rect"),
                    confidence=1.0, source="manual")
    proj.state.geometry = geom
    proj.mark("layout", confidence=1.0, source="manual")
    _print_geometry(geom)


# --------------------------------------------------------------------------- EDL
@app.command("edl-init")
def edl_init(name: str, out: Optional[Path] = None, root: Path = typer.Option(DEFAULT_ROOT)):
    """Write a small starter EDL (both layouts, a lower third, chapters) to work/<name>/edl.json."""
    proj = Project.load(name, root)
    edl = starter_edl(str(proj.source), proj.state.probe.duration)
    edl.meta["title"] = proj.title().title
    dest = out or proj.edl_path
    edl.save(dest)
    console.print(f"[green]wrote[/] {dest} ({len(edl.segments)} segments, {edl.duration:.0f}s) — edit it, then `rae render {name}`")


@app.command("edl-check")
def edl_check(edl_path: Path, clip_cap: float = typer.Option(7.0), name: Optional[str] = typer.Option(None, help="project (for source duration)"), root: Path = typer.Option(DEFAULT_ROOT)):
    """Validate an EDL and print warnings."""
    edl = EDL.load(edl_path)
    dur = Project.load(name, root).state.probe.duration if name else None
    warns = edl.validate_rules(clip_cap_s=clip_cap, source_duration=dur)
    console.print(f"{len(edl.segments)} segments, {edl.duration/60:.1f} min output")
    for w in warns:
        console.print(f"[yellow]warn:[/] {w}")
    if not warns:
        console.print("[green]ok[/]")


# --------------------------------------------------------------------------- render
@app.command()
def render(
    name: str,
    edl: Optional[Path] = typer.Option(None, "--edl", help="EDL to render (default work/<name>/edl.json)"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
    preview: bool = typer.Option(False, help="fast 480p preview"),
    jobs: Optional[int] = typer.Option(None, help="parallel segment encodes"),
    force: bool = typer.Option(False, help="ignore cached intermediates"),
    encoder: Optional[str] = typer.Option(None, help="force ffmpeg video encoder (libx264, h264_nvenc, …)"),
    root: Path = typer.Option(DEFAULT_ROOT),
):
    """Stage 4/5: render an EDL to video (+ chapters, description, mid-roll suggestions, EDL copy)."""
    from .assemble.render import render as do_render

    proj = Project.load(name, root)
    geom = proj.require_geometry()
    edl_path = edl or proj.edl_path
    if not edl_path.exists():
        raise typer.BadParameter(f"no EDL at {edl_path} (run `rae edl-init {name}` or pass --edl)")
    e = EDL.load(edl_path)
    if not Path(e.source).exists():
        console.print(f"[yellow]EDL source {e.source} missing; using project source[/]")
        e.source = str(proj.source)
    tag = "preview" if preview else "final"
    dest = out or (proj.renders_dir / f"{name}_{tag}.mp4")
    profile = compute.detect(encoder=encoder) if encoder else compute.detect()
    res = do_render(e, geom, out=dest, reactor=proj.reactor(), title=proj.title(), preview=preview,
                    jobs=jobs, force=force, profile=profile, source_duration=proj.state.probe.duration)
    proj.mark("render_preview" if preview else "render", output=str(res.output), duration=res.duration)


# --------------------------------------------------------------------------- assets / fixtures
@app.command("make-fixture")
def make_fixture_cmd(
    out_dir: Path = typer.Option(Path("samples"), help="where to write fixture_<preset>.mp4"),
    preset: str = typer.Option("all", help="sbs | pip | all"),
    duration: float = typer.Option(60.0),
):
    """Generate synthetic composite recordings with known geometry (for tests / dry runs)."""
    from .ingest.fixture import PRESETS, make_fixture

    presets = PRESETS if preset == "all" else (preset,)
    for p in presets:
        with console.status(f"rendering fixture {p}…"):
            v, j = make_fixture(out_dir, p, duration=duration)
        console.print(f"[green]wrote[/] {v}  (truth: {j})")


@app.command("make-templates")
def make_templates_cmd(
    reactor: Optional[Path] = typer.Option(None, help="reactor config for URL / name"),
    out_dir: Path = typer.Option(Path("templates")),
):
    """Generate placeholder endcard + lower-third PNGs (replace with designed assets later)."""
    from .assemble.templates import make_endcard, make_lower_third

    rc = load_reactor(reactor)
    e = make_endcard(out_dir / "endcard.png", patreon_url=rc.patreon_url, display_name=rc.display_name)
    lt = make_lower_third(out_dir / "lower_third.png", text="FULL REACTION ON PATREON", sub=rc.patreon_url)
    console.print(f"[green]wrote[/] {e}\n[green]wrote[/] {lt}")


@app.command("init-config")
def init_config(out_dir: Path = typer.Option(Path("configs"))):
    """Write example reactor + title config files."""
    dump_json(ReactorConfig(name="example", display_name="the reactor", patreon_url="https://www.patreon.com/yourname"),
              out_dir / "reactors" / "example.json")
    dump_json(TitleConfig(title="Example Movie", year=1999, studio="Example Studios"), out_dir / "titles" / "example.json")
    console.print(f"[green]wrote[/] {out_dir}/reactors/example.json and {out_dir}/titles/example.json")


# --------------------------------------------------------------------------- later stages (stubs)
def _not_yet(stage: str, module: str):
    console.print(f"[yellow]{stage} is not implemented yet[/] — see `{module}` for the planned interface.")
    raise typer.Exit(code=2)


@app.command()
def analyze(
    name: str,
    only: str = typer.Option("transcribe,video,speakers,tags,music,peaks,deadair", help="comma list of steps: transcribe,video,speakers,tags,music,peaks,deadair"),
    range_: Optional[str] = typer.Option(None, "--range", help="analyse only T0-T1 seconds (e.g. 1500-1800); cached separately"),
    model: Optional[str] = typer.Option(None, help="whisper model (tiny/base/small/medium/large-v3); default from compute profile"),
    voice: list[Path] = typer.Option([], help="extra voice sample(s) for enrolment"),
    backend: str = typer.Option("auto", help="speaker backend: auto | ecapa | resemblyzer"),
    fusion: float = typer.Option(0.0, help="face-motion fusion weight (0 = audio only; motion still computed for peaks)"),
    chunk_min: float = typer.Option(10.0, help="local speaker-adaptation chunk length in minutes (0 = global)"),
    force: bool = typer.Option(False, help="recompute even if cached"),
    no_gpu: bool = typer.Option(False, "--no-gpu"),
    root: Path = typer.Option(DEFAULT_ROOT),
):
    """Stage 2: transcribe, video signals, speaker attribution, audio tags, music tiering, peaks, dead air."""
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    from .analysis import pipeline

    proj = Project.load(name, root)
    t0, t1 = _parse_range(range_)
    steps = tuple(x.strip() for x in only.split(",") if x.strip())
    steps = tuple(pipeline.ALIASES.get(x, x) for x in steps)
    bad = [x for x in steps if x not in pipeline.STEPS]
    if bad:
        raise typer.BadParameter(f"unknown step(s) {bad}; choose from {pipeline.STEPS}")
    profile = compute.detect(prefer_gpu=not no_gpu)
    with Progress(TextColumn("[bold]{task.description}"), BarColumn(), TextColumn("{task.fields[msg]}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("analyze", total=1.0, msg="")

        def progress(frac: float, msg: str):
            prog.update(task, completed=frac, msg=msg)

        outs = pipeline.run(proj, steps=steps, t0=t0, t1=t1, force=force, model=model, profile=profile,
                            voice=list(voice), backend=backend, fusion_weight=fusion, chunk_min=chunk_min,
                            log=lambda m: console.print(f"[dim]{m}[/]"), progress=progress)
        prog.update(task, completed=1.0, msg="done")
    for k, v in outs.items():
        console.print(f"  {k}: {v}")
    if "speakers" in outs:
        _speaker_summary(outs["speakers"])


def _parse_range(r: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    if not r:
        return None, None
    try:
        a, b = r.split("-", 1)
        return (float(a) if a else None), (float(b) if b else None)
    except ValueError:
        raise typer.BadParameter("--range must look like T0-T1 (seconds), e.g. 1500-1800")


def _speaker_summary(path: Path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    c = d["counts"]
    tot = max(1, sum(c.values()))
    a = d.get("adaptation", {})
    console.print(f"speakers[{d.get('backend')}]: key={d.get('score_key','sim')} threshold={d['threshold']} ({d.get('threshold_source','')}) margin={d['margin']}; "
                  f"enrol sim p10={d['enrollment'].get('sim_p10')} median={d['enrollment'].get('sim_median')}")
    if a:
        console.print(f"  adaptation: {a}")
    console.print("  " + "  ".join(f"{k}={v} ({100*v/tot:.0f}%)" for k, v in c.items())
                  + f"  | reactor spans: {len(d['reactor_spans'])}")


@app.command()
def transcript(
    name: str,
    range_: Optional[str] = typer.Option(None, "--range", help="which cached range to show (default: full)"),
    speaker: Optional[str] = typer.Option(None, help="filter: REACTOR | FILM | MIXED | UNKNOWN"),
    limit: int = typer.Option(200),
    root: Path = typer.Option(DEFAULT_ROOT),
):
    """Print the (speaker-tagged) transcript for review."""
    from .analysis import pipeline

    proj = Project.load(name, root)
    t0, t1 = _parse_range(range_)
    adir = pipeline.analysis_dir(proj, t0, t1)
    sp, tp = adir / "speakers.json", adir / "transcript.json"
    if sp.exists():
        segs = json.loads(sp.read_text(encoding="utf-8"))["segments"]
    elif tp.exists():
        segs = [{**s, "speaker": "?", "sim_mean": None} for s in json.loads(tp.read_text(encoding="utf-8"))["segments"]]
    else:
        raise typer.BadParameter(f"nothing analysed in {adir} yet (run `rae analyze {name}`)")
    colors = {"REACTOR": "green", "FILM": "cyan", "MIXED": "yellow", "UNKNOWN": "dim", "?": "white"}
    n = 0
    for s in segs:
        if speaker and s["speaker"] != speaker.upper():
            continue
        m, sec = divmod(int(s["start"]), 60)
        sim = f"{s['sim_mean']:.2f}" if s.get("sim_mean") is not None else "  - "
        console.print(f"[{colors.get(s['speaker'], 'white')}]{m:3d}:{sec:02d} {s['speaker']:<7} {sim}[/] {s['text']}")
        n += 1
        if n >= limit:
            console.print(f"[dim]… (limit {limit}; use --limit)[/]")
            break


@app.command()
def peaks(
    name: str,
    range_: Optional[str] = typer.Option(None, "--range"),
    top: int = typer.Option(25, help="show the N highest peaks"),
    root: Path = typer.Option(DEFAULT_ROOT),
):
    """List detected reaction peaks (plus music tiering + dead-air totals) for review."""
    from .analysis import pipeline

    proj = Project.load(name, root)
    t0, t1 = _parse_range(range_)
    adir = pipeline.analysis_dir(proj, t0, t1)
    pk = adir / "peaks.json"
    if not pk.exists():
        raise typer.BadParameter(f"no peaks.json in {adir} (run `rae analyze {name}`)")
    d = json.loads(pk.read_text(encoding="utf-8"))
    ps = sorted(d["peaks"], key=lambda p: -p["score"])[:top]
    t = Table(title=f"top {len(ps)} of {d['n']} peaks")
    for c in ("time", "dur", "score", "kind", "reactor says"):
        t.add_column(c)
    for p in sorted(ps, key=lambda p: p["t"]):
        m, s = divmod(int(p["t"]), 60)
        t.add_row(f"{m}:{s:02d}", f"{p['t1']-p['t0']:.1f}s", f"{p['score']:.2f}", p["kind"], (p["text"] or "")[:70])
    console.print(t)
    mu, da = adir / "music.json", adir / "deadair.json"
    if mu.exists():
        m = json.loads(mu.read_text(encoding="utf-8"))
        console.print(f"music: {len(m['spans'])} spans — song {m['totals']['song_s']/60:.1f} min, score {m['totals']['score_s']/60:.1f} min")
    if da.exists():
        dd = json.loads(da.read_text(encoding="utf-8"))
        console.print(f"dead air: {len(dd['spans'])} spans, {dd['total_s']/60:.1f} min total")


@app.command("speaker-review")
def speaker_review(
    name: str,
    range_: Optional[str] = typer.Option(None, "--range"),
    per_class: int = typer.Option(12, help="clips per class"),
    within: Optional[str] = typer.Option(None, help="only pick from these sub-ranges, e.g. 2343-2362,2600-2638"),
    out: Optional[Path] = typer.Option(None, help="output dir (default <analysis dir>/review)"),
    root: Path = typer.Option(DEFAULT_ROOT),
):
    """Export audio contact sheets (reactor / borderline / film windows) to verify the tagger by ear."""
    from .analysis import pipeline
    from .analysis.speakers import export_review

    proj = Project.load(name, root)
    t0, t1 = _parse_range(range_)
    adir = pipeline.analysis_dir(proj, t0, t1)
    sp = adir / "speakers.json"
    if not sp.exists():
        raise typer.BadParameter(f"no speakers.json in {adir}")
    w = [tuple(float(x) for x in r.split("-")) for r in within.split(",")] if within else None
    outs = export_review(proj.analysis_dir / "audio16k.wav", sp, out or (adir / "review"), per_class=per_class, within=w)
    for k, v in outs.items():
        console.print(f"  {k}: {v}")
    console.print("listen to each file; note the clip numbers that are wrong (see review_index.txt)")


@app.command("speaker-eval")
def speaker_eval(
    name: str,
    labels: Optional[Path] = typer.Option(None, help="labels.json (default work/<name>/analysis/labels.json)"),
    range_: Optional[str] = typer.Option(None, "--range", help="which speakers.json to score (default: full)"),
    root: Path = typer.Option(DEFAULT_ROOT),
):
    """Score speakers.json against human labels (from annotated speaker-review picks)."""
    from .analysis import pipeline
    from .analysis.speakers import evaluate

    proj = Project.load(name, root)
    t0, t1 = _parse_range(range_)
    sp = pipeline.analysis_dir(proj, t0, t1) / "speakers.json"
    lp = labels or (proj.analysis_dir / "labels.json")
    if not sp.exists() or not lp.exists():
        raise typer.BadParameter(f"need both {sp} and {lp}")
    res = evaluate(sp, lp)
    console.print_json(json.dumps(res))


@app.command("speaker-check")
def speaker_check(
    name: str,
    holdout: Path = typer.Option(..., help="a clean clip of the reactor NOT used for enrolment"),
    voice: list[Path] = typer.Option([], help="extra enrolment sample(s)"),
    root: Path = typer.Option(DEFAULT_ROOT),
):
    """Sanity-check enrolment: does a held-out clip of the reactor score as REACTOR?"""
    from .analysis import pipeline
    from .analysis.speakers import self_check

    proj = Project.load(name, root)
    res = self_check(pipeline.voice_samples(proj, list(voice)), holdout, device=compute.detect().device)
    console.print_json(json.dumps(res))


@app.command()
def select(
    name: str,
    runtime: Optional[float] = typer.Option(None, help="target runtime in minutes (default: title config, 55)"),
    clip_cap: Optional[float] = typer.Option(None, help="max continuous movie footage in seconds (default: title config, 7)"),
    withhold: Optional[bool] = typer.Option(None, help="withhold-the-climax (default: title config)"),
    out: Optional[Path] = typer.Option(None, help="EDL path (default work/<name>/edl.json)"),
    root: Path = typer.Option(DEFAULT_ROOT),
):
    """Stage 3: two-budget selection (narrative spine + reaction peaks) → EDL."""
    from .select.selector import Analysis, SelectParams, select as do_select

    proj = Project.load(name, root)
    tc = proj.title()
    params = SelectParams(
        runtime_target_s=(runtime or tc.runtime_target_min) * 60.0,
        clip_cap_s=clip_cap or tc.clip_cap_s,
        withhold_climax=tc.withhold_climax if withhold is None else withhold,
        layout_min_s=tc.layout_min_s,
    )
    an = Analysis.load(proj.analysis_dir, proj.state.probe.duration)
    if not an.peaks:
        console.print("[yellow]no peaks.json — run `rae analyze` first (selection will be spine-only)[/]")
    edl = do_select(an, params, source=str(proj.source), reactor=proj.reactor(), title=tc)
    dest = out or proj.edl_path
    edl.save(dest)
    proj.mark("select", path=str(dest), duration=edl.duration, segments=len(edl.segments))
    kinds = {}
    for sgm in edl.segments:
        kinds[sgm.kind] = kinds.get(sgm.kind, 0) + sgm.dur
    console.print(f"[green]wrote[/] {dest}: {len(edl.segments)} segments, {edl.duration/60:.1f} min "
                  f"(target {params.runtime_target_s/60:.0f}); film {an.film_start/60:.1f}→{an.film_end/60:.1f} min; "
                  f"peaks used {edl.meta['peaks_used']}/{edl.meta['peaks_available']}, spine gap {edl.meta['spine_gap_s']}s, "
                  f"withheld {edl.meta['withheld']}")
    console.print("  by kind: " + ", ".join(f"{k} {v/60:.1f} min" for k, v in kinds.items()))
    warns = edl.validate_rules(clip_cap_s=params.clip_cap_s, source_duration=proj.state.probe.duration)
    for w in warns[:10]:
        console.print(f"[yellow]warn:[/] {w}")
    console.print(f"next: `rae render {name} --preview` (review), then `rae render {name}`")


@app.command()
def preflight(name: str, root: Path = typer.Option(DEFAULT_ROOT)):
    """Stage 0: per-title risk / monetization check. [M5]"""
    _not_yet("Stage 0 (preflight)", "reaction_autoedit/preflight/")


@app.command("log-outcome")
def log_outcome(name: str, outcome: str = typer.Argument(..., help="none | sharing | redirect | block"), root: Path = typer.Option(DEFAULT_ROOT)):
    """Stage 6: record the actual claim outcome for an upload. [M5]"""
    from .preflight.outcomes import OutcomeStore

    proj = Project.load(name, root)
    store = OutcomeStore.default()
    store.record(title=proj.title().title, studio=proj.title().studio, outcome=outcome, project=name)
    console.print(f"[green]recorded[/] {outcome} for {proj.title().title} → {store.path}")


if __name__ == "__main__":
    app()
