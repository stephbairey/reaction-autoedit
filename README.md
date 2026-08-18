# reaction-autoedit

Automated editing pipeline for movie reaction videos. Takes a full-length reaction recording (a single composite video with the film and the reactor's facecam in one frame) and produces an abridged, chronological cut suitable for YouTube, with a full-length version reserved for members.

The output is a compressed co-watch, not a highlights reel. The cut follows the film's story from start to finish, keeps the reactor present throughout, and spends its runtime budget on narrative coverage first and big reaction moments second, so the viewer feels like they watched the movie with someone, just faster.

## What it does

Given one input file, the pipeline detects the layout (movie region and facecam region), transcribes and attributes the audio (reactor speech vs. film audio), finds reaction peaks using vocal energy and facial expression analysis, maps scene boundaries and dead air, and then builds a cut list under a set of configurable editing rules: a target runtime, a hard cap on continuous film footage, chronological ordering, music-segment avoidance, and an optional hold-back of the biggest moments for the members-only version.

Assembly is done by cropping and zooming regions of the composite frame, switching between a movie-forward layout (small facecam overlay) and a reactor-forward layout (facecam enlarged) depending on what matters in each segment. The render package includes the finished video, a chapters file, a draft title and description, suggested mid-roll ad timestamps placed at natural lulls, and the cut list itself as an editable EDL.

## Modes

`--review` (default) stops after selection and emits the EDL plus a fast low-res preview so a human can adjust the cut before final render. `--render-from-edl` finishes the job from an edited list. `--auto` runs straight through with no stop. An optional preflight step checks how similar content for a given title has fared on the platform before committing to an edit.

## Editing rules (all configurable)

Runtime target (default 55 minutes), maximum continuous film clip length (default 7 seconds), music handling (song and needle-drop segments excluded or shown reactor-forward; score minimized), layout hysteresis, and the hold-back toggle. Per-reactor config covers layout templates, a one-time voice enrollment sample for speaker attribution, branding assets, and outro links. Per-title config covers aspect ratio, runtime, and risk flags.

## Stack

Python 3.11+, ffmpeg, faster-whisper for transcription, speaker embeddings (pyannote / resemblyzer) for voice attribution, librosa for audio features and music detection, PySceneDetect and OpenCV for scene and layout analysis, with optional Real-ESRGAN upscaling for enlarged facecam segments and an optional LLM pass for narrative-beat identification.

## Status

Early development. Build order: render path first (layout detection plus crop/zoom assembly from a hand-written EDL), then speaker attribution, then peak and music detection, then automated selection, then the review loop and full automation.

## Notes

Input recordings are single-track composites, so audio handling works by segment selection rather than remixing. Higher-resolution source recordings (4K) noticeably improve the enlarged-facecam layout and are recommended where possible.

---

## Getting started

### Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or plain `pip`)
- **ffmpeg + ffprobe** on your PATH (or set `FFMPEG_BIN` / `FFPROBE_BIN`)
  - Windows: `winget install Gyan.FFmpeg` (or download a build from gyan.dev and add its `bin` to PATH)
  - macOS: `brew install ffmpeg`
  - Linux / WSL: `sudo apt install ffmpeg` — or, without root, drop a [static build](https://johnvansickle.com/ffmpeg/) into `~/.local/bin`
- No GPU required. If an NVIDIA GPU is present (torch sees CUDA / ffmpeg has a working `h264_nvenc`), it is used automatically for analysis and encoding. `rae compute` shows what was detected; `RAE_DEVICE=cpu` / `RAE_ENCODER=libx264` force CPU paths.

### Install

```bash
git clone <this repo> && cd reaction-autoedit
uv sync --extra dev            # core: CLI, layout detection, rendering
uv sync --extra analysis       # later: whisper / speaker / audio analysis stack (heavy)
uv run rae --help
```

### Quickstart (real recording)

```bash
# 1. project + probe
uv run rae init mymovie --input samples/mymovie.mp4 \
    --reactor configs/reactors/example.json --title configs/titles/example.json

# 2. Stage 1: where are the movie and the facecam?
uv run rae detect-layout mymovie          # writes geometry into work/mymovie/project.json
#    → check work/mymovie/layout_debug.png. Wrong? `rae set-layout mymovie --movie x,y,w,h --face x,y,w,h`
#      or put a layout_template into the reactor config (it wins whenever detection is unsure or disagrees).

# 3. a cut list (until Stage 3 exists, hand-write or start from the starter)
uv run rae edl-init mymovie               # work/mymovie/edl.json — edit segments freely
uv run rae edl-check work/mymovie/edl.json --name mymovie

# 4. render
uv run rae render mymovie --preview       # fast 480p check
uv run rae render mymovie                 # 1080p final → work/mymovie/renders/mymovie_final.mp4
```

Every render writes sidecars next to the video: `*.edl.json` (the exact cut), `*.chapters.txt`,
`*.description.txt` (draft with Patreon link), `*.midrolls.txt` (suggested ad break timestamps).
Intermediate segment clips are cached in `renders/.tmp_*` and reused when the EDL/geometry/style
did not change, so tweaking one segment re-encodes only that segment.

### Dry run without footage

```bash
uv run rae make-fixture                   # samples/fixture_sbs.mp4 + fixture_pip.mp4 (synthetic composites)
uv run rae init demo --input samples/fixture_sbs.mp4
uv run rae detect-layout demo
uv run rae render demo --edl examples/edl_fixture.json --preview
uv run pytest                             # unit + fixture-based integration tests (no real footage needed)
```

## The EDL (cut list) format

`work/<name>/edl.json` is the contract between selection (Stage 3) and assembly (Stage 4). It is
plain JSON, meant to be edited by hand in `--review` mode:

```json
{
  "version": 1,
  "source": "samples/mymovie.mp4",
  "target": {"w": 1920, "h": 1080, "fps": 30},
  "segments": [
    {"id": "s001", "in": 123.4, "out": 129.9, "layout": "movie-large",
     "kind": "story", "transition": "cut", "chapter": "The setup", "note": ""},
    {"id": "s002", "in": 129.9, "out": 133.0, "layout": "reactor-large",
     "kind": "reaction", "transition": "cut"}
  ],
  "overlays": [{"type": "lower_third", "at": 1800, "dur": 6}],
  "endcard": {"template": "templates/endcard.png", "dur": 8}
}
```

- `in` / `out` are **source** seconds; segments must be chronological. `at` (overlays) is on the **output** timeline.
- `layout`: `movie-large` (film fills frame, facecam PiP) or `reactor-large` (facecam blown up over a blurred film background).
- `kind`: `story | reaction | cta | intro | outro` — informational now, used by selection/mid-roll logic.
- `transition`: `cut` or `xfade` (rendered as a short dip-to-black into the segment).
- `chapter`: starts a YouTube chapter. `rae edl-check` warns about clip-cap violations, ordering, out-of-range times.

## Configuration

- `configs/reactors/<name>.json` — per reactor: display name, Patreon URL, PiP style (corner/size), reactor-large style
  (face size, blur, darken, sharpen, vignette, optional movie PiP), branding template paths, `voice_sample`
  (for speaker enrolment), optional `layout_template` (geometry fallback/override).
- `configs/titles/<name>.json` — per title: runtime target, clip cap, withhold-climax toggle, layout hysteresis,
  aspect override, risk flag, studio.
- `configs/outcomes.json` — Stage 6 outcome table (`rae log-outcome <project> none|sharing|redirect|block`).
- `templates/endcard.png`, `templates/lower_third.png` — placeholders generated by `rae make-templates`; replace with designed assets.

`rae init-config` writes example files.

## Layout of the code

```
src/reaction_autoedit/
  cli.py            rae commands
  models.py         Rect / Geometry / ProbeInfo
  ffmpeg.py         ffmpeg/ffprobe wrappers, encoder checks
  compute.py        CPU/GPU + encoder detection, defaults
  config.py         reactor / title / render-target config models
  project.py        work/<name>/project.json lifecycle
  edl.py            EDL schema, validation, starter EDL
  ingest/           Stage 1: probe, layout detection, synthetic fixtures
  analysis/         Stage 2 (interfaces documented; implementation in Milestone 2/3)
  select/           Stage 3 (documented; Milestone 4)
  assemble/         Stage 4/5: layout filtergraphs, renderer, packaging, template generation
  preflight/        Stage 0 + Stage 6 outcome table
```

## Roadmap

1. ✅ **M1** — Stage 1 layout detection + Stage 4/5 render from a hand-written EDL (this).
2. **M2** — Stage 2 transcription + speaker attribution (resemblyzer first, pyannote optional); validate on real footage.
3. **M3** — reaction peaks, music tiering, scenes, dead air.
4. **M4** — Stage 3 two-budget selection with clip cap and withhold-the-climax; `--review` / `--auto`.
5. **M5** — Stage 0 preflight, Stage 6 outcome loop, optional Real-ESRGAN pass for `reactor-large`.
