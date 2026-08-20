# User settings inventory (phase-2 GUI contract)

Every setting the GUI exposes already exists as a config key today — the GUI is a form over these
files, nothing more. `configs/reactors/<name>.json` is per-channel (set once), `configs/titles/<name>.json`
is per-movie, and per-recording facts live in `work/<name>/project.json`.

## Output

| GUI setting | Config key | Values / default |
|---|---|---|
| Render resolution | `branding.resolution` (reactor) or `rae render --resolution` | `480` / `720` / `1080` (default 1080; preview always 480) |
| Runtime target | `runtime_target_min` (title) | minutes, default 55 (main content; bumpers/cards ride on top) |
| Max continuous movie clip | `clip_cap_s` (title) | seconds, default 7 (range 5–10 per the brief) |
| Movie vs reactor screen time | `movie_frac` (title) | 0.5–0.95, default 0.75 |

## Branding (per reactor, "upload" = file picker)

| GUI setting | Config key |
|---|---|
| Upload title-card background | `branding.title_card_base` |
| Upload opening bumper | `branding.opening_bumper` |
| Upload ending bumper | `branding.ending_bumper` |
| Upload endcard | `branding.endcard` |
| Upload CTA banner (lower third) | `branding.lower_third` |
| Patreon link | `patreon_url` |
| Display name | `display_name` |

## Layout & style

| GUI setting | Config key | Notes |
|---|---|---|
| Inset corner | `pip.corner` | default `bottom-left` (reactor faces the film) |
| Inset size | `pip.width_frac` | default 0.24 |
| Border width / colors | `pip.border.px`, `.color_from`, `.color_to` (same under `reactor_large.border`) | default 4px, `#9762FF` → `#FF01F8` |
| Reactor-large face size | `reactor_large.face_height_frac` | default 0.92 |
| Background blur / darken / vignette | `reactor_large.blur_strength`, `.darken`, `.vignette` | |
| Facecam sharpen | `reactor_large.sharpen` | fights upscale softness |

## Structure (per title)

| GUI setting | Config key | Values / default |
|---|---|---|
| Show title card | `show_title_card` | on/off, default on |
| Auto-cut intro | `trim_intro` | default off (uncut, full-frame) |
| Auto-cut outro | `trim_outro` | default off (uncut, full-frame) |
| Cut intro/outro silences | `silence_cut_s` | seconds (e.g. 2.5); silences longer than this are removed; default off |
| Withhold the climax | `withhold_climax` | default on |
| Film start/end | `rae set-film-bounds` (project) | auto-detected; manual pin accepts preview timestamps |

## CTA banner schedule (reactor, `branding.lower_third_schedule`)

| GUI setting | Config key | Notes |
|---|---|---|
| Mode | periodic vs explicit | explicit when `at_min` is set |
| First showing | `start_min` | default 20 |
| Every X minutes | `every_min` | default 20 |
| At specific times | `at_min` | list of minute marks, replaces periodic |
| Minimum gap | `min_gap_min` | default 10; also spaces against withheld-climax CTAs |
| Banner duration | `lower_third_duration` (branding) | default 6 s |

## CTA styling

Today the banner and endcard are PNG templates (upload your own; placeholders via
`rae make-templates`). Planned for the GUI: a style form (text, colors, position, gradient accents)
that regenerates the PNGs — same mechanism as the title card composer.
