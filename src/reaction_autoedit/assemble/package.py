"""Stage 5 packaging: chapters, title/description drafts, mid-roll suggestions."""

from __future__ import annotations

from pathlib import Path

from ..config import ReactorConfig, TitleConfig
from ..edl import EDL


def _ts(t: float) -> str:
    t = int(round(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def chapters(edl: EDL, head_offset: float = 0.0) -> list[tuple[float, str]]:
    """(output_time, title) pairs; YouTube needs the first at 0:00, so one is synthesised if missing.
    ``head_offset`` accounts for anything concatenated before the first segment (opening bumper)."""
    out: list[tuple[float, str]] = []
    for s, off in zip(edl.segments, edl.offsets()):
        if s.chapter:
            out.append((off + head_offset, s.chapter))
    if not out or out[0][0] > 0.5:
        out.insert(0, (0.0, "Start"))
    return out


def write_chapters(edl: EDL, path: str | Path, head_offset: float = 0.0) -> Path:
    p = Path(path)
    p.write_text("\n".join(f"{_ts(t)} {name}" for t, name in chapters(edl, head_offset)) + "\n", encoding="utf-8")
    return p


def draft_title(title: TitleConfig, reactor: ReactorConfig) -> str:
    year = f" ({title.year})" if title.year else ""
    return f"{title.title}{year} — First Time Watching | Reaction"


def write_description(edl: EDL, reactor: ReactorConfig, title: TitleConfig, path: str | Path,
                      head_offset: float = 0.0) -> Path:
    lines = [
        draft_title(title, reactor),
        "",
        f"This is the abridged YouTube cut. The FULL uncut reaction to {title.title} is on Patreon:",
        reactor.patreon_url,
        "",
        "Chapters:",
    ] + [f"{_ts(t)} {name}" for t, name in chapters(edl, head_offset)] + [
        "",
        "All footage is used for commentary and criticism under fair use; clips are kept short "
        "and the film is not shown in full.",
    ]
    p = Path(path)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def suggest_midrolls(edl: EDL, *, first_frac: float = 0.4, spacing_s: float = 420.0) -> list[float]:
    """Suggested mid-roll timestamps (output timeline).

    M1 heuristic: at segment boundaries, no earlier than ``first_frac`` of the runtime, spaced by
    ``spacing_s``, preferring boundaries where the *next* segment is ``story`` (i.e. not landing on a
    reaction peak). Stage 2 will refine this using scene boundaries and reaction lulls.
    """
    total = edl.duration
    if total < spacing_s * 1.5:
        return []
    picks: list[float] = []
    last = first_frac * total - spacing_s
    for s, off in zip(edl.segments[1:], edl.offsets()[1:]):
        if off < first_frac * total:
            continue
        if off - last >= spacing_s and s.kind == "story":
            picks.append(off)
            last = off
    return picks


def write_midrolls(edl: EDL, path: str | Path) -> Path:
    p = Path(path)
    ts = suggest_midrolls(edl)
    body = "\n".join(_ts(t) for t in ts) if ts else "# runtime too short for mid-rolls"
    p.write_text(body + "\n", encoding="utf-8")
    return p
