"""Narrative beats via an LLM pass over the FILM dialogue (the brief's optional Anthropic call).

Why: spine slices picked purely by dialogue density keep the cut *watchable* but not *coherent* —
the model knows which scenes are load-bearing (setup, promise of the premise, midpoint, low point,
climax). We ask for a save-the-cat-style beat list with timestamps grounded in the transcript we
send, so it works for any movie in the backlog — no external "important scenes" dataset needed
(the public ones cover only a fixed film list).

Usage: ``rae beats <name>`` (needs ``ANTHROPIC_API_KEY`` and the ``llm`` extra). Output
``analysis/beats.json``::

    {"model": "...", "beats": [{"t0": 512.0, "t1": 590.0, "label": "The pact: find the body",
                                "importance": 0.9, "act": "setup"}]}

The selector uses beats to (a) boost spine candidates inside important beats and (b) name chapters.
Also useful later for Stage 0: beat labels double as a scene inventory per title.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SYSTEM = """You are a film editor's assistant. You receive the dialogue transcript of a movie
(as heard during a reaction recording; timestamps are seconds into that recording, and the
transcript may contain occasional commentary mislabelled as film dialogue — ignore lines that are
clearly a viewer commenting). Identify the narrative beats a viewer must see to follow the story,
in the spirit of save-the-cat beat sheets: opening image, setup, catalyst, debate, break into two,
midpoint, bad guys close in, all is lost, dark night of the soul, finale, final image — but only
the ones actually present, merged where scenes overlap.

Return STRICT JSON: {"beats": [{"t0": <sec>, "t1": <sec>, "label": "<short beat name>",
"importance": <0..1>, "act": "setup|confrontation|resolution"}]}
Rules: 8-20 beats; t0/t1 must be timestamps that appear in (or between) the transcript lines you
were given; labels ≤ 8 words, no spoilers beyond what the dialogue shows; importance reflects how
essential the beat is to following the plot."""


def extract_beats(
    transcript_segments: list[dict],
    out: Path,
    *,
    model: str = "claude-sonnet-5",
    film_only: bool = True,
    max_chars: int = 120_000,
    force: bool = False,
) -> Path:
    out = Path(out)
    if out.exists() and not force:
        return out
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set — the beats pass is optional and needs it")
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("install the llm extra: uv sync --extra llm") from e

    lines = []
    for s in transcript_segments:
        if film_only and s.get("speaker") not in ("FILM", "MIXED", None, "?"):
            continue
        txt = (s.get("text") or "").strip()
        if txt:
            lines.append(f"[{s['start']:.0f}] {txt}")
    body = "\n".join(lines)
    if len(body) > max_chars:  # decimate evenly rather than truncate the ending
        keep = max_chars / len(body)
        lines = [l for i, l in enumerate(lines) if (i * keep) % 1 < keep]
        body = "\n".join(lines)

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Transcript:\n{body}\n\nReturn the JSON now."}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    beats = sorted((b for b in data.get("beats", []) if b.get("t1", 0) > b.get("t0", 0)), key=lambda b: b["t0"])
    result = {"model": model, "n": len(beats), "beats": beats}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
