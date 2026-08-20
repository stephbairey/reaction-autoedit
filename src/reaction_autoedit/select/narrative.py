"""The narrative structure process — what MUST the abridged cut show for the story to work?

Three artifacts, first two produced here (each one Anthropic call, cached):

N1  ``narrative_plan.json`` — film knowledge only, no transcript. The Wikipedia plot section for
    the title (fetched via the public API; grounds the model for obscure titles — famous ones it
    knows cold) → a Save-the-Cat beat sheet: for each beat present in the film, a summary, why it's
    essential, an importance score, and the **big lines** — dialogue a viewer expects to hear.

N2  ``narrative.json`` — the plan aligned to THIS recording. The model receives the beat sheet and
    the timestamped FILM dialogue transcript and returns each beat's span in the recording plus
    each big line matched to the (whisper-mangled) transcript text with timestamps and a priority:
    ``must`` / ``should`` / ``could``. Grounding to transcript lines is what prevents invented
    timestamps.

N3 lives in the selector: musts are placed first (split into clip-cap slices — film audio runs
uninterrupted under reactor-cam cutaways), then shoulds, then peaks under the film/reactor ratio,
then gap-filling spine. Chapters come from beat labels.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_MODEL = "claude-sonnet-5"

N1_SYSTEM = """You are a story analyst helping edit an abridged movie-reaction cut. Given a film's
title/year and (when available) its Wikipedia plot section, produce a Save-the-Cat-style beat sheet
of the beats ACTUALLY PRESENT in the film, in order. For each beat give the dialogue lines a viewer
would most expect to hear (the famous/big lines and the lines that carry the plot).

Return STRICT JSON:
{"beats": [{"stc": "opening_image|setup|theme_stated|catalyst|debate|break_into_two|b_story|
fun_and_games|midpoint|bad_guys_close_in|all_is_lost|dark_night_of_the_soul|break_into_three|
finale|final_image|other", "label": "<≤8 words>", "summary": "<1-2 sentences>",
"why_essential": "<1 sentence>", "importance": <0..1>,
"big_lines": ["<verbatim or near-verbatim dialogue>", ...]}]}
Rules: 10-18 beats; merge tiny beats; big_lines 0-4 per beat, exact wording where famous;
importance reflects how lost a viewer is without the beat."""

N2_SYSTEM = """You are aligning a film's beat sheet to a specific recording of that film (a reaction
recording; timestamps are seconds into the recording). You get the beat sheet and the film-dialogue
transcript (auto-transcribed: quotes may be garbled — match them fuzzily). Occasional viewer
commentary may be mislabelled as film dialogue; ignore lines that are clearly commentary.

Return STRICT JSON:
{"beats": [{"label": "<from the sheet>", "stc": "...", "importance": <0..1>,
            "t0": <sec>, "t1": <sec>, "priority": "must|should|could"}],
 "key_lines": [{"beat": "<beat label>", "expected": "<line from the sheet>",
                "heard": "<matching transcript text>", "t0": <sec>, "t1": <sec>,
                "priority": "must|should|could"}]}
Rules: every beat from the sheet appears once with its span in THIS recording (t0/t1 bracketed by
transcript timestamps you actually saw; if a beat is absent from the transcript, give your best
span estimate between its neighbours and priority "could"). key_lines only where you genuinely
matched the line (or its garbled form) in the transcript. Priorities: "must" = the story breaks
without it (aim for 10-16 musts total across beats+lines), "should" = strongly expected,
"could" = nice to have."""


# ---------------------------------------------------------------- wikipedia
def fetch_wikipedia_plot(title: str, year: int | None) -> dict | None:
    """Best-effort: search for the film's article, return {'title','url','plot'} or None."""
    api = "https://en.wikipedia.org/w/api.php"

    def call(params: dict) -> dict:
        qs = urllib.parse.urlencode({**params, "format": "json"})
        req = Request(f"{api}?{qs}", headers={"User-Agent": "reaction-autoedit/0.1 (fair-use editing tool)"})
        with urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    try:
        q = f"{title} {year} film" if year else f"{title} film"
        hits = call({"action": "query", "list": "search", "srsearch": q, "srlimit": 5}).get("query", {}).get("search", [])
        if not hits:
            return None
        page = hits[0]["title"]
        data = call({"action": "query", "prop": "extracts", "explaintext": 1, "titles": page, "redirects": 1})
        pages = data.get("query", {}).get("pages", {})
        text = next(iter(pages.values())).get("extract", "")
        m = re.search(r"^==\s*Plot\s*==\s*$(.*?)(?=^==[^=])", text, re.S | re.M)
        plot = (m.group(1) if m else text[:6000]).strip()
        if len(plot) < 200:
            return None
        return {"title": page, "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page.replace(' ', '_'))}",
                "plot": plot[:12000]}
    except Exception:  # noqa: BLE001 — fetch is best-effort; the model still knows famous films
        return None


# ---------------------------------------------------------------- anthropic
def _ask(system: str, user: str, model: str, max_tokens: int = 6000) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                 messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    if msg.stop_reason == "max_tokens":
        # ask the model to continue and stitch the halves
        msg2 = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                      messages=[{"role": "user", "content": user},
                                                {"role": "assistant", "content": text}])
        text += "".join(b.text for b in msg2.content if b.type == "text")
    return _parse_json_lenient(text)


def _parse_json_lenient(text: str) -> dict:
    """Extract the largest parseable JSON object: exact parse first, then progressively repair a
    truncated tail (drop the last partial element, close open brackets)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model response")
    raw = text[start:text.rfind("}") + 1] if "}" in text else text[start:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    body = text[start:]
    for cut in range(len(body), max(len(body) - 20000, 100), -1):
        frag = body[:cut].rstrip().rstrip(",")
        opens = frag.count("{") - frag.count("}")
        opens_sq = frag.count("[") - frag.count("]")
        if opens < 0 or opens_sq < 0 or frag.count('"') % 2:
            continue
        try:
            return json.loads(frag + "]" * opens_sq + "}" * opens)
        except json.JSONDecodeError:
            continue
    raise ValueError("could not parse model JSON (truncated beyond repair)")


def build_plan(title: str, year: int | None, out: Path, *, model: str = DEFAULT_MODEL, force: bool = False) -> Path:
    """N1: beat sheet with big lines, from film knowledge + Wikipedia plot."""
    out = Path(out)
    if out.exists() and not force:
        return out
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set (put it in .env)")
    wiki = fetch_wikipedia_plot(title, year)
    src = f"Wikipedia plot section of '{wiki['title']}':\n{wiki['plot']}" if wiki else \
        "(No plot text available — use your own knowledge of the film; if you don't know it, return fewer, high-confidence beats.)"
    user = f"Film: {title}" + (f" ({year})" if year else "") + f"\n\n{src}\n\nReturn the JSON now."
    data = _ask(N1_SYSTEM, user, model)
    beats = [b for b in data.get("beats", []) if b.get("label")]
    result = {"model": model, "title": title, "year": year,
              "source": {k: wiki[k] for k in ("title", "url")} if wiki else None,
              "n": len(beats), "beats": beats}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def ground_plan(plan_path: Path, transcript_segments: list[dict], out: Path, *,
                model: str = DEFAULT_MODEL, film_bounds: tuple[float, float] | None = None,
                max_chars: int = 130_000, force: bool = False) -> Path:
    """N2: align the beat sheet to this recording's FILM transcript → narrative.json."""
    out = Path(out)
    if out.exists() and not force:
        return out
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    lines = []
    for s in transcript_segments:
        if s.get("speaker") == "REACTOR":
            continue
        if film_bounds and not (film_bounds[0] - 30 <= s["start"] <= film_bounds[1] + 30):
            continue
        txt = (s.get("text") or "").strip()
        if txt:
            lines.append(f"[{s['start']:.0f}-{s['end']:.0f}] {txt}")
    body = "\n".join(lines)
    if len(body) > max_chars:
        keep = max_chars / len(body)
        body = "\n".join(l for i, l in enumerate(lines) if (i * keep) % 1 < keep)
    user = (f"Beat sheet:\n{json.dumps({'beats': plan['beats']}, ensure_ascii=False)}\n\n"
            f"Film-dialogue transcript of the recording:\n{body}\n\nReturn the JSON now.")
    data = _ask(N2_SYSTEM, user, model, max_tokens=16000)
    beats = sorted((b for b in data.get("beats", []) if b.get("t1", 0) > b.get("t0", 0)), key=lambda b: b["t0"])
    keys = sorted((k for k in data.get("key_lines", []) if k.get("t1", 0) > k.get("t0", 0)), key=lambda k: k["t0"])
    if film_bounds:
        a, b_ = film_bounds
        beats = [x for x in beats if x["t1"] > a and x["t0"] < b_]
        keys = [x for x in keys if x["t1"] > a and x["t0"] < b_]
    result = {"model": model, "plan": str(plan_path), "n_beats": len(beats), "n_key_lines": len(keys),
              "beats": beats, "key_lines": keys}
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
