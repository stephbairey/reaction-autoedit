"""Stage 3 — two-budget selection → EDL.

Inputs (all from ``work/<name>/analysis``): speaker-tagged transcript, peaks, music spans, scene cuts,
dead air; plus TitleConfig (runtime target, clip cap, withhold toggle, layout hysteresis).

Model of the cut (a *compressed co-watch*, strictly chronological):

1. **Intro / outro** — his pre-film monologue and post-film wrap-up are kept, trimmed
   (``intro_s`` / ``outro_s``), as ``reactor-large``.
2. **Budget B — reaction peaks**: the top peaks become *reaction groups*: a ``movie-large`` slice of
   what he is reacting to (≤ clip cap, ends at the peak) followed by a ``reactor-large`` segment
   covering the peak extent. Peaks inside *song* music spans get no movie slice (reactor-large only).
3. **Budget A — narrative spine**: walk the film chronologically; wherever the gap between covered
   film time exceeds ``max_gap_s``, insert a *spine slice*: a ``movie-large`` slice ≤ clip cap on the
   densest FILM dialogue near a scene cut, avoiding song spans and dead air, followed (if he says
   something there) by a short ``reactor-large`` cutaway. This keeps the story followable.
4. **Runtime**: while over target, drop the weakest peaks (Budget B) and widen the spine gap
   (Budget A) — never violate the clip cap; while under target, add more peaks / tighten the gap.
5. **Withhold-the-climax** (toggle): the ``withhold_n`` biggest peaks in the last ``withhold_last_frac``
   of the film are replaced by a short ``cta`` segment (reactor-large + lower third): "his full
   reaction to the ending is on Patreon".
6. **Layout hysteresis**: adjacent segments with the same layout are merged; segments shorter than
   ``layout_min_s`` are stretched into their neighbour's time or dropped.

Everything here is heuristic and deliberately transparent: every segment carries a ``note`` saying
why it exists (peak #, spine gap, intro…), so the human review pass can reason about it.
An optional LLM pass (Anthropic) to name chapters / pick key beats plugs in via ``beat_hints``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import ReactorConfig, RenderTarget, TitleConfig
from ..edl import EDL, Endcard, Overlay, Segment


@dataclass
class Analysis:
    duration: float
    segments: list[dict]                 # speaker-tagged transcript segments
    peaks: list[dict]
    music: list[dict]
    cuts: np.ndarray
    dead: list[dict]
    film_start: float
    film_end: float

    @classmethod
    def load(cls, adir: Path, duration: float) -> "Analysis":
        def rd(name, default):
            p = adir / name
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

        sp = rd("speakers.json", {"segments": []})
        segs = sp.get("segments") or rd("transcript.json", {"segments": []}).get("segments", [])
        peaks = rd("peaks.json", {"peaks": []})["peaks"]
        music = rd("music.json", {"spans": []})["spans"]
        cuts = np.asarray(rd("scenes.json", {"cuts": []})["cuts"], dtype=float)
        dead = rd("deadair.json", {"spans": []})["spans"]
        fs, fe = _film_bounds(segs, cuts, duration)
        return cls(duration=duration, segments=segs, peaks=peaks, music=music, cuts=cuts, dead=dead,
                   film_start=fs, film_end=fe)

    # ---- helpers ------------------------------------------------------------------------
    def song_overlap(self, a: float, b: float) -> float:
        return sum(max(0.0, min(b, s["t1"]) - max(a, s["t0"])) for s in self.music if s["kind"] == "song")

    def score_overlap(self, a: float, b: float) -> float:
        return sum(max(0.0, min(b, s["t1"]) - max(a, s["t0"])) for s in self.music if s["kind"] == "score")

    def in_dead(self, t: float) -> bool:
        return any(d["t0"] <= t <= d["t1"] for d in self.dead)

    def snap_to_cut(self, t: float, max_dist: float = 1.5, prefer: str = "before") -> float:
        if self.cuts.size == 0:
            return t
        i = int(np.argmin(np.abs(self.cuts - t)))
        c = float(self.cuts[i])
        return c if abs(c - t) <= max_dist else t

    def film_segments(self, a: float, b: float) -> list[dict]:
        return [s for s in self.segments if s.get("speaker") in ("FILM", "MIXED", None, "?") and s["end"] > a and s["start"] < b]

    def reactor_segments(self, a: float, b: float) -> list[dict]:
        return [s for s in self.segments if s.get("speaker") == "REACTOR" and s["end"] > a and s["start"] < b]


def _film_bounds(segs: list[dict], cuts: np.ndarray, duration: float) -> tuple[float, float]:
    """Where does the film start/end inside the recording? First/last FILM-tagged speech, sanity-bounded
    by scene cuts. Falls back to 5 % / 97 % of the duration."""
    film = sorted((s for s in segs if s.get("speaker") == "FILM"), key=lambda s: s["start"])
    fs = fe = None
    # sustained: a FILM segment followed by ≥3 more FILM segments within 90 s (and mirrored for the end)
    for i, s in enumerate(film):
        if sum(1 for x in film[i + 1:i + 8] if x["start"] - s["start"] <= 90) >= 3:
            fs = s["start"]
            break
    for i in range(len(film) - 1, -1, -1):
        s = film[i]
        if sum(1 for x in film[max(0, i - 7):i] if s["end"] - x["end"] <= 90) >= 3:
            fe = s["end"]
            break
    if fs is None or fe is None or fe <= fs:
        fs, fe = duration * 0.05, duration * 0.97
    if cuts.size:
        early = cuts[cuts < fs]
        if early.size and fs - early[-1] < 120:
            fs = float(early[-1])
    return max(0.0, fs), min(duration, fe)


@dataclass
class SelectParams:
    runtime_target_s: float = 55 * 60
    clip_cap_s: float = 7.0
    withhold_climax: bool = True
    withhold_n: int = 3
    withhold_last_frac: float = 0.25
    layout_min_s: float = 2.0
    intro_s: float = 25.0
    outro_s: float = 20.0
    reaction_min_s: float = 3.0
    reaction_max_s: float = 12.0
    spine_slice_s: float = 6.0
    spine_cutaway_s: float = 3.0
    max_gap_s: float = 75.0            # film seconds allowed without any coverage (tuned by runtime loop)
    peak_share: float = 0.55           # fraction of the runtime budget for Budget B before spine fills
    tolerance_s: float = 60.0
    fps: float = 30.0


@dataclass
class _Piece:
    """A candidate group of segments anchored at a film time."""
    anchor: float
    segs: list[Segment]
    kind: str                          # peak | spine | intro | outro | cta
    score: float = 0.0
    note: str = ""

    @property
    def dur(self) -> float:
        return sum(s.dur for s in self.segs)


def select(analysis: Analysis, params: SelectParams, *, source: str, reactor: ReactorConfig | None = None,
           title: TitleConfig | None = None) -> EDL:
    A, P = analysis, params
    reactor = reactor or ReactorConfig()
    title = title or TitleConfig()
    pieces: list[_Piece] = []

    # ---- intro / outro --------------------------------------------------------------------
    intro_segs = A.reactor_segments(0.0, A.film_start)
    if intro_segs and P.intro_s > 0:
        a = intro_segs[0]["start"]
        b = min(a + P.intro_s, A.film_start)
        pieces.append(_Piece(anchor=a, kind="intro", note="intro monologue (trimmed)",
                             segs=[Segment(id="intro", **{"in": a}, out=b, layout="reactor-large", kind="intro", chapter="Intro")]))
    outro_segs = A.reactor_segments(A.film_end, A.duration)
    if outro_segs and P.outro_s > 0:
        b = min(outro_segs[-1]["end"], A.duration)
        a = max(A.film_end, b - P.outro_s)
        pieces.append(_Piece(anchor=a, kind="outro", note="outro (trimmed)",
                             segs=[Segment(id="outro", **{"in": a}, out=b, layout="reactor-large", kind="outro", chapter="Final thoughts")]))

    # ---- Budget B: peaks ------------------------------------------------------------------
    peaks = sorted([p for p in A.peaks if A.film_start <= p["t"] <= A.film_end], key=lambda p: -p["score"])
    withheld: list[dict] = []
    if P.withhold_climax and P.withhold_n > 0:
        late = A.film_start + (A.film_end - A.film_start) * (1 - P.withhold_last_frac)
        late_peaks = [p for p in peaks if p["t"] >= late][:P.withhold_n]
        withheld = late_peaks
        peaks = [p for p in peaks if p not in late_peaks]
    peak_pieces = [_peak_piece(p, i, A, P) for i, p in enumerate(peaks, 1)]
    peak_pieces = [pp for pp in peak_pieces if pp is not None]

    # ---- CTA for withheld climaxes ---------------------------------------------------------
    for j, p in enumerate(withheld, 1):
        a = max(A.film_start, p["t0"] - 2.0)
        b = min(a + 4.0, p["t"])
        pieces.append(_Piece(anchor=a, kind="cta", score=p["score"], note=f"withheld peak (score {p['score']:.2f}) → CTA",
                             segs=[Segment(id=f"cta{j}", **{"in": a}, out=b, layout="reactor-large", kind="cta",
                                           note="his full reaction to this moment is on Patreon")]))

    # ---- runtime loop -----------------------------------------------------------------------
    fixed_dur = sum(pc.dur for pc in pieces)
    budget = P.runtime_target_s - fixed_dur - (reactor.branding.endcard_duration or 0)
    # peaks by score until peak_share of the budget
    chosen_peaks: list[_Piece] = []
    used = 0.0
    for pp in peak_pieces:
        if used + pp.dur > budget * P.peak_share:
            break
        chosen_peaks.append(pp)
        used += pp.dur
    gap = P.max_gap_s
    best: list[_Piece] | None = None
    for _ in range(12):
        spine = _spine_pieces(A, P, gap, chosen_peaks + pieces)
        total = fixed_dur + sum(pp.dur for pp in chosen_peaks) + sum(sp.dur for sp in spine)
        best = pieces + chosen_peaks + spine
        if abs(total - (P.runtime_target_s - (reactor.branding.endcard_duration or 0))) <= P.tolerance_s:
            break
        if total > P.runtime_target_s:
            # over: widen gap first, then drop weakest peaks
            if gap < 240:
                gap *= 1.25
            elif chosen_peaks:
                chosen_peaks.pop()
            else:
                break
        else:
            # under: add peaks, then tighten gap
            remaining = [pp for pp in peak_pieces if pp not in chosen_peaks]
            if remaining:
                chosen_peaks.append(remaining[0])
            elif gap > 20:
                gap *= 0.8
            else:
                break
    assert best is not None
    edl = _assemble(best, A, P, source, reactor, title)
    edl.meta.update({"generator": "select v1", "runtime_target_s": P.runtime_target_s, "clip_cap_s": P.clip_cap_s,
                     "peaks_used": len(chosen_peaks), "peaks_available": len(peak_pieces), "spine_gap_s": round(gap, 1),
                     "withheld": [round(p["t"], 1) for p in withheld], "film_start": round(A.film_start, 1),
                     "film_end": round(A.film_end, 1)})
    return edl


def _peak_piece(p: dict, idx: int, A: Analysis, P: SelectParams) -> _Piece | None:
    t0, t1 = p["t0"], p["t1"]
    dur = min(max(t1 - t0, P.reaction_min_s), P.reaction_max_s)
    r_in = max(A.film_start, p["t"] - dur * 0.35)
    r_out = min(A.film_end, r_in + dur)
    segs: list[Segment] = []
    song = A.song_overlap(r_in - P.clip_cap_s, r_in) > 0.5
    if not song:
        m_out = r_in
        m_in = max(A.film_start, m_out - P.clip_cap_s)
        m_in = max(m_in, A.snap_to_cut(m_in, 1.5))
        if m_out - m_in >= 2.0:
            segs.append(Segment(id=f"p{idx:03d}m", **{"in": m_in}, out=m_out, layout="movie-large", kind="story",
                                score=p["score"], note=f"trigger for peak {idx} ({p['kind']})"))
    segs.append(Segment(id=f"p{idx:03d}r", **{"in": r_in}, out=r_out, layout="reactor-large", kind="reaction",
                        score=p["score"], note=f"peak {idx} ({p['kind']}, {p['score']:.2f})" + (" — song under it, no movie slice" if song else "")))
    return _Piece(anchor=segs[0].in_, segs=segs, kind="peak", score=p["score"], note=f"peak {idx}")


def _spine_pieces(A: Analysis, P: SelectParams, gap: float, existing: list[_Piece]) -> list[_Piece]:
    """Fill uncovered film stretches: whenever more than ``gap`` seconds of film would pass without any
    coverage, drop a spine slice roughly ``0.6·gap`` in (searching ±20 s around that anchor)."""
    covered = sorted((s.in_, s.out) for pc in existing for s in pc.segs)
    out: list[_Piece] = []
    t = A.film_start
    k = 0
    while t < A.film_end - 10:
        nxt = next((c for c in covered if c[0] >= t), None)
        next_cov = nxt[0] if nxt else A.film_end
        if next_cov - t <= gap:
            t = max(t + 1.0, nxt[1] if nxt else A.film_end)
            continue
        anchor = t + gap * 0.6
        piece = _spine_at(anchor, k, A, P, lo=max(t + 1.0, anchor - 20), until=min(next_cov, anchor + 20))
        if piece:
            out.append(piece)
            k += 1
            t = max(piece.segs[-1].out, anchor)
        else:
            t = anchor
    return out


def _spine_at(anchor: float, k: int, A: Analysis, P: SelectParams, lo: float, until: float) -> _Piece | None:
    """Pick the best ≤cap movie slice around ``anchor`` (within [lo, until]): densest FILM dialogue
    near a scene cut, not in a song span, not dead air."""
    lo, hi = max(A.film_start, lo, anchor - 25), min(until, anchor + 25)
    if hi - lo < 3.0:
        return None
    cands = []
    for s in A.film_segments(lo, hi):
        start = max(lo, A.snap_to_cut(s["start"] - 0.3, 1.5))
        end = min(start + P.spine_slice_s, A.film_end, until)
        if end - start < 2.5:
            continue
        words = len((s.get("text") or "").split())
        density = words / max(1.0, s["end"] - s["start"])
        score = density
        if A.song_overlap(start, end) > 0.5:
            continue
        if A.in_dead(start):
            score *= 0.3
        score -= 0.5 * A.score_overlap(start, end) / P.spine_slice_s
        cands.append((score, start, end, s))
    if not cands:
        start = max(lo, A.snap_to_cut(anchor, 2.0))
        end = min(start + P.spine_slice_s, until, A.film_end)
        if end - start < 2.5 or A.song_overlap(start, end) > 0.5:
            return None
        cands.append((0.0, start, end, None))
    score, start, end, seg = max(cands, key=lambda c: c[0])
    segs = [Segment(id=f"s{k:03d}m", **{"in": start}, out=end, layout="movie-large", kind="story", score=score,
                    note="spine slice" + (f": {seg['text'][:60]}" if seg and seg.get("text") else ""))]
    # short reactor cutaway if he speaks right after
    rs = A.reactor_segments(end, end + 8.0)
    if rs:
        r = rs[0]
        a = max(end, r["start"] - 0.3)
        b = min(a + P.spine_cutaway_s, r["end"] + 0.3, A.film_end)
        if b - a >= 1.2:
            segs.append(Segment(id=f"s{k:03d}r", **{"in": a}, out=b, layout="reactor-large", kind="reaction",
                                note=f"cutaway: {r['text'][:60]}"))
    return _Piece(anchor=start, segs=segs, kind="spine", score=score, note="spine")


def _assemble(pieces: list[_Piece], A: Analysis, P: SelectParams, source: str, reactor: ReactorConfig, title: TitleConfig) -> EDL:
    segs = [s for pc in sorted(pieces, key=lambda p: p.anchor) for s in pc.segs]
    segs.sort(key=lambda s: s.in_)
    # de-overlap chronologically
    cleaned: list[Segment] = []
    for s in segs:
        if cleaned and s.in_ < cleaned[-1].out:
            if s.out - cleaned[-1].out < 1.0:
                continue
            s = s.model_copy(update={"in_": cleaned[-1].out})
        if s.dur < 0.8:
            continue
        cleaned.append(s)
    # hysteresis: merge same-layout neighbours that are contiguous in source time
    merged: list[Segment] = []
    for s in cleaned:
        if merged and merged[-1].layout == s.layout and abs(merged[-1].out - s.in_) < 0.05 and merged[-1].kind == s.kind:
            merged[-1] = merged[-1].model_copy(update={"out": s.out, "note": merged[-1].note + " + " + s.note})
        else:
            merged.append(s)
    # enforce clip cap for movie-large (split into cap-sized slices with a tiny reactor cutaway is
    # not possible without extra material → hard-trim to cap)
    final: list[Segment] = []
    for s in merged:
        if s.layout == "movie-large" and s.dur > P.clip_cap_s + 1e-6:
            s = s.model_copy(update={"out": s.in_ + P.clip_cap_s, "note": s.note + " (trimmed to clip cap)"})
        if s.dur < P.layout_min_s and s.kind not in ("cta",):
            # too short for hysteresis: drop unless it is the only thing here
            if final and final[-1].layout == s.layout:
                continue
            if s.dur < 1.0:
                continue
        final.append(s)
    # chapters roughly every 10 min of output at story segments
    out_t, last_ch, n = 0.0, -1e9, 1
    for s in final:
        if s.chapter is None and s.kind == "story" and out_t - last_ch >= 600:
            s.chapter = f"Part {n}"
            n += 1
            last_ch = out_t
        elif s.chapter is not None:
            last_ch = out_t
        out_t += s.dur
    # renumber ids chronologically but keep meaning
    for i, s in enumerate(final, 1):
        s.id = f"{i:03d}_{s.id}"
    overlays = []
    total = sum(s.dur for s in final)
    ctas = [s for s in final if s.kind == "cta"]
    off = 0.0
    for s in final:
        if s.kind == "cta":
            overlays.append(Overlay(at=round(off, 2), dur=min(reactor.branding.lower_third_duration, s.dur)))
        off += s.dur
    if not ctas and total > 600:
        overlays.append(Overlay(at=round(total * 0.45, 2), dur=reactor.branding.lower_third_duration))
    return EDL(source=source, target=RenderTarget(fps=P.fps), segments=final, overlays=overlays,
               endcard=Endcard(dur=reactor.branding.endcard_duration), meta={"title": title.title})
