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
from ..edl import EDL, Card, Endcard, Overlay, Segment


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
    wt: np.ndarray = field(default_factory=lambda: np.zeros(0))       # timeline window centres
    wdb: np.ndarray = field(default_factory=lambda: np.zeros(0))      # per-window RMS dB
    wreact: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    beats: list[dict] = field(default_factory=list)          # narrative beats (t0,t1,label,importance[,priority])
    key_lines: list[dict] = field(default_factory=list)      # must/should dialogue lines (t0,t1,heard,beat,priority)

    @classmethod
    def load(cls, adir: Path, duration: float) -> "Analysis":
        def rd(name, default):
            p = adir / name
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

        sp = rd("speakers.json", {"segments": []})
        segs = sp.get("segments") or rd("transcript.json", {"segments": []}).get("segments", [])
        tl = sp.get("timeline", [])
        wt = np.asarray([w["t"] for w in tl], dtype=float)
        wdb = np.asarray([w["db"] for w in tl], dtype=float)
        key = sp.get("score_key", "sim")
        thr = float(sp.get("threshold", 0.0))
        wreact = np.asarray([(w.get(key) is not None and w[key] >= thr) for w in tl], dtype=bool)
        peaks = rd("peaks.json", {"peaks": []})["peaks"]
        music = rd("music.json", {"spans": []})["spans"]
        scenes_d = rd("scenes.json", {"cuts": []})
        cuts = np.asarray(scenes_d["cuts"], dtype=float)
        movie_change = np.asarray(scenes_d.get("movie_change", []), dtype=np.float32)
        sig_fps = float(scenes_d.get("fps", 5.0))
        dead = rd("deadair.json", {"spans": []})["spans"]
        narr = rd("narrative.json", None)
        if narr:
            beats = narr.get("beats", [])
            key_lines = narr.get("key_lines", [])
        else:
            beats = rd("beats.json", {"beats": []})["beats"]
            key_lines = []
        fs, fe = _film_bounds(segs, cuts, duration, movie_change=movie_change, sig_fps=sig_fps)
        return cls(duration=duration, segments=segs, peaks=peaks, music=music, cuts=cuts, dead=dead,
                   film_start=fs, film_end=fe, wt=wt, wdb=wdb, wreact=wreact, beats=beats, key_lines=key_lines)

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

    def beat_at(self, t: float) -> dict | None:
        for b in self.beats:
            if b["t0"] <= t <= b["t1"]:
                return b
        return None

    def keep_spans_without_silence(self, a: float, b: float, min_silence: float,
                                   pad: float = 0.35, min_keep: float = 1.0) -> list[tuple[float, float]]:
        """Split [a, b] into spans, dropping silent stretches longer than ``min_silence`` seconds.
        Silence = timeline windows below the speakers-stage silence floor. Falls back to [(a, b)]."""
        if self.wt.size == 0 or min_silence <= 0:
            return [(a, b)]
        m = (self.wt >= a) & (self.wt <= b)
        if not m.any():
            return [(a, b)]
        ts, dbs = self.wt[m], self.wdb[m]
        silent = dbs < -42.0
        spans: list[tuple[float, float]] = []
        cur = a
        i = 0
        n = len(ts)
        while i < n:
            if silent[i]:
                j = i
                while j < n and silent[j]:
                    j += 1
                s0 = ts[i] - 0.25
                s1 = (ts[j - 1] + 0.25) if j < n else b
                if s1 - s0 >= min_silence:
                    if s0 + pad - cur >= min_keep:
                        spans.append((cur, s0 + pad))
                    cur = max(cur, s1 - pad)
                i = j
            else:
                i += 1
        if b - cur >= min_keep:
            spans.append((cur, b))
        return spans or [(a, b)]

    def quiet_point(self, t: float, radius: float = 1.2) -> float:
        """Nearest time within ±radius where the mixed audio is locally quiet — cut there instead of
        mid-dialogue. Falls back to t."""
        if self.wt.size == 0:
            return t
        m = (self.wt >= t - radius) & (self.wt <= t + radius)
        if not m.any():
            return t
        idx = np.where(m)[0]
        best = idx[int(np.argmin(self.wdb[idx]))]
        # only move if it is meaningfully quieter than the immediate point
        here = self.wdb[int(np.argmin(np.abs(self.wt - t)))]
        return float(self.wt[best]) if self.wdb[best] < here - 3.0 else t

    def avoid_flash(self, a: float, b: float, guard: float = 0.6) -> tuple[float, float]:
        """Nudge [a, b] so neither endpoint sits a few frames on the wrong side of a scene cut
        (which reads as a 'flash' of the neighbouring scene)."""
        if self.cuts.size:
            after = self.cuts[(self.cuts > a) & (self.cuts < a + guard)]
            if after.size:
                a = float(after[0]) + 0.06        # start cleanly on the new scene
            before = self.cuts[(self.cuts > b - guard) & (self.cuts < b)]
            if before.size:
                b = float(before[-1]) - 0.04      # end before the next scene sneaks in
        return a, b

    def loud_film_end(self, a: float, b: float) -> float | None:
        """If [a, b] contains a loud non-reactor stretch (gunshot, crash, bat on mailbox …), return
        when it ends — the movie should stay on screen until then."""
        if self.wt.size == 0:
            return None
        m = (self.wt >= a) & (self.wt <= b)
        if not m.any():
            return None
        loudness = self.wdb[m]
        loud_thr = np.percentile(self.wdb[self.wdb > -80], 88)
        loud = (loudness >= loud_thr) & ~self.wreact[m]
        if not loud.any():
            return None
        return float(self.wt[m][np.where(loud)[0][-1]] + 0.8)

    def film_segments(self, a: float, b: float) -> list[dict]:
        return [s for s in self.segments if s.get("speaker") in ("FILM", "MIXED", None, "?") and s["end"] > a and s["start"] < b]

    def reactor_segments(self, a: float, b: float) -> list[dict]:
        return [s for s in self.segments if s.get("speaker") == "REACTOR" and s["end"] > a and s["start"] < b]


def _film_bounds(segs: list[dict], cuts: np.ndarray, duration: float,
                 movie_change: np.ndarray | None = None, sig_fps: float = 5.0) -> tuple[float, float]:
    """Where does the film start/end inside the recording?

    Primary cue (when video signals exist): the longest *sustained* activity run in the movie region —
    browsing/menus produce bursts, a playing film produces near-continuous change; boundaries are
    snapped to scene cuts. Fallback: first/last sustained FILM-tagged speech. Manual override
    (``rae set-film-bounds``) beats both — studio logos, credits-over-imagery and animated menus can
    fool any heuristic."""
    if movie_change is not None and movie_change.size > sig_fps * 600:
        active = (movie_change > 0.35).astype(np.float32)
        k = int(90 * sig_fps)
        frac = np.convolve(active, np.ones(k) / k, mode="same")
        on = frac > 0.65
        runs, st = [], None
        for i, f in enumerate(on):
            if f and st is None:
                st = i
            elif not f and st is not None:
                runs.append((st, i)); st = None
        if st is not None:
            runs.append((st, len(on)))
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
            fs, fe = a / sig_fps, b / sig_fps
            if fe - fs > 0.4 * duration:                 # plausible feature film
                if cuts.size:
                    ca = cuts[(cuts > fs - 20) & (cuts < fs + 8)]
                    if ca.size:
                        fs = float(ca[0])                # snap start to the first nearby cut (logo pop-on)
                    cb = cuts[(cuts > fe - 60) & (cuts < fe + 5)]
                    if cb.size:
                        fe = float(cb[0])                # credits: first cut at the end of the run
                return max(0.0, fs), min(duration, fe)
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
    silence_cut_s: float | None = None # drop silent stretches longer than this from intro/outro (None = off)
    trim_intro: bool = False           # False = uncut: the whole pre-film stretch, full-frame
    trim_outro: bool = False           # False = uncut: the whole post-film stretch, full-frame
    intro_max_s: float = 150.0         # cap when trimming
    outro_max_s: float = 240.0         # cap when trimming
    movie_frac: float = 0.75           # target fraction of the in-film content that is movie-large
    reaction_min_s: float = 2.5
    reaction_max_s: float = 9.0
    spine_slice_s: float = 7.0            # ≤ clip cap (clamped)
    spine_cutaway_s: float = 3.0
    max_gap_s: float = 75.0            # film seconds allowed without any coverage (tuned by runtime loop)
    split_cutaway_s: float = 1.8       # reactor-cam visual during must-scene splits (film audio continues)
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
           title: TitleConfig | None = None, film_bounds: tuple[float, float] | None = None) -> EDL:
    A, P = analysis, params
    if film_bounds is not None:
        A.film_start, A.film_end = film_bounds
        A.notes = getattr(A, "notes", []) + ["film bounds set manually"]
    P.spine_slice_s = min(P.spine_slice_s, P.clip_cap_s)
    reactor = reactor or ReactorConfig()
    title = title or TitleConfig()
    pieces: list[_Piece] = []

    # ---- intro / outro (full-frame: the streamer's own composite layout IS the shot) ---------
    # default UNCUT: the whole pre-film / post-film stretch. With trim_intro/trim_outro, cut down
    # to his speech ("let's get into it" … first words → last words), capped.
    intro_segs = A.reactor_segments(0.0, A.film_start)
    if P.trim_intro:
        if intro_segs and P.intro_max_s > 0:
            a = max(0.0, intro_segs[0]["start"] - 0.4)
            b = min(intro_segs[-1]["end"] + 0.6, A.film_start, a + P.intro_max_s)
        else:
            a = b = 0.0
        note = "intro (trimmed to his monologue)"
    else:
        a, b = 0.0, A.film_start
        note = "intro (uncut)"
    if b - a >= 3.0:
        spans = A.keep_spans_without_silence(a, b, P.silence_cut_s or 0.0)
        if len(spans) > 1:
            note += f" — {len(spans) - 1} silence(s) > {P.silence_cut_s:.1f}s cut"
        pieces.append(_Piece(anchor=spans[0][0], kind="intro", note=note,
                             segs=[Segment(id=f"intro{k}" if k else "intro", **{"in": round(x, 2)}, out=round(y, 2),
                                           layout="full", kind="intro", chapter="Intro" if k == 0 else None, note=note)
                                   for k, (x, y) in enumerate(spans)]))
    outro_segs = A.reactor_segments(A.film_end, A.duration)
    if P.trim_outro:
        if outro_segs and P.outro_max_s > 0:
            a = max(A.film_end, outro_segs[0]["start"] - 0.4)
            b = min(outro_segs[-1]["end"] + 0.8, A.duration, a + P.outro_max_s)
        else:
            a = b = A.film_end
        note = "outro (trimmed to his wrap-up)"
    else:
        a, b = A.film_end, A.duration
        note = "outro (uncut)"
    if b - a >= 3.0:
        spans = A.keep_spans_without_silence(a, b, P.silence_cut_s or 0.0)
        if len(spans) > 1:
            note += f" — {len(spans) - 1} silence(s) > {P.silence_cut_s:.1f}s cut"
        pieces.append(_Piece(anchor=spans[0][0], kind="outro", note=note,
                             segs=[Segment(id=f"outro{k}" if k else "outro", **{"in": round(x, 2)}, out=round(y, 2),
                                           layout="full", kind="outro", chapter="Final thoughts" if k == 0 else None,
                                           note=note) for k, (x, y) in enumerate(spans)]))

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

    # ---- Budget A first: narrative musts (then shoulds), under the movie/reactor ratio -------
    musts = _narrative_pieces(A, P, ("must",))
    shoulds = _narrative_pieces(A, P, ("should",)) if (A.key_lines or A.beats) else []
    # de-conflict shoulds that overlap musts
    mspans = [(pc.segs[0].in_, pc.segs[-1].out) for pc in musts]
    shoulds = [pc for pc in shoulds if not any(a - 3 < pc.anchor < b + 3 for a, b in mspans)]

    fixed_dur = sum(pc.dur for pc in pieces)              # intro/outro/cta
    content_target = P.runtime_target_s - fixed_dur - (reactor.branding.endcard_duration or 0)
    b_target = content_target * (1.0 - P.movie_frac)

    chosen_peaks: list[_Piece] = []
    b_used = _b_time(musts)
    for pp in peak_pieces:
        pb = _b_time([pp])
        if b_used + pb > b_target:
            continue
        chosen_peaks.append(pp)
        b_used += pb
    chosen_shoulds = list(shoulds)
    gap = P.max_gap_s
    best: list[_Piece] | None = None
    for _ in range(24):
        base = pieces + musts + chosen_shoulds + chosen_peaks
        spine = _spine_pieces(A, P, gap, base)
        total = sum(pc.dur for pc in base) + sum(sp.dur for sp in spine)
        best = base + spine
        if abs(total - (P.runtime_target_s - (reactor.branding.endcard_duration or 0))) <= P.tolerance_s:
            break
        if total > P.runtime_target_s:
            if gap < 240:
                gap *= 1.25
            elif chosen_peaks:
                chosen_peaks.pop()          # weakest-last ordering: peaks were added by score
            elif chosen_shoulds:
                chosen_shoulds.pop()
            else:
                break
        else:
            short = (P.runtime_target_s - (reactor.branding.endcard_duration or 0)) - total
            remaining = [pp for pp in peak_pieces if pp not in chosen_peaks]
            added = False
            for pp in remaining:
                pb = _b_time([pp])
                if short <= 0 or _b_time(musts + chosen_peaks + [pp]) > b_target:
                    continue
                chosen_peaks.append(pp)
                short -= pp.dur
                added = True
            if not added:
                if gap > 8:
                    gap *= max(0.6, 1.0 - short / max(1.0, total))
                else:
                    break
    assert best is not None
    edl = _assemble(best, A, P, source, reactor, title)
    a_t, b_t = _a_time(best), _b_time(best)
    edl.meta.update({"generator": "select v2 (narrative)", "runtime_target_s": P.runtime_target_s,
                     "clip_cap_s": P.clip_cap_s, "movie_frac_target": P.movie_frac,
                     "movie_frac_actual": round(a_t / max(1.0, a_t + b_t), 3),
                     "narrative": {"musts": len(musts), "shoulds_used": len(chosen_shoulds),
                                   "key_lines": len(A.key_lines), "beats": len(A.beats)},
                     "peaks_used": len(chosen_peaks), "peaks_available": len(peak_pieces), "spine_gap_s": round(gap, 1),
                     "withheld": [round(p["t"], 1) for p in withheld], "film_start": round(A.film_start, 1),
                     "film_end": round(A.film_end, 1)})
    return edl


def _b_time(pieces: list) -> float:
    return sum(s.dur for pc in pieces for s in pc.segs if s.layout == "reactor-large")


def _a_time(pieces: list) -> float:
    return sum(s.dur for pc in pieces for s in pc.segs if s.layout == "movie-large")


def _cover_span(a: float, b: float, tag: str, A: Analysis, P: SelectParams, *, score: float,
                note: str, chapter: str | None = None) -> _Piece:
    """Cover [a, b] chronologically under the clip cap: movie-large slices, with short reactor-cam
    cutaways between them. The mixed audio track runs uninterrupted through the whole span — a split
    only changes what is on SCREEN, never the dialogue — so long must-lines complete naturally."""
    segs: list[Segment] = []
    t = a
    k = 0
    while t < b - 0.4:
        m_out = min(t + P.clip_cap_s, b)
        if k == 0:
            t2, m_out = A.avoid_flash(t, m_out)
            if t2 < b - 0.4:
                t = t2
        if m_out - t >= 1.2:
            segs.append(Segment(id=f"{tag}m{k}", **{"in": round(t, 2)}, out=round(m_out, 2), layout="movie-large",
                                kind="story", score=score, chapter=chapter if k == 0 else None, note=note))
        t = m_out
        k += 1
        if t < b - 0.4:  # visual breather; audio (the film line) continues underneath
            c_out = min(t + P.split_cutaway_s, b - 0.2)
            if c_out - t >= 1.0:
                segs.append(Segment(id=f"{tag}c{k}", **{"in": round(t, 2)}, out=round(c_out, 2),
                                    layout="reactor-large", kind="reaction", score=score,
                                    note=note + " (split cutaway; film audio continues)"))
                t = c_out
    if not segs:
        segs = [Segment(id=f"{tag}m0", **{"in": round(a, 2)}, out=round(max(a + 1.5, b), 2), layout="movie-large",
                        kind="story", score=score, chapter=chapter, note=note)]
    return _Piece(anchor=segs[0].in_, segs=segs, kind="narrative", score=score, note=note)


def _narrative_pieces(A: Analysis, P: SelectParams, priorities: tuple[str, ...]) -> list[_Piece]:
    """Coverage pieces for narrative musts/shoulds: matched key lines framed exactly; beats without
    matched lines get one dialogue-dense slice inside their span."""
    out: list[_Piece] = []
    covered: list[tuple[float, float]] = []
    lines = [k for k in A.key_lines if k.get("priority") in priorities]
    for i, k in enumerate(lines):
        a = max(A.film_start, k["t0"] - 1.2)
        b = min(A.film_end, k["t1"] + 0.8)
        a = A.quiet_point(A.snap_to_cut(a, 1.2))
        if any(x0 - 2 < a < x1 + 2 for x0, x1 in covered):
            continue
        if A.song_overlap(a, b) > 0.5 * (b - a):
            continue
        out.append(_cover_span(a, b, f"n{i:03d}", A, P, score=3.0,
                               note=f"key line [{k.get('priority')}]: {str(k.get('heard') or k.get('expected'))[:60]}"))
        covered.append((a, b))
    beat_labels_with_lines = {k.get("beat") for k in lines}
    for j, bt in enumerate(x for x in A.beats if x.get("priority") in priorities):
        if bt.get("label") in beat_labels_with_lines:
            continue
        mid = (bt["t0"] + bt["t1"]) / 2
        if any(x0 - 5 < mid < x1 + 5 for x0, x1 in covered):
            continue
        piece = _spine_at(mid, 900 + j, A, P, lo=max(A.film_start, bt["t0"]), until=min(A.film_end, bt["t1"]))
        if piece:
            piece.kind = "narrative"
            piece.note = f"beat [{bt.get('priority')}]: {bt.get('label', '')}"
            piece.segs[0].note = piece.note + " — " + piece.segs[0].note
            out.append(piece)
            covered.append((piece.segs[0].in_, piece.segs[-1].out))
    out.sort(key=lambda pc: pc.anchor)
    return out


def _peak_piece(p: dict, idx: int, A: Analysis, P: SelectParams) -> _Piece | None:
    t0, t1 = p["t0"], p["t1"]
    dur = min(max(t1 - t0, P.reaction_min_s), P.reaction_max_s)
    r_in = max(A.film_start, p["t"] - dur * 0.35)
    # loud film event (gunshot, crash …) inside the trigger window → the movie stays on screen
    # until the event ends; his reaction follows it.
    loud_end = A.loud_film_end(r_in - 3.0, min(p["t"] + 2.0, t1))
    note_extra = ""
    if loud_end is not None and loud_end > r_in:
        r_in = min(loud_end, p["t"] + 2.0)
        note_extra = " — after loud film moment"
    r_out = min(A.film_end, r_in + dur)
    segs: list[Segment] = []
    song = A.song_overlap(r_in - P.clip_cap_s, r_in) > 0.5
    if not song:
        m_out = r_in
        m_in = max(A.film_start, m_out - P.clip_cap_s)
        m_in = max(m_in, A.snap_to_cut(m_in, 1.5))
        m_in, m_out = A.avoid_flash(m_in, m_out)
        m_in = A.quiet_point(m_in)
        if m_out - m_in >= 2.0:
            segs.append(Segment(id=f"p{idx:03d}m", **{"in": round(m_in, 2)}, out=round(m_out, 2), layout="movie-large",
                                kind="story", score=p["score"], note=f"trigger for peak {idx} ({p['kind']}){note_extra}"))
            r_in = m_out
    r_out = A.quiet_point(r_out)
    if r_out - r_in < P.reaction_min_s:
        r_out = min(A.film_end, r_in + P.reaction_min_s)
    segs.append(Segment(id=f"p{idx:03d}r", **{"in": round(r_in, 2)}, out=round(r_out, 2), layout="reactor-large", kind="reaction",
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
        beat = A.beat_at((start + end) / 2)
        if beat:
            score += 2.0 * float(beat.get("importance", 0.5))
        cands.append((score, start, end, s))
    if not cands:
        start = max(lo, A.snap_to_cut(anchor, 2.0))
        end = min(start + P.spine_slice_s, until, A.film_end)
        if end - start < 2.5 or A.song_overlap(start, end) > 0.5:
            return None
        cands.append((0.0, start, end, None))
    score, start, end, seg = max(cands, key=lambda c: c[0])
    start, end = A.avoid_flash(start, end)
    end = A.quiet_point(end)
    if end - start < 2.5:
        return None
    beat = A.beat_at((start + end) / 2)
    segs = [Segment(id=f"s{k:03d}m", **{"in": round(start, 2)}, out=round(end, 2), layout="movie-large", kind="story", score=score,
                    chapter=(beat["label"] if beat and beat.get("importance", 0) >= 0.7 else None),
                    note=("beat: " + beat["label"] + " — " if beat else "") + "spine slice" + (f": {seg['text'][:60]}" if seg and seg.get("text") else ""))]
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
    # anti flip-flop: drop a short segment sandwiched between two segments of the other layout
    # (e.g. 1 s of movie between two reactor shots, or vice versa)
    for _ in range(2):
        drop = set()
        for i in range(1, len(merged) - 1):
            a, b, c = merged[i - 1], merged[i], merged[i + 1]
            if b.kind in ("cta", "intro", "outro"):
                continue
            if a.layout == c.layout != b.layout and b.dur < max(P.layout_min_s, 2.2):
                drop.add(i)
        if not drop:
            break
        merged = [s for i, s in enumerate(merged) if i not in drop]
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
    seen_labels: set[str] = set()
    for s in final:
        if s.chapter and s.kind == "story":
            if s.chapter in seen_labels:
                s.chapter = None            # a beat announces itself once
            else:
                seen_labels.add(s.chapter)
    have_beat_chapters = any(s.chapter for s in final if s.kind == "story")
    out_t, last_ch, n = 0.0, -1e9, 1
    for s in final:
        if have_beat_chapters:
            if s.chapter is not None:
                if out_t - last_ch < 240:   # chapters too dense — keep the first of the cluster
                    s.chapter = None
                else:
                    last_ch = out_t
            out_t += s.dur
            continue
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
    # transition: dip to black between intro and the film
    for i, s in enumerate(final):
        if i > 0 and final[i - 1].kind == "intro":
            final[i] = s.model_copy(update={"transition": "xfade"})
        if s.kind == "outro" and i > 0:
            final[i] = s.model_copy(update={"transition": "xfade"})
    # lower-third schedule: at CTAs, plus every `every_min` minutes, never two within `min_gap_min`
    sched = reactor.branding.lower_third_schedule
    total = sum(s.dur for s in final)
    shown: list[float] = []
    overlays = []
    off = 0.0
    for s in final:
        if s.kind == "cta":
            overlays.append(Overlay(at=round(off, 2), dur=min(reactor.branding.lower_third_duration, s.dur)))
            shown.append(off)
        off += s.dur
    if sched.at_min is not None:
        marks = [m * 60.0 for m in sched.at_min]
    else:
        marks, tmark = [], sched.start_min * 60.0
        while tmark < total - 120:
            marks.append(tmark)
            tmark += sched.every_min * 60.0
    for tmark in marks:
        if tmark < total - 30 and all(abs(tmark - x) >= sched.min_gap_min * 60.0 for x in shown):
            overlays.append(Overlay(at=round(tmark, 2), dur=reactor.branding.lower_third_duration))
            shown.append(tmark)
    overlays.sort(key=lambda o: o.at)
    # title card between intro and film (image provided per title or via reactor branding)
    card = (title.title_card or reactor.branding.title_card) if title.show_title_card else None
    edl = EDL(source=source, target=RenderTarget(fps=P.fps), segments=final, overlays=overlays,
              endcard=Endcard(dur=reactor.branding.endcard_duration), meta={"title": title.title})
    if card:
        intro_idx = next((i for i, s in enumerate(final) if s.kind == "intro"), None)
        edl.cards = [Card(before_id=final[intro_idx + 1].id if intro_idx is not None and intro_idx + 1 < len(final) else None,
                          template=card, dur=3.0)]
    return edl
