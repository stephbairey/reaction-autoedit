"""Speaker attribution: who is talking — the REACTOR or the FILM?

This is the tool's hardest problem (single mixed track, film dialogue under his voice), so it is
built as a swappable backend behind one interface and validated on real footage before anything
is built on top of it.

Backend v1: **resemblyzer** speaker embeddings (pip-only, CPU-friendly).
  * enrol once from a clean voice sample → reference embedding + calibrated threshold
  * slide 1.6 s windows (hop 0.5 s) over the mixed track → cosine similarity to the reference
    ("timeline"); silent windows are gated out by RMS
  * **in-domain adaptation**: the clean enrolment only *seeds* confident REACTOR / FILM windows on
    the real track; both centroids are re-estimated there and each window is scored by the contrast
    ``score = cos(e, reactor) - cos(e, film)`` (his voice inside the mix vs. the film's mix). The
    threshold is Otsu's split of that score histogram; a drift guard keeps the reactor centroid
    anchored to the clean enrolment
  * **audio-visual fusion** (when ``face_motion.json`` exists): z(score) + 0.5·z(mouth-region motion)
    — his mouth moves when he talks; on labelled data this lifts AUC noticeably
  * every transcript segment is labelled from the windows it overlaps:
      REACTOR (≥60 % windows above threshold) | FILM (≤15 %) | MIXED (in between) | UNKNOWN (silent)

Backend v2 (planned): pyannote diarisation mapped to the enrolment (needs an HF token).

Output ``speakers.json``::

    {"backend": "resemblyzer", "threshold": 0.71, "margin": 0.04,
     "enrollment": {"sample": "...", "n_partials": 90, "sim_p10": 0.83, "sim_median": 0.9},
     "timeline": [{"t": 12.5, "sim": 0.81, "score": 0.12, "db": -28.1}, ...],  # t = window centre
     "segments": [{"id": 0, "start": 12.3, "end": 15.1, "speaker": "REACTOR",
                   "reactor_frac": 0.9, "sim_max": 0.88, "sim_mean": 0.82, "text": "..."}]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .audio import SR, load_wav, rms_db
from .face_motion import FaceMotion

WIN_S = 1.6
HOP_S = 0.5
SILENCE_DB = -42.0
LABELS = ("REACTOR", "FILM", "MIXED", "UNKNOWN")


@dataclass
class Enrollment:
    embedding: np.ndarray
    sims: np.ndarray                 # similarity of each enrolment partial to the mean embedding
    sample: str
    threshold: float = 0.7
    margin: float = 0.04
    extra: dict = field(default_factory=dict)

    def stats(self) -> dict:
        return {"sample": self.sample, "n_partials": int(self.sims.size),
                "sim_p10": round(float(np.percentile(self.sims, 10)), 3) if self.sims.size else None,
                "sim_median": round(float(np.median(self.sims)), 3) if self.sims.size else None,
                **self.extra}


class SpeakerTagger(Protocol):
    name: str

    def enroll(self, sample_paths: list[Path]) -> Enrollment: ...

    def timeline(self, wav: np.ndarray, sr: int, offset: float, enrol: Enrollment,
                 progress: Callable[[float, str], None] | None = None) -> list[dict]: ...


class ResemblyzerTagger:
    name = "resemblyzer"

    def __init__(self, device: str = "cpu"):
        from resemblyzer import VoiceEncoder  # heavy import

        self._enc = VoiceEncoder(device=device, verbose=False)

    # ---- enrolment ------------------------------------------------------------------------
    def enroll(self, sample_paths: list[Path]) -> Enrollment:
        from resemblyzer import preprocess_wav

        partials = []
        for p in sample_paths:
            wav = preprocess_wav(str(p))          # normalises volume, trims long silences, 16 kHz
            _, parts, _ = self._enc.embed_utterance(wav, return_partials=True, rate=2)
            partials.append(parts)
        allp = np.concatenate(partials, axis=0)
        ref = allp.mean(axis=0)
        ref /= np.linalg.norm(ref) + 1e-9
        sims = allp @ ref
        e = Enrollment(embedding=ref, sims=sims, sample=";".join(str(p) for p in sample_paths))
        # provisional threshold; refined against the real track in calibrate()
        e.threshold = float(np.clip(np.percentile(sims, 10) - 0.10, 0.60, 0.80))
        return e

    # ---- timeline ---------------------------------------------------------------------------
    def timeline(self, wav: np.ndarray, sr: int, offset: float, enrol: Enrollment,
                 progress: Callable[[float, str], None] | None = None) -> list[dict]:
        tl, _ = self.timeline_with_embeddings(wav, sr, offset, enrol, progress)
        return tl

    def timeline_with_embeddings(self, wav: np.ndarray, sr: int, offset: float, enrol: Enrollment,
                                 progress: Callable[[float, str], None] | None = None) -> tuple[list[dict], np.ndarray]:
        """Sliding-window similarity timeline; also returns the (n_windows, 256) embedding matrix
        (NaN rows for silent windows) so callers can adapt centroids in-domain."""
        from resemblyzer.audio import normalize_volume

        assert sr == SR
        chunk_s = 600.0                              # keep the LSTM batch bounded
        win, hop = int(WIN_S * sr), int(HOP_S * sr)
        out: list[dict] = []
        emb_rows: list[np.ndarray] = []
        n = len(wav)
        pos = 0
        while pos < n:
            end = min(n, pos + int(chunk_s * sr) + win)  # overlap by one window
            chunk = wav[pos:end]
            if len(chunk) < win:
                break
            starts = np.arange(0, len(chunk) - win + 1, hop)
            frames = np.stack([chunk[s:s + win] for s in starts])
            dbs = np.array([rms_db(f) for f in frames])
            voiced = dbs > SILENCE_DB
            sims = np.full(len(starts), np.nan, dtype=np.float32)
            embs_all = np.full((len(starts), 256), np.nan, dtype=np.float32)
            if voiced.any():
                # normalise each window individually (loudness-independent), embed in one batch
                normed = [normalize_volume(f.astype(np.float32), -30, increase_only=True) for f in frames[voiced]]
                embs = self._embed_batch(normed)
                sims[voiced] = embs @ enrol.embedding
                embs_all[voiced] = embs
            for s, d, sm in zip(starts, dbs, sims):
                t = offset + (pos + s + win / 2) / sr
                out.append({"t": round(float(t), 2), "sim": None if np.isnan(sm) else round(float(sm), 3), "db": round(float(d), 1)})
            emb_rows.append(embs_all)
            pos += int(chunk_s * sr)
            if progress:
                progress(min(1.0, pos / n), f"{min(pos, n)/sr/60:.1f}/{n/sr/60:.1f} min embedded")
        E = np.concatenate(emb_rows, axis=0) if emb_rows else np.zeros((0, 256), np.float32)
        return out, E

    def _embed_batch(self, wavs: list[np.ndarray]) -> np.ndarray:
        import torch
        from resemblyzer import audio as raudio

        mels = [raudio.wav_to_mel_spectrogram(w) for w in wavs]
        L = min(len(m) for m in mels)
        batch = np.stack([m[:L] for m in mels]).astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(batch).to(self._enc.device)
            embeds = []
            for i in range(0, len(x), 256):
                embeds.append(self._enc(x[i:i + 256]).cpu().numpy())
        e = np.concatenate(embeds, axis=0)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
        return e


# ---- in-domain adaptation ----------------------------------------------------------------------
def _spherical_kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """k-means on unit vectors (cosine). Returns (labels, centroids). Deterministic k-means++ init."""
    rng = np.random.default_rng(seed)
    n = len(X)
    k = max(1, min(k, n))
    cents = [X[rng.integers(n)]]
    for _ in range(1, k):
        d = 1 - np.max(X @ np.stack(cents).T, axis=1)
        d = np.clip(d, 1e-9, None) ** 2
        cents.append(X[rng.choice(n, p=d / d.sum())])
    C = np.stack(cents)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        new = np.argmax(X @ C.T, axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                c = X[m].mean(axis=0)
                C[j] = c / (np.linalg.norm(c) + 1e-9)
    return labels, C


def adapt(timeline: list[dict], E: np.ndarray, enrol: Enrollment, k: int | None = None) -> tuple[list[dict], dict]:
    """In-domain speaker clustering, anchored by the clean enrolment.

    The film has many voices, so "reactor vs one film centroid" drifts onto whichever film character
    sounds most like him. Instead: spherical k-means over all voiced windows → the cluster(s) whose
    centroid is closest to the clean enrolment are REACTOR (only clusters within a small margin of
    the best), every other cluster is a film voice. Each window is scored by the contrast
    ``score = cos(e, nearest reactor centroid) - cos(e, nearest film centroid)``.
    Adds ``score`` to each timeline entry; returns (timeline, info)."""
    valid = ~np.isnan(E[:, 0])
    nv = int(valid.sum())
    if nv < 40:
        for w in timeline:
            w["score"] = None if w["sim"] is None else round(w["sim"] - 0.5, 3)
        return timeline, {"adapted": False, "reason": "too few voiced windows"}
    Ev = E[valid]
    k = k or int(np.clip(nv // 20, 12, 48))
    labels, C = _spherical_kmeans(Ev, k)
    csim = C @ enrol.embedding                       # each cluster's similarity to the clean voice
    order = np.argsort(-csim)
    top = float(csim[order[0]])
    reactor_clusters = [int(j) for j in order if csim[j] >= top - 0.03]
    film_clusters = [j for j in range(len(C)) if j not in reactor_clusters]
    info: dict = {"adapted": True, "k": int(len(C)), "voiced_windows": nv,
                  "cluster_sims": [round(float(v), 3) for v in csim[order][:8]],
                  "reactor_clusters": reactor_clusters,
                  "reactor_windows": int(np.isin(labels, reactor_clusters).sum())}
    if not film_clusters or top < 0.5:
        info.update(adapted=False, reason="no separable reactor cluster")
        for w in timeline:
            w["score"] = None if w["sim"] is None else round(w["sim"] - 0.5, 3)
        return timeline, info
    Cr, Cf = C[reactor_clusters], C[film_clusters]
    score = np.max(Ev @ Cr.T, axis=1) - np.max(Ev @ Cf.T, axis=1)
    info["score_thr"] = round(float(_otsu(score)), 3)
    info["second_best_gap"] = round(float(top - csim[order[len(reactor_clusters)]]), 3) if len(order) > len(reactor_clusters) else None
    j = 0
    for i, w in enumerate(timeline):
        if valid[i]:
            w["score"] = round(float(score[j]), 3)
            j += 1
        else:
            w["score"] = None
    return timeline, info


def fuse_motion(timeline: list[dict], motion: np.ndarray, weight: float = 0.5) -> list[dict]:
    """Audio-visual fusion: fused = z(score) + weight * z(face motion). Windows without motion data
    keep their audio score. Adds ``motion`` and ``fused`` to each timeline entry."""
    sc = np.array([np.nan if w.get("score") is None else w["score"] for w in timeline], dtype=np.float64)
    mo = np.asarray(motion, dtype=np.float64)
    ok = ~np.isnan(sc)
    if ok.sum() < 10:
        for w in timeline:
            w["fused"] = w.get("score")
        return timeline
    zs = (sc - np.nanmean(sc[ok])) / (np.nanstd(sc[ok]) + 1e-9)
    mo_ok = ok & ~np.isnan(mo)
    if mo_ok.sum() >= 10:
        # robust z for motion (heavy right tail): median / MAD, clipped
        med = np.nanmedian(mo[mo_ok])
        mad = np.nanmedian(np.abs(mo[mo_ok] - med)) * 1.4826 + 1e-6
        zm = np.clip((mo - med) / mad, -3, 3)
    else:
        zm = np.zeros_like(mo)
    fused = zs + weight * np.where(np.isnan(zm), 0.0, zm)
    for w, m, f, o in zip(timeline, mo, fused, ok):
        w["motion"] = None if np.isnan(m) else round(float(m), 2)
        w["fused"] = round(float(f), 3) if o else None
    return timeline


def _otsu(x: np.ndarray, bins: int = 64) -> float:
    hist, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    best_t, best_v = float(np.median(x)), -1.0
    for k in range(1, len(hist)):
        w0, w1 = hist[:k].sum(), hist[k:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (hist[:k] * centers[:k]).sum() / w0
        m1 = (hist[k:] * centers[k:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2 / (total ** 2)
        if v > best_v:
            best_v, best_t = v, float(edges[k])
    return best_t


# ---- calibration + labelling (backend-independent) --------------------------------------------
def calibrate(enrol: Enrollment, timeline: list[dict]) -> Enrollment:
    """Set the REACTOR/FILM threshold from the real track: Otsu split of voiced-window similarities,
    kept within [p10-0.15, p10] of the enrolment's own similarity distribution."""
    sims = np.array([w["sim"] for w in timeline if w["sim"] is not None], dtype=np.float64)
    p10 = float(np.percentile(enrol.sims, 10)) if enrol.sims.size else 0.8
    lo, hi = max(0.5, p10 - 0.15), min(0.9, p10)
    if sims.size < 50:
        enrol.threshold = float(np.clip(enrol.threshold, lo, hi))
        enrol.extra["calibration"] = "provisional (too few voiced windows)"
        return enrol
    hist, edges = np.histogram(sims, bins=60, range=(0.3, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    best_t, best_v = None, -1.0
    total = hist.sum()
    for k in range(1, len(hist)):
        w0, w1 = hist[:k].sum(), hist[k:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (hist[:k] * centers[:k]).sum() / w0
        m1 = (hist[k:] * centers[k:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2 / (total ** 2)
        if v > best_v:
            best_v, best_t = v, float(edges[k])
    thr = float(np.clip(best_t if best_t is not None else enrol.threshold, lo, hi))
    enrol.threshold = thr
    enrol.extra["calibration"] = f"otsu={best_t:.3f} clamped to [{lo:.3f},{hi:.3f}]" if best_t else "provisional"
    enrol.extra["voiced_windows"] = int(sims.size)
    enrol.extra["reactor_window_frac"] = round(float((sims >= thr).mean()), 3)
    return enrol


def label_segments(transcript_segments: list[dict], timeline: list[dict], threshold: float, margin: float,
                   key: str = "sim") -> list[dict]:
    ts = np.array([w["t"] for w in timeline])
    sims = np.array([np.nan if w.get(key) is None else w[key] for w in timeline], dtype=np.float64)
    out = []
    for s in transcript_segments:
        a, b = s["start"] - WIN_S / 2, s["end"] + WIN_S / 2
        idx = np.where((ts >= a) & (ts <= b))[0]
        vals = sims[idx]
        vals = vals[~np.isnan(vals)]
        rec = {"id": s["id"], "start": s["start"], "end": s["end"], "text": s.get("text", "")}
        if vals.size == 0:
            rec.update(speaker="UNKNOWN", reactor_frac=None, sim_max=None, sim_mean=None)
        else:
            rf = float((vals >= threshold + margin).mean())
            ff = float((vals < threshold - margin).mean())
            if rf >= 0.6:
                spk = "REACTOR"
            elif rf <= 0.15 and ff >= 0.5:
                spk = "FILM"
            elif rf <= 0.15:
                spk = "FILM" if vals.max() < threshold else "MIXED"
            else:
                spk = "MIXED"
            rec.update(speaker=spk, reactor_frac=round(rf, 3), sim_max=round(float(vals.max()), 3),
                       sim_mean=round(float(vals.mean()), 3))
        out.append(rec)
    return out


def reactor_spans(timeline: list[dict], threshold: float, min_dur: float = 1.0, gap: float = 1.0,
                  key: str = "sim") -> list[dict]:
    """Merge consecutive REACTOR windows into spans (independent of the transcript)."""
    spans: list[dict] = []
    cur = None
    for w in timeline:
        v = w.get(key)
        on = v is not None and v >= threshold
        t0, t1 = w["t"] - WIN_S / 2, w["t"] + WIN_S / 2
        if on:
            if cur and t0 - cur["t1"] <= gap:
                cur["t1"] = t1
                cur["n"] += 1
                cur["score_max"] = max(cur["score_max"], v)
            else:
                cur = {"t0": round(t0, 2), "t1": t1, "n": 1, "score_max": v}
                spans.append(cur)
    result = []
    for s in spans:
        s["t1"] = round(s["t1"], 2)
        if s["t1"] - s["t0"] >= min_dur:
            result.append(s)
    return result


# ---- top-level ------------------------------------------------------------------------------
def run_speakers(
    wav_path: Path,
    transcript: dict,
    out: Path,
    *,
    sample_paths: list[Path],
    t0: float | None = None,
    t1: float | None = None,
    device: str = "cpu",
    force: bool = False,
    face_motion: "FaceMotion | None" = None,
    progress: Callable[[float, str], None] | None = None,
) -> Path:
    if out.exists() and not force:
        return out
    tagger = ResemblyzerTagger(device=device)
    enrol = tagger.enroll(sample_paths)
    wav, sr = load_wav(wav_path, t0, t1)
    tl, E = tagger.timeline_with_embeddings(wav, sr, t0 or 0.0, enrol, progress=progress)
    np.save(out.with_name("embeddings.npy"), E.astype(np.float16))   # for re-thresholding without re-embedding
    np.save(out.with_name("enrol_embedding.npy"), enrol.embedding.astype(np.float32))
    enrol = calibrate(enrol, tl)                      # threshold on raw clean-enrolment similarity
    tl, ainfo = adapt(tl, E, enrol)                   # in-domain contrast score
    fusion = {"used": False}
    if ainfo.get("adapted"):
        key = "score"
        if face_motion is not None:
            centres = np.array([w["t"] for w in tl])
            if face_motion.covers(float(centres[0]), float(centres[-1])):
                tl = fuse_motion(tl, face_motion.per_window(centres))
                key = "fused"
                fusion = {"used": True, "weight": 0.5}
        vals = np.array([w[key] for w in tl if w.get(key) is not None])
        thr = float(_otsu(vals))
        margin = float(0.15 * vals.std()) if vals.size else 0.02
    else:
        thr, margin, key = enrol.threshold, enrol.margin, "sim"
    segs = label_segments(transcript["segments"], tl, thr, margin, key=key)
    counts = {k: sum(1 for s in segs if s["speaker"] == k) for k in LABELS}
    data = {"backend": tagger.name, "score_key": key, "threshold": round(thr, 3), "margin": round(margin, 3),
            "raw_threshold": round(enrol.threshold, 3),
            "window_s": WIN_S, "hop_s": HOP_S, "silence_db": SILENCE_DB,
            "enrollment": enrol.stats(), "adaptation": ainfo, "fusion": fusion,
            "range": [t0, t1] if (t0 is not None or t1 is not None) else None,
            "counts": counts, "reactor_spans": reactor_spans(tl, thr, key=key),
            "timeline": tl, "segments": segs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def self_check(sample_paths: list[Path], holdout: Path, device: str = "cpu") -> dict:
    """Enrol on ``sample_paths`` and score a held-out clip of the same speaker: what fraction of its
    voiced windows would be called REACTOR at the provisional threshold?"""
    from resemblyzer import preprocess_wav

    tagger = ResemblyzerTagger(device=device)
    enrol = tagger.enroll(sample_paths)
    wav = preprocess_wav(str(holdout))
    tl = tagger.timeline(wav.astype(np.float32), SR, 0.0, enrol)
    sims = np.array([w["sim"] for w in tl if w["sim"] is not None])
    return {"threshold": round(enrol.threshold, 3), "n_windows": int(sims.size),
            "holdout_sim_median": round(float(np.median(sims)), 3) if sims.size else None,
            "holdout_sim_p10": round(float(np.percentile(sims, 10)), 3) if sims.size else None,
            "frac_reactor": round(float((sims >= enrol.threshold).mean()), 3) if sims.size else None,
            "enrollment": enrol.stats()}


# ---- human review export ---------------------------------------------------------------------
def export_review(wav_path: Path, speakers_path: Path, out_dir: Path, *, per_class: int = 12,
                  clip_s: float = 2.4, within: list[tuple[float, float]] | None = None) -> dict[str, Path]:
    """Write short audio contact sheets so a human can verify the tagger by ear:
    ``review_reactor.wav`` (confident REACTOR windows), ``review_film.wav`` (confident FILM) and
    ``review_borderline.wav`` (near the threshold), each clip separated by a beep, plus a
    ``review_index.txt`` listing the source timestamps in playback order."""
    from .. import ffmpeg

    d = json.loads(Path(speakers_path).read_text(encoding="utf-8"))
    key, thr, margin = d.get("score_key", "sim"), d["threshold"], d["margin"]
    tl = [w for w in d["timeline"] if w.get(key) is not None]
    if within:
        tl = [w for w in tl if any(a <= w["t"] <= b for a, b in within)]
    rng = np.random.default_rng(0)

    def pick(pred, n):
        cand = [w for w in tl if pred(w[key])]
        rng.shuffle(cand)
        # spread out: avoid two picks within 3 s of each other
        chosen: list[dict] = []
        for w in cand:
            if all(abs(w["t"] - c["t"]) > 3.0 for c in chosen):
                chosen.append(w)
            if len(chosen) >= n:
                break
        return sorted(chosen, key=lambda w: w["t"])

    sets = {
        "reactor": pick(lambda v: v >= thr + margin, per_class),
        "borderline": pick(lambda v: thr - margin <= v < thr + margin, per_class),
        "film": pick(lambda v: v < thr - margin, per_class),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    outs: dict[str, Path] = {}
    index_lines = []
    for name, ws in sets.items():
        if not ws:
            continue
        parts, labels = [], []
        for i, w in enumerate(ws):
            a = max(0.0, w["t"] - clip_s / 2)
            parts.append(f"[0:a]atrim=start={a:.2f}:end={a + clip_s:.2f},asetpts=PTS-STARTPTS,afade=t=in:d=0.03,afade=t=out:st={clip_s-0.03:.2f}:d=0.03[c{i}]")
            parts.append(f"sine=frequency=1000:d=0.15,volume=0.15[b{i}]")
            labels.append(f"[c{i}][b{i}]")
        graph = ";".join(parts) + ";" + "".join(labels) + f"concat=n={2*len(ws)}:v=0:a=1[out]"
        dest = out_dir / f"review_{name}.wav"
        ffmpeg.run(["-i", str(wav_path), "-filter_complex", graph, "-map", "[out]", "-y", str(dest)])
        outs[name] = dest
        index_lines.append(f"# {name} ({len(ws)} clips of {clip_s}s, beep between)")
        for i, w in enumerate(ws, 1):
            m, s = divmod(int(w["t"]), 60)
            index_lines.append(f"{i:2d}. {m:3d}:{s:02d}  {key}={w[key]:+.3f}")
        index_lines.append("")
    idx = out_dir / "review_index.txt"
    idx.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    outs["index"] = idx
    picks = out_dir / "review_picks.json"
    picks.write_text(json.dumps({k: [{"n": i + 1, "t": w["t"], key: w[key]} for i, w in enumerate(ws)]
                                 for k, ws in sets.items()}, indent=1) + "\n", encoding="utf-8")
    outs["picks"] = picks
    return outs


# ---- evaluation against human labels --------------------------------------------------------
def evaluate(speakers_path: Path, labels_path: Path) -> dict:
    """AUC + accuracy of the stored scoring against ``labels.json``
    (``{"labels": [{"t": 1520.3, "speaker": "REACTOR|FILM|EITHER"}]}``; EITHER is ignored)."""
    d = json.loads(Path(speakers_path).read_text(encoding="utf-8"))
    labs = [l for l in json.loads(Path(labels_path).read_text(encoding="utf-8"))["labels"] if l["speaker"] in ("REACTOR", "FILM")]
    key, thr = d.get("score_key", "sim"), d["threshold"]
    tl = d["timeline"]
    ts = np.array([w["t"] for w in tl])
    res: dict = {"n_labels": len(labs), "score_key": key, "threshold": thr}
    if not labs:
        return res
    idx = [int(np.argmin(np.abs(ts - l["t"]))) for l in labs]
    y = np.array([l["speaker"] == "REACTOR" for l in labs])
    keep = np.array([abs(ts[i] - l["t"]) <= 1.0 and tl[i].get(key) is not None for i, l in zip(idx, labs)])
    if keep.sum() == 0:
        res["error"] = "no labelled windows inside this speakers.json range"
        return res
    for k in ("sim", "score", "fused"):
        vals = np.array([np.nan if tl[i].get(k) is None else tl[i][k] for i in idx], dtype=float)
        m = keep & ~np.isnan(vals)
        if m.sum() < 4 or y[m].all() or (~y[m]).all():
            continue
        pos, neg = vals[m & y], vals[m & ~y]
        auc = float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
        res[f"auc_{k}"] = round(auc, 3)
    vals = np.array([np.nan if tl[i].get(key) is None else tl[i][key] for i in idx], dtype=float)
    m = keep & ~np.isnan(vals)
    pred = vals[m] >= thr
    tp = int((pred & y[m]).sum()); fp = int((pred & ~y[m]).sum()); fn = int((~pred & y[m]).sum()); tn = int((~pred & ~y[m]).sum())
    res.update(accuracy=round((tp + tn) / max(1, m.sum()), 3),
               precision=round(tp / max(1, tp + fp), 3), recall=round(tp / max(1, tp + fn), 3),
               confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return res
