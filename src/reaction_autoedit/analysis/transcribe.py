"""Transcription (faster-whisper) over the mixed track — captures both the reactor and film dialogue.

Output ``transcript.json``::

    {"model": "small", "compute_type": "int8", "device": "cpu", "language": "en",
     "range": [t0, t1] | null,
     "segments": [{"id": 0, "start": 12.3, "end": 15.1, "text": "...",
                   "words": [{"w": "..", "s": 12.3, "e": 12.6, "p": 0.93}]}]}

Times are absolute source seconds even when a ``range`` slice was transcribed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from ..compute import ComputeProfile, detect
from .audio import SR, load_wav


def transcribe(
    wav_path: str | Path,
    out: str | Path,
    *,
    t0: float | None = None,
    t1: float | None = None,
    model: str | None = None,
    compute_type: str | None = None,
    device: str | None = None,
    language: str | None = None,
    force: bool = False,
    profile: ComputeProfile | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> Path:
    out = Path(out)
    if out.exists() and not force:
        return out
    from faster_whisper import WhisperModel  # heavy import, keep local

    profile = profile or detect()
    model = model or profile.whisper_model
    compute_type = compute_type or profile.whisper_compute_type
    device = device or profile.device
    audio, sr = load_wav(wav_path, t0, t1)
    assert sr == SR, f"expected {SR} Hz wav, got {sr}"
    offset = t0 or 0.0

    wm = WhisperModel(model, device=device, compute_type=compute_type,
                      cpu_threads=max(1, profile.cpu_count - 1) if device == "cpu" else 0)
    segs_iter, info = wm.transcribe(
        audio, language=language, word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=3, condition_on_previous_text=False,
    )
    total = len(audio) / sr
    segments = []
    started = time.time()
    for i, s in enumerate(segs_iter):
        words = [{"w": w.word.strip(), "s": round(offset + w.start, 3), "e": round(offset + w.end, 3), "p": round(w.probability, 3)}
                 for w in (s.words or [])]
        segments.append({"id": i, "start": round(offset + s.start, 3), "end": round(offset + s.end, 3),
                         "text": s.text.strip(), "words": words})
        if progress and i % 10 == 0:
            done = s.end / total if total else 0.0
            el = time.time() - started
            progress(done, f"{s.end/60:.1f}/{total/60:.1f} min transcribed, {el/60:.1f} min elapsed")
    data = {"model": model, "compute_type": compute_type, "device": device, "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "range": [t0, t1] if (t0 is not None or t1 is not None) else None, "segments": segments}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_transcript(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
