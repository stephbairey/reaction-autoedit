"""Audio extraction + small helpers shared by the analysers.

The whole mixed track is extracted once to ``analysis/audio16k.wav`` (mono, 16 kHz, PCM) — the
common input format for whisper, resemblyzer and librosa. Ranges are sliced from that file.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .. import ffmpeg

SR = 16000


def extract_audio(src: str | Path, out: str | Path, *, force: bool = False, sr: int = SR) -> Path:
    out = Path(out)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.wav")
    ffmpeg.run(["-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", "-y", str(tmp)], timeout=3600)
    tmp.replace(out)
    return out


def load_wav(path: str | Path, t0: float | None = None, t1: float | None = None) -> tuple[np.ndarray, int]:
    """Read (a slice of) a mono 16-bit PCM wav as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        a = int(max(0, (t0 or 0.0)) * sr)
        b = n if t1 is None else min(n, int(t1 * sr))
        w.setpos(min(a, n))
        raw = w.readframes(max(0, b - a))
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def wav_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return 20 * np.log10(max(r, 1e-9))
