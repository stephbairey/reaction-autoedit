"""Speaker attribution: tag transcript segments as REACTOR vs FILM. [M2 — the risk item]

Design: a ``SpeakerTagger`` protocol so backends are swappable behind one interface.

    class SpeakerTagger(Protocol):
        def enroll(self, wav_path: Path) -> None: ...            # 30–60 s clean reactor sample
        def tag(self, wav_path: Path, segments: list[Seg]) -> list[TaggedSeg]: ...

Backends:
    ResemblyzerTagger  — first choice: pip-only, CPU-friendly speaker embeddings; cosine similarity
                         to the enrolled voice per transcript segment (+ per 1.5 s window inside long
                         segments) with a calibrated threshold → REACTOR / FILM / UNKNOWN.
    PyannoteTagger     — optional upgrade (needs HF token / gated model): full diarisation, then map
                         the diarised speaker closest to the enrolment to REACTOR.

Validation plan: hand-label ~10 minutes of the real sample; report precision/recall per class
before anything is built on top of this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict


class Seg(TypedDict):
    start: float
    end: float
    text: str


class TaggedSeg(Seg):
    speaker: str      # "REACTOR" | "FILM" | "UNKNOWN"
    confidence: float


class SpeakerTagger(Protocol):
    name: str

    def enroll(self, wav_path: Path) -> None: ...

    def tag(self, wav_path: Path, segments: list[Seg]) -> list[TaggedSeg]: ...


def get_tagger(backend: str = "resemblyzer") -> SpeakerTagger:  # pragma: no cover - M2
    raise NotImplementedError("Stage 2 speaker attribution lands in Milestone 2")
