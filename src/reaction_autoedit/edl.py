"""The EDL (edit decision list): the stable, human-editable contract between selection and assembly.

Example::

    {
      "version": 1,
      "source": "samples/x.mp4",
      "target": {"w": 1920, "h": 1080, "fps": 30},
      "segments": [
        {"id": "s001", "in": 123.4, "out": 129.9, "layout": "movie-large",
         "kind": "story", "transition": "cut", "chapter": "Act 1", "note": ""}
      ],
      "overlays": [{"type": "lower_third", "at": 1800, "dur": 6}],
      "endcard": {"template": "templates/endcard.png", "dur": 8}
    }

Times are **source** seconds for segments (``in``/``out``) and **output-timeline** seconds for
overlays (``at``). Segments must be chronological (``in`` non-decreasing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .config import RenderTarget

Layout = Literal["movie-large", "reactor-large", "full"]
Kind = Literal["story", "reaction", "cta", "intro", "outro"]
Transition = Literal["cut", "xfade"]


class Segment(BaseModel):
    id: str
    in_: float = Field(alias="in", ge=0)
    out: float = Field(gt=0)
    layout: Layout = "movie-large"
    kind: Kind = "story"
    transition: Transition = "cut"     # transition INTO this segment
    score: float = 0.0
    chapter: str | None = None         # starts a YouTube chapter with this title
    tags: list[str] = Field(default_factory=list)
    note: str = ""

    model_config = {"populate_by_name": True}

    @property
    def dur(self) -> float:
        return self.out - self.in_

    @model_validator(mode="after")
    def _order(self) -> "Segment":
        if self.out <= self.in_:
            raise ValueError(f"segment {self.id}: out ({self.out}) must be > in ({self.in_})")
        return self


class Overlay(BaseModel):
    type: Literal["lower_third"] = "lower_third"
    at: float = Field(ge=0)            # output-timeline seconds
    dur: float = Field(6.0, gt=0)
    template: str | None = None        # defaults to reactor branding


class Card(BaseModel):
    """A full-frame still card (e.g. movie title card) inserted into the output timeline,
    before the segment named by ``before_id`` (or at the very start if None). Fades in/out."""

    before_id: str | None = None
    template: str
    dur: float = Field(3.5, gt=0)


class Endcard(BaseModel):
    template: str | None = None        # defaults to reactor branding
    dur: float = Field(8.0, gt=0)


class EDL(BaseModel):
    version: int = 1
    source: str
    target: RenderTarget = Field(default_factory=RenderTarget)
    segments: list[Segment] = Field(default_factory=list)
    overlays: list[Overlay] = Field(default_factory=list)
    cards: list[Card] = Field(default_factory=list)
    endcard: Endcard | None = Field(default_factory=Endcard)
    meta: dict = Field(default_factory=dict)  # free-form: title, generator, params

    # ---- derived ----------------------------------------------------------
    @property
    def duration(self) -> float:
        return sum(s.dur for s in self.segments)

    def offsets(self) -> list[float]:
        """Output-timeline start time of each segment (accounts for inserted cards)."""
        card_before: dict[str | None, float] = {}
        for c in self.cards:
            card_before[c.before_id] = card_before.get(c.before_id, 0.0) + c.dur
        out, t = [], card_before.get(None, 0.0)
        for s in self.segments:
            t += card_before.get(s.id, 0.0)
            out.append(t)
            t += s.dur
        return out

    # ---- validation -------------------------------------------------------
    def validate_rules(self, clip_cap_s: float | None = None, source_duration: float | None = None) -> list[str]:
        """Return human-readable warnings (non-fatal). Structural errors raise at parse time."""
        warns: list[str] = []
        ids = [s.id for s in self.segments]
        if len(set(ids)) != len(ids):
            warns.append("duplicate segment ids")
        prev = None
        for s in self.segments:
            if prev is not None and s.in_ < prev.in_:
                warns.append(f"{s.id}: not chronological (in {s.in_:.2f} < previous in {prev.in_:.2f})")
            if source_duration is not None and s.out > source_duration + 0.05:
                warns.append(f"{s.id}: out {s.out:.2f} exceeds source duration {source_duration:.2f}")
            if clip_cap_s is not None and s.layout == "movie-large" and s.dur > clip_cap_s + 1e-6:
                warns.append(f"{s.id}: movie-large segment {s.dur:.1f}s exceeds clip cap {clip_cap_s:.1f}s")
            if s.dur < 0.3:
                warns.append(f"{s.id}: very short segment ({s.dur:.2f}s)")
            prev = s
        total = self.duration
        for o in self.overlays:
            if o.at > total:
                warns.append(f"overlay at {o.at:.1f}s is beyond output duration {total:.1f}s")
        return warns

    # ---- io ---------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "EDL":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.model_dump(by_alias=True, mode="json"), indent=2) + "\n", encoding="utf-8"
        )


def starter_edl(source: str, source_duration: float, target: RenderTarget | None = None) -> EDL:
    """A small hand-editable EDL touching both layouts, for kicking the tyres on a new project."""
    d = source_duration
    pts = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
    step = d * (pts[1] - pts[0])
    a = min(6.0, round(step * 0.5, 2))   # movie-large slice
    b = min(4.0, round(step * 0.3, 2))   # reactor-large cutaway
    segs: list[Segment] = []
    for i, f in enumerate(pts, 1):
        t = round(d * f, 2)
        segs.append(Segment(id=f"s{2*i-1:03d}", **{"in": t}, out=round(t + a, 2), layout="movie-large", kind="story",
                            chapter=f"Beat {i}" if i in (1, 3, 5) else None))
        segs.append(Segment(id=f"s{2*i:03d}", **{"in": round(t + a, 2)}, out=round(t + a + b, 2), layout="reactor-large", kind="reaction"))
    lt_at = min(30.0, sum(s.dur for s in segs) * 0.5)
    return EDL(source=source, target=target or RenderTarget(), segments=segs,
               overlays=[Overlay(at=round(lt_at, 2), dur=min(6.0, b))], meta={"generator": "starter_edl"})
