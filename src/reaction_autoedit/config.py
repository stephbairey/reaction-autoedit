"""Per-reactor and per-title configuration (JSON files under ``configs/``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .models import Geometry


class BorderStyle(BaseModel):
    """Gradient border drawn around insets (both the small PiP and the large reactor frame)."""

    px: int = 4
    color_from: str = "#9762FF"     # bottom-left
    color_to: str = "#FF01F8"       # top-right


class PipStyle(BaseModel):
    """Small overlay window used for the facecam in ``movie-large`` (and movie in ``reactor-large`` if enabled)."""

    corner: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "bottom-left"  # reactor 'faces' the film
    width_frac: float = Field(0.24, gt=0.05, lt=0.6)   # fraction of output width
    margin_px: int = 24
    border: BorderStyle = Field(default_factory=BorderStyle)
    circle: bool | None = None                          # None → follow detected face shape


class ReactorLargeStyle(BaseModel):
    face_height_frac: float = Field(0.92, gt=0.3, le=1.0)  # face crop scaled to this fraction of output height
    background: Literal["blur", "black"] = "blur"
    blur_strength: int = Field(24, ge=0, le=60)
    darken: float = Field(0.25, ge=0.0, le=0.9)
    movie_pip: bool = False                                # also show the movie small in a corner
    sharpen: float = Field(0.4, ge=0.0, le=1.5)            # unsharp amount to fight upscale softness
    vignette: bool = True
    border: BorderStyle = Field(default_factory=BorderStyle)


class LowerThirdSchedule(BaseModel):
    every_min: float = 20.0        # periodic display interval on the output timeline
    min_gap_min: float = 10.0      # never show two within this distance (incl. CTA-triggered ones)


class Branding(BaseModel):
    endcard: str | None = "templates/endcard.png"
    lower_third: str | None = "templates/lower_third.png"
    endcard_duration: float = 8.0
    lower_third_duration: float = 6.0
    lower_third_schedule: LowerThirdSchedule = Field(default_factory=LowerThirdSchedule)
    title_card: str | None = None        # composed per-title card (movie logo over base); shown after the intro
    title_card_base: str | None = None   # the channel's base card the logo gets composed onto
    opening_bumper: str | None = None    # short branded video prepended to every upload
    ending_bumper: str | None = None     # short branded video appended after the endcard


class ReactorConfig(BaseModel):
    name: str = "reactor"
    display_name: str = "the reactor"
    patreon_url: str = "https://www.patreon.com/"
    voice_sample: str | None = None            # 30–60 s clean sample for speaker enrollment (Stage 2)
    layout_template: Geometry | None = None    # fallback / override for Stage 1
    pip: PipStyle = Field(default_factory=PipStyle)
    reactor_large: ReactorLargeStyle = Field(default_factory=ReactorLargeStyle)
    branding: Branding = Field(default_factory=Branding)


class TitleConfig(BaseModel):
    title: str = "Untitled"
    year: int | None = None
    studio: str | None = None
    runtime_target_min: float = Field(55.0, ge=10, le=120)
    clip_cap_s: float = Field(7.0, ge=2, le=30)
    withhold_climax: bool = True
    layout_min_s: float = Field(2.0, ge=0.5)   # hysteresis: min duration per layout
    aspect_override: float | None = None      # movie aspect inside frame, e.g. 2.39
    risk_flag: Literal["green", "yellow", "red", "unknown"] = "unknown"
    trim_intro: bool = False                  # False (default) = intro uncut: everything before the film
    trim_outro: bool = False                  # False (default) = outro uncut: everything after the film
    title_card: str | None = None             # card image shown between intro and film (overrides reactor branding)
    clearlogo_url: str | None = None          # e.g. TVDB clearlogo PNG; `rae make-card` fetches + composes


class RenderTarget(BaseModel):
    w: int = 1920
    h: int = 1080
    fps: float = 30.0

    @classmethod
    def preview(cls) -> "RenderTarget":
        return cls(w=854, h=480, fps=30.0)


def _load(path: str | Path, model):
    p = Path(path)
    return model.model_validate(json.loads(p.read_text(encoding="utf-8")))


def load_reactor(path: str | Path | None) -> ReactorConfig:
    return _load(path, ReactorConfig) if path else ReactorConfig()


def load_title(path: str | Path | None) -> TitleConfig:
    return _load(path, TitleConfig) if path else TitleConfig()


def dump_json(model: BaseModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
