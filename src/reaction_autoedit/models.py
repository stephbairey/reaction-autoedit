"""Shared data models used across stages (geometry, probe info)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Rect(BaseModel):
    """Axis-aligned rectangle in source-frame pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else 0.0

    def iou(self, other: "Rect") -> float:
        ix = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        iy = max(0, min(self.y2, other.y2) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def clamp(self, w: int, h: int) -> "Rect":
        x = max(0, min(self.x, w - 1))
        y = max(0, min(self.y, h - 1))
        return Rect(x=x, y=y, w=max(2, min(self.w, w - x)), h=max(2, min(self.h, h - y)))

    def even(self) -> "Rect":
        """Snap to even coordinates/sizes (required by yuv420p crops)."""
        return Rect(x=self.x - self.x % 2, y=self.y - self.y % 2, w=self.w - self.w % 2, h=self.h - self.h % 2)

    def ffmpeg_crop(self) -> str:
        r = self.even()
        return f"crop={r.w}:{r.h}:{r.x}:{r.y}"


class FaceRegion(Rect):
    shape: Literal["rect", "circle"] = "rect"


class FrameInfo(BaseModel):
    w: int
    h: int
    fps: float


class Geometry(BaseModel):
    """Where things live inside the composite frame.

    ``movie`` is the box the compositor allotted to the film (may include letterbox bars);
    ``movie_inner`` is the active picture inside it (what we actually crop for output);
    ``face`` is the reactor's camera region.
    """

    frame: FrameInfo
    movie: Rect
    movie_inner: Rect
    face: FaceRegion
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source: Literal["auto", "template", "manual", "fixture"] = "auto"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _inner_inside_movie(self) -> "Geometry":
        m, i = self.movie, self.movie_inner
        if not (m.x <= i.x and m.y <= i.y and i.x2 <= m.x2 and i.y2 <= m.y2):
            self.notes.append("movie_inner not fully inside movie; clamped")
            self.movie_inner = Rect(
                x=max(m.x, i.x), y=max(m.y, i.y),
                w=min(m.x2, i.x2) - max(m.x, i.x), h=min(m.y2, i.y2) - max(m.y, i.y),
            )
        return self


class AudioInfo(BaseModel):
    codec: str
    sample_rate: int
    channels: int


class ProbeInfo(BaseModel):
    path: str
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    bit_rate: int | None = None
    audio: AudioInfo | None = None
