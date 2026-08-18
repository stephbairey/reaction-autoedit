"""Generate placeholder branding assets (endcard, lower third) with OpenCV so the pipeline runs
out of the box. Replace them with designed PNGs of the same names when available."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _text(img: np.ndarray, text: str, y: int, scale: float, color=(255, 255, 255), thick: int = 2) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def make_endcard(path: str | Path, *, patreon_url: str = "patreon.com/…", display_name: str = "the reactor",
                 w: int = 1920, h: int = 1080) -> Path:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (24, 22, 20)
    cv2.rectangle(img, (60, 60), (w - 60, h - 60), (70, 65, 60), 2)
    _text(img, "THE FULL REACTION", 380, 2.2, (240, 240, 240), 4)
    _text(img, "is on Patreon", 470, 1.6, (200, 200, 200), 3)
    _text(img, patreon_url, 600, 1.4, (255, 140, 60), 3)
    _text(img, f"Thanks for watching with {display_name}", 760, 1.0, (160, 160, 160), 2)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)
    return p


def make_lower_third(path: str | Path, *, text: str = "FULL REACTION ON PATREON", sub: str = "",
                     w: int = 1920, h: int = 1080) -> Path:
    """Transparent PNG (full frame size) with a bar along the bottom."""
    img = np.zeros((h, w, 4), np.uint8)
    bar_h = 110
    y0 = h - bar_h - 40
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, y0 + bar_h), (20, 20, 20, 210), -1)
    cv2.rectangle(overlay, (0, y0), (14, y0 + bar_h), (60, 140, 255, 255), -1)
    img[:] = overlay
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, text, (48, y0 + 62 if not sub else y0 + 50), font, 1.5, (255, 255, 255, 255), 3, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (48, y0 + 92), font, 0.9, (200, 200, 200, 255), 2, cv2.LINE_AA)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)
    return p
