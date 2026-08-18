"""Stage 1 — detect where the movie and the facecam live inside the composite frame.

Approach (all OpenCV, CPU-cheap):

1. Sample N frames spread across the recording (downscaled greyscale).
2. Per-pixel temporal standard deviation → *motion map*. The film region changes constantly and
   dominates; the reactor moves less; the compositor background (static wallpaper) is ~0.
3. Threshold + morphology + connected components. Largest high-motion blob = movie active picture
   (``movie_inner``). Grow it outward over near-black rows/cols to recover letterbox bars → ``movie``.
4. Zero the movie box (dilated) in the motion mask; the largest remaining blob is the reactor region.
   Refine with a Haar face detector on the median frame; the face box is expanded into a
   comfortable ``face`` region. Circular PiPs are recognised by a persistent round edge.
5. Merge with an optional per-reactor template: template wins if auto confidence is low, otherwise
   auto is kept and disagreement is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..models import FaceRegion, FrameInfo, Geometry, Rect

ANALYSIS_WIDTH = 480


@dataclass
class _Sampled:
    frames: np.ndarray        # (n, h, w) uint8 grey, downscaled
    color_median: np.ndarray  # (h, w, 3) BGR median frame, downscaled
    scale: float              # source px per analysis px
    src_w: int
    src_h: int
    fps: float


def sample_frames(video: str | Path, n: int = 120, start_frac: float = 0.03, end_frac: float = 0.97) -> _Sampled:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = src_w / ANALYSIS_WIDTH
    aw, ah = ANALYSIS_WIDTH, max(2, int(round(src_h / scale)))
    if total <= 0:
        total = int(fps * 60)
    idxs = np.linspace(int(total * start_frac), int(total * end_frac), num=n).astype(int)
    greys, colors = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA)
        colors.append(small)
        greys.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(greys) < 8:
        raise RuntimeError(f"could only read {len(greys)} frames from {video}")
    frames = np.stack(greys).astype(np.float32)
    med = np.median(np.stack(colors), axis=0).astype(np.uint8)
    return _Sampled(frames=frames, color_median=med, scale=scale, src_w=src_w, src_h=src_h, fps=fps)


def motion_map(frames: np.ndarray) -> np.ndarray:
    """Per-pixel temporal std, normalised to 0..1 (robust to outliers)."""
    std = frames.std(axis=0)
    hi = np.percentile(std, 99.5) or 1.0
    return np.clip(std / hi, 0, 1)


def _largest_component(mask: np.ndarray, exclude: Rect | None = None) -> tuple[Rect | None, float]:
    """Return bounding rect (analysis px) and fill ratio of the largest connected component."""
    m = mask.copy()
    if exclude is not None:
        m[max(0, exclude.y):exclude.y2, max(0, exclude.x):exclude.x2] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    x, y, w, h, area = stats[k]
    fill = area / float(w * h) if w * h else 0.0
    return Rect(x=int(x), y=int(y), w=int(w), h=int(h)), float(fill)


def _grow_over_dark(rect: Rect, med_grey: np.ndarray, motion: np.ndarray, dark_thr: int = 40,
                    motion_thr: float = 0.06, max_frac: float = 0.22, tol: int = 2) -> Rect:
    """Extend a rect over adjacent rows/cols that are dark and static (letterbox/pillarbox bars).

    Growth per side is capped at ``max_frac`` of the rect's size (bars are never wider than that),
    and up to ``tol`` mixed (anti-aliased) rows/cols at the picture edge are skipped over.
    """
    H, W = med_grey.shape
    x1, y1, x2, y2 = rect.x, rect.y, rect.x2, rect.y2

    def dark_static(vals_g: np.ndarray, vals_m: np.ndarray) -> bool:
        return vals_g.mean() < dark_thr and vals_m.mean() < motion_thr

    def grow(lo: int, hi: int, limit_lo: int, limit_hi: int, get) -> tuple[int, int]:
        cap = int(max_frac * (hi - lo))
        # upward/leftward
        n, skipped = 0, 0
        while lo > limit_lo and n < cap:
            g, m = get(lo - 1)
            if dark_static(g, m):
                lo -= 1; n += 1; skipped = 0
            elif skipped < tol and lo - 2 >= limit_lo and dark_static(*get(lo - 2)):
                lo -= 1; n += 1; skipped += 1
            else:
                break
        n, skipped = 0, 0
        while hi < limit_hi and n < cap:
            g, m = get(hi)
            if dark_static(g, m):
                hi += 1; n += 1; skipped = 0
            elif skipped < tol and hi + 1 < limit_hi and dark_static(*get(hi + 1)):
                hi += 1; n += 1; skipped += 1
            else:
                break
        return lo, hi

    y1, y2 = grow(y1, y2, 0, H, lambda y: (med_grey[y, x1:x2], motion[y, x1:x2]))
    x1, x2 = grow(x1, x2, 0, W, lambda x: (med_grey[y1:y2, x], motion[y1:y2, x]))
    return Rect(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _trim_dark(rect: Rect, med_grey: np.ndarray, dark_thr: int = 40) -> Rect:
    """Shrink a rect until its border rows/cols are not dark (tightens active picture)."""
    x1, y1, x2, y2 = rect.x, rect.y, rect.x2, rect.y2
    while y2 - y1 > 8 and med_grey[y1, x1:x2].mean() < dark_thr:
        y1 += 1
    while y2 - y1 > 8 and med_grey[y2 - 1, x1:x2].mean() < dark_thr:
        y2 -= 1
    while x2 - x1 > 8 and med_grey[y1:y2, x1].mean() < dark_thr:
        x1 += 1
    while x2 - x1 > 8 and med_grey[y1:y2, x2 - 1].mean() < dark_thr:
        x2 -= 1
    return Rect(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _haar_face(color_bgr: np.ndarray, region: Rect) -> Rect | None:
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        return None
    if cascade.empty():
        return None
    roi = color_bgr[region.y:region.y2, region.x:region.x2]
    if roi.size == 0:
        return None
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    grey = cv2.equalizeHist(grey)
    faces = cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4, minSize=(int(region.w * 0.12), int(region.w * 0.12)))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return Rect(x=region.x + int(x), y=region.y + int(y), w=int(w), h=int(h))


def _circle_pip(med_grey: np.ndarray, motion: np.ndarray, blob: Rect) -> Rect | None:
    """If the reactor blob is a circular PiP, return its tight bounding square; else None.

    A circular PiP has a crisp round boundary between moving content and static background.
    We test candidate circles from Hough on the median frame's edges and require that the ring just
    inside the circle is high-motion while the ring just outside is low-motion.
    """
    H, W = med_grey.shape
    if not (0.55 < blob.aspect < 1.8):
        return None
    pad = int(0.15 * max(blob.w, blob.h))
    roi = Rect(x=blob.x - pad, y=blob.y - pad, w=blob.w + 2 * pad, h=blob.h + 2 * pad).clamp(W, H)
    g = med_grey[roi.y:roi.y2, roi.x:roi.x2]
    r_est = min(blob.w, blob.h) / 2
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(g, (5, 5), 0), cv2.HOUGH_GRADIENT, dp=1.2, minDist=r_est,
        param1=100, param2=25, minRadius=int(r_est * 0.7), maxRadius=int(r_est * 1.25),
    )
    if circles is None:
        return None
    ys, xs = np.mgrid[0:H, 0:W]
    best = None
    for cx, cy, r in circles[0]:
        cx, cy = cx + roi.x, cy + roi.y
        d = np.hypot(xs - cx, ys - cy)
        inner = motion[(d > r * 0.80) & (d < r * 0.95)]
        outer = motion[(d > r * 1.05) & (d < r * 1.25)]
        if inner.size < 20 or outer.size < 20:
            continue
        contrast = float(inner.mean() - outer.mean())
        if contrast > 0.08 and inner.mean() > 2.5 * (outer.mean() + 1e-3):
            if best is None or contrast > best[0]:
                best = (contrast, cx, cy, r)
    if best is None:
        return None
    _, cx, cy, r = best
    return Rect(x=int(round(cx - r)), y=int(round(cy - r)), w=int(round(2 * r)), h=int(round(2 * r))).clamp(W, H)


def _split_on_static_bands(mask: np.ndarray, rect: Rect, min_gap: int = 3, thr: float = 0.08) -> Rect:
    """Compositor layouts are axis-aligned: if a blob's bbox spans a static column/row band, split it
    there and keep the largest piece. Repeats until stable."""
    cur = rect
    for _ in range(6):
        sub = mask[cur.y:cur.y2, cur.x:cur.x2]
        if sub.size == 0:
            return cur
        best = cur
        for axis, size, off in ((0, cur.w, cur.x), (1, cur.h, cur.y)):
            frac = sub.mean(axis=axis)  # axis=0 → per column, axis=1 → per row
            low = frac < thr
            # find runs of low columns/rows strictly inside
            pieces, start = [], 0
            i = 0
            while i < size:
                if low[i]:
                    j = i
                    while j < size and low[j]:
                        j += 1
                    if j - i >= min_gap and i > 0:
                        pieces.append((start, i))
                        start = j
                    elif j - i >= min_gap and i == 0:
                        start = j
                    i = j
                else:
                    i += 1
            pieces.append((start, size))
            pieces = [(a, b) for a, b in pieces if b - a > 2]
            if len(pieces) > 1 or (pieces and (pieces[0] != (0, size))):
                a, b = max(pieces, key=lambda p: (p[1] - p[0]) * float(sub[:, p[0]:p[1]].sum() if axis == 0 else sub[p[0]:p[1], :].sum()))
                cand = Rect(x=off + a, y=cur.y, w=b - a, h=cur.h) if axis == 0 else Rect(x=cur.x, y=off + a, w=cur.w, h=b - a)
                if cand.area < best.area:
                    best = cand
        if best == cur:
            return cur
        cur = best
    return cur


def _persistent_circle(frames: np.ndarray, min_persist: float = 0.45) -> Rect | None:
    """Find a circle whose edge is present in (almost) every sampled frame — a PiP border.
    Film content rarely keeps a fixed circular edge; a compositor overlay does."""
    n, H, W = frames.shape
    step = max(1, n // 40)
    acc = np.zeros((H, W), np.float32)
    used = 0
    for k in range(0, n, step):
        acc += cv2.Canny(frames[k].astype(np.uint8), 60, 140) > 0
        used += 1
    persist = acc / max(1, used)
    p8 = (np.clip(persist, 0, 1) * 255).astype(np.uint8)
    rmin, rmax = int(W * 0.04), int(W * 0.25)
    circles = cv2.HoughCircles(cv2.GaussianBlur(p8, (3, 3), 0), cv2.HOUGH_GRADIENT, dp=1.0, minDist=rmin,
                               param1=120, param2=18, minRadius=rmin, maxRadius=rmax)
    if circles is None:
        return None
    ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    best = None
    for cx, cy, r in circles[0]:
        xs = np.clip((cx + r * np.cos(ang)).round().astype(int), 0, W - 1)
        ys = np.clip((cy + r * np.sin(ang)).round().astype(int), 0, H - 1)
        # tolerate ±1px by taking max over a 3px neighbourhood
        vals = np.max(np.stack([persist[np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)]
                                for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), axis=0)
        score = float(vals.mean())
        if score >= min_persist and (best is None or score > best[0]):
            best = (score, cx, cy, r)
    if best is None:
        return None
    _, cx, cy, r = best
    return Rect(x=int(round(cx - r)), y=int(round(cy - r)), w=int(round(2 * r)), h=int(round(2 * r))).clamp(W, H)


def _scale_rect(r: Rect, s: float, W: int, H: int) -> Rect:
    return Rect(x=int(round(r.x * s)), y=int(round(r.y * s)), w=int(round(r.w * s)), h=int(round(r.h * s))).clamp(W, H).even()


def detect_layout(
    video: str | Path,
    *,
    n_frames: int = 120,
    template: Geometry | None = None,
    debug_image: str | Path | None = None,
) -> Geometry:
    smp = sample_frames(video, n=n_frames)
    mot = motion_map(smp.frames)
    med_grey = cv2.cvtColor(smp.color_median, cv2.COLOR_BGR2GRAY)
    H, W = mot.shape
    notes: list[str] = []

    # --- movie ---------------------------------------------------------------
    mot8 = (mot * 255).astype(np.uint8)
    thr, _ = cv2.threshold(cv2.GaussianBlur(mot8, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strong = (mot8 > max(thr, 25)).astype(np.uint8)
    strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    movie_a, movie_fill = _largest_component(strong)
    if movie_a is None:
        raise RuntimeError("could not find a high-motion region (movie)")
    movie_a = _split_on_static_bands(strong, movie_a)
    movie_fill = float(strong[movie_a.y:movie_a.y2, movie_a.x:movie_a.x2].mean()) if movie_a.area else 0.0
    movie_inner_a = _trim_dark(movie_a, med_grey)
    movie_box_a = _grow_over_dark(movie_inner_a, med_grey, mot)
    movie_motion = float(mot[movie_inner_a.y:movie_inner_a.y2, movie_inner_a.x:movie_inner_a.x2].mean())

    # --- face ----------------------------------------------------------------
    weak = (mot8 > max(8, int(thr * 0.35))).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    pad = 6
    excl = Rect(x=movie_box_a.x - pad, y=movie_box_a.y - pad, w=movie_box_a.w + 2 * pad, h=movie_box_a.h + 2 * pad).clamp(W, H)
    face_blob, _ = _largest_component(weak, exclude=excl)
    if face_blob is not None:
        face_blob = _split_on_static_bands(weak, face_blob)
    face_found = False
    circle: Rect | None = None
    if face_blob is None or face_blob.area < 0.01 * W * H:
        # maybe the facecam is a PiP *inside* the movie box → look for a persistent circular border
        circle = _persistent_circle(smp.frames)
        if circle is not None:
            face_blob = circle
            notes.append("facecam PiP found inside movie box via persistent circular edge")
        else:
            notes.append("no distinct reactor motion region found; guessing largest non-movie area")
            left, right = movie_box_a.x, W - movie_box_a.x2
            if left >= right:
                face_blob = Rect(x=0, y=0, w=max(2, left), h=H)
            else:
                face_blob = Rect(x=movie_box_a.x2, y=0, w=max(2, right), h=H)
    if circle is None:
        circle = _circle_pip(med_grey, mot, face_blob)
        if circle is not None:
            face_blob = circle
            notes.append("circular facecam detected")
    shape = "circle" if circle is not None else "rect"

    face_a = face_blob
    haar = _haar_face(smp.color_median, face_blob)
    if haar is not None:
        face_found = True
        # expand the detected face into a framing region: ~2.4x wide, head room above, torso below
        cx, cy = haar.x + haar.w / 2, haar.y + haar.h / 2
        fw, fh = haar.w * 2.6, haar.h * 3.2
        fx, fy = cx - fw / 2, cy - fh * 0.42
        cand = Rect(x=int(fx), y=int(fy), w=int(fw), h=int(fh)).clamp(W, H)
        # keep within the motion blob's extents when the blob is a real region (rect layouts)
        if shape == "rect":
            x1, y1 = max(cand.x, face_blob.x), max(cand.y, face_blob.y)
            x2, y2 = min(cand.x2, face_blob.x2), min(cand.y2, face_blob.y2)
            if x2 - x1 > 20 and y2 - y1 > 20:
                cand = Rect(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
        face_a = cand if shape == "rect" else face_blob
    else:
        notes.append("Haar face detector found no face; using motion blob for reactor region")

    # never overlap the movie box (unless the facecam is a PiP inside it)
    ix = max(0, min(face_a.x2, movie_box_a.x2) - max(face_a.x, movie_box_a.x))
    iy = max(0, min(face_a.y2, movie_box_a.y2) - max(face_a.y, movie_box_a.y))
    inside_movie = (ix * iy) / max(1, face_a.area) > 0.5
    if inside_movie:
        notes.append("facecam lies inside the movie box (overlay PiP layout)")
    elif face_a.iou(movie_box_a) > 0.05:
        notes.append("reactor region overlapped movie box; trimmed")
        if face_a.x < movie_box_a.x:
            face_a = Rect(x=face_a.x, y=face_a.y, w=max(20, movie_box_a.x - face_a.x - 2), h=face_a.h)
        elif face_a.x2 > movie_box_a.x2:
            nx = movie_box_a.x2 + 2
            face_a = Rect(x=nx, y=face_a.y, w=max(20, face_a.x2 - nx), h=face_a.h)

    # --- confidence -----------------------------------------------------------
    conf = 0.0
    conf += min(0.45, movie_motion * 1.2)                        # movie region clearly moving
    conf += 0.15 if 1.2 < movie_inner_a.aspect < 2.7 else 0.0    # plausible film aspect
    conf += 0.25 if face_found else 0.1                          # face located
    conf += 0.15 if movie_fill > 0.7 else 0.05                   # solid rectangular blob
    conf = float(min(1.0, conf))

    s = smp.scale
    geom = Geometry(
        frame=FrameInfo(w=smp.src_w, h=smp.src_h, fps=smp.fps),
        movie=_scale_rect(movie_box_a, s, smp.src_w, smp.src_h),
        movie_inner=_scale_rect(movie_inner_a, s, smp.src_w, smp.src_h),
        face=FaceRegion(**_scale_rect(face_a, s, smp.src_w, smp.src_h).model_dump(), shape=shape),
        confidence=conf,
        source="auto",
        notes=notes,
    )

    # --- template merge ------------------------------------------------------
    if template is not None:
        agree = geom.movie_inner.iou(template.movie_inner) > 0.8 and geom.face.iou(template.face) > 0.6
        if conf < 0.5 or not agree:
            t = template.model_copy(deep=True)
            t.source = "template"
            t.notes = notes + [
                f"auto detection {'low confidence' if conf < 0.5 else 'disagreed with template'} "
                f"(conf={conf:.2f}, movie IoU={geom.movie_inner.iou(template.movie_inner):.2f}, "
                f"face IoU={geom.face.iou(template.face):.2f}); template used"
            ]
            t.confidence = max(conf, 0.5)
            geom = t
        else:
            geom.notes.append("auto detection agrees with template")

    if debug_image:
        write_debug_image(smp.color_median, geom, s, debug_image)
    return geom


def write_debug_image(color_small: np.ndarray, geom: Geometry, scale: float, out: str | Path) -> None:
    img = color_small.copy()

    def draw(r: Rect, color, label: str):
        x, y, w, h = (int(round(v / scale)) for v in (r.x, r.y, r.w, r.h))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    draw(geom.movie, (0, 200, 255), "movie")
    draw(geom.movie_inner, (0, 255, 0), "movie_inner")
    draw(geom.face, (255, 80, 80), f"face/{geom.face.shape}")
    cv2.putText(img, f"conf={geom.confidence:.2f} src={geom.source}", (6, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
