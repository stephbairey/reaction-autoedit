"""Stage 0, evidence source #2 — the survival survey: how have OTHER reactors fared with this title?

Uses the YouTube Data API v3 (free key in ``.env`` as ``YOUTUBE_API_KEY``). We can never see another
channel's monetization or claim status; what we CAN see is what *survived*: long-form reactions to
this exact title that are still up, and for how long. A two-year-old 90-minute reaction still
standing is the strongest public signal of a tolerant rights holder; finding only shorts and
recent uploads suggests longer cuts get removed.

Output ``work/<name>/analysis/preflight.json``::

    {"title": ..., "n_longform": 14, "median_age_months": 16.2, "oldest_age_months": 50.1,
     "median_views": 48211, "channels": ["...", ...], "videos": [...], "verdict": "tolerant"}
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

API = "https://www.googleapis.com/youtube/v3"


class SurveyError(RuntimeError):
    pass


def _call(path: str, **params) -> dict:
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise SurveyError("YOUTUBE_API_KEY not set (put it in .env)")
    qs = urllib.parse.urlencode({**params, "key": key})
    req = Request(f"{API}/{path}?{qs}", headers={"User-Agent": "reaction-autoedit/0.1"})
    try:
        with urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        raise SurveyError(f"YouTube API call failed: {e}") from e


def _iso_dur_s(d: str) -> float:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d or "")
    if not m:
        return 0.0
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + s


def _age_months(published: str) -> float:
    dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - dt).days / 30.44


def survey(title: str, year: int | None, out: Path, *, min_minutes: float = 20.0,
           force: bool = False) -> dict:
    out = Path(out)
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    q = f"{title} reaction"
    found: dict[str, dict] = {}
    for dur_filter in ("long",):                      # >20 min per API definition
        res = _call("search", part="snippet", q=q, type="video", videoDuration=dur_filter,
                    maxResults=50, safeSearch="none")
        for item in res.get("items", []):
            vid = item["id"].get("videoId")
            if vid:
                found[vid] = item["snippet"]
    ids = list(found)
    videos = []
    for i in range(0, len(ids), 50):
        det = _call("videos", part="contentDetails,statistics,snippet", id=",".join(ids[i:i + 50]))
        for v in det.get("items", []):
            sn, cd, st = v["snippet"], v["contentDetails"], v.get("statistics", {})
            dur = _iso_dur_s(cd.get("duration", ""))
            if dur < min_minutes * 60:
                continue
            t = sn.get("title", "")
            # keep only videos that actually name the film (search is fuzzy)
            tokens = [w for w in re.split(r"\W+", title.lower()) if len(w) > 2]
            if tokens and sum(w in t.lower() for w in tokens) < max(1, len(tokens) - 1):
                continue
            videos.append({
                "id": v["id"], "title": t[:120], "channel": sn.get("channelTitle", ""),
                "published": sn.get("publishedAt", ""), "age_months": round(_age_months(sn["publishedAt"]), 1),
                "minutes": round(dur / 60, 1), "views": int(st.get("viewCount", 0)),
            })
    videos.sort(key=lambda v: -v["age_months"])
    n = len(videos)
    med = lambda xs: (sorted(xs)[len(xs) // 2] if xs else 0)  # noqa: E731
    ages = [v["age_months"] for v in videos]
    result = {
        "query": q, "title": title, "year": year, "min_minutes": min_minutes,
        "n_longform": n,
        "median_age_months": round(med(ages), 1) if ages else None,
        "oldest_age_months": round(max(ages), 1) if ages else None,
        "n_older_6mo": sum(a >= 6 for a in ages),
        "median_views": med([v["views"] for v in videos]) if videos else None,
        "channels": sorted({v["channel"] for v in videos})[:20],
        "videos": videos[:25],
        "verdict": verdict(n, ages),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def verdict(n: int, ages_months: list[float]) -> str:
    """Public-survival verdict. Absence of evidence is weak evidence here (search is lossy), so the
    negative verdicts are phrased as caution, not doom."""
    old = sum(a >= 6 for a in ages_months)
    if n >= 5 and old >= 3:
        return "tolerant"          # plenty of long-form reactions surviving for months+
    if n >= 2 and old >= 1:
        return "mixed"             # some survive; proceed, watch outcomes
    if n >= 1:
        return "sparse"            # little long-form survives — conservative cut advised
    return "none-found"            # nothing surfaced; could be graveyard or just search miss
