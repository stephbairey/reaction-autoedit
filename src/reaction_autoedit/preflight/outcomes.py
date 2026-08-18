"""Stage 6 — per-studio claim-outcome table, appended after every upload (manual entry in v1).

File: ``configs/outcomes.json``::

    {"entries": [{"title": "...", "studio": "...", "outcome": "none|sharing|redirect|block",
                  "project": "work name", "at": "ISO date"}]}
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUTCOMES = ("none", "sharing", "redirect", "block")


class OutcomeStore:
    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict] = []
        if path.exists():
            self.entries = json.loads(path.read_text(encoding="utf-8")).get("entries", [])

    @classmethod
    def default(cls) -> "OutcomeStore":
        return cls(Path("configs") / "outcomes.json")

    def record(self, *, title: str, studio: str | None, outcome: str, project: str | None = None) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        self.entries.append({"title": title, "studio": studio, "outcome": outcome, "project": project,
                             "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"entries": self.entries}, indent=2) + "\n", encoding="utf-8")

    def by_studio(self) -> dict[str, Counter]:
        table: dict[str, Counter] = defaultdict(Counter)
        for e in self.entries:
            table[e.get("studio") or "unknown"][e["outcome"]] += 1
        return table

    def flag_for(self, studio: str | None) -> str:
        """green / yellow / red / unknown from this studio's history."""
        c = self.by_studio().get(studio or "unknown")
        if not c:
            return "unknown"
        n = sum(c.values())
        if c["block"] > 0:
            return "red"
        if c["redirect"] / n > 0.5:
            return "yellow"
        return "green"
