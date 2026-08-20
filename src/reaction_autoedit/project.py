"""A *project* is one reaction recording being processed: ``work/<name>/project.json`` plus cache dirs.

Layout::

    work/<name>/
      project.json      # source, probe, geometry, stage status
      analysis/         # Stage 2 artifacts (transcript.json, speakers.json, ...)
      renders/          # outputs + intermediates
      edl.json          # current cut list (Stage 3 output / hand-written)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from . import ffmpeg
from .config import ReactorConfig, TitleConfig
from .models import Geometry, ProbeInfo

DEFAULT_ROOT = Path("work")


class ProjectState(BaseModel):
    name: str
    source: str
    created: str
    probe: ProbeInfo | None = None
    geometry: Geometry | None = None
    reactor_config: str | None = None   # path to configs/reactors/*.json
    title_config: str | None = None
    film_bounds: list[float] | None = None   # manual [film_start, film_end] in source seconds
    stages: dict[str, Any] = Field(default_factory=dict)  # stage name → status/metadata


class Project:
    def __init__(self, root: Path, state: ProjectState):
        self.root = root
        self.state = state

    # ---- paths -------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.root / "project.json"

    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis"

    @property
    def renders_dir(self) -> Path:
        return self.root / "renders"

    @property
    def edl_path(self) -> Path:
        return self.root / "edl.json"

    @property
    def source(self) -> Path:
        return Path(self.state.source)

    # ---- lifecycle ---------------------------------------------------------
    @classmethod
    def create(
        cls,
        name: str,
        source: str | Path,
        *,
        root: Path = DEFAULT_ROOT,
        reactor_config: str | None = None,
        title_config: str | None = None,
        overwrite: bool = False,
    ) -> "Project":
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(src)
        pdir = root / name
        if pdir.exists() and (pdir / "project.json").exists() and not overwrite:
            raise FileExistsError(f"project '{name}' already exists at {pdir} (use --overwrite)")
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "analysis").mkdir(exist_ok=True)
        (pdir / "renders").mkdir(exist_ok=True)
        state = ProjectState(
            name=name,
            source=str(src.resolve()),
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            reactor_config=reactor_config,
            title_config=title_config,
        )
        proj = cls(pdir, state)
        proj.state.probe = ffmpeg.probe(src)
        proj.save()
        return proj

    @classmethod
    def load(cls, name_or_path: str | Path, root: Path = DEFAULT_ROOT) -> "Project":
        p = Path(name_or_path)
        pdir = p if (p / "project.json").exists() else root / str(name_or_path)
        f = pdir / "project.json"
        if not f.exists():
            raise FileNotFoundError(f"no project at {pdir} (run `rae init` first)")
        state = ProjectState.model_validate(json.loads(f.read_text(encoding="utf-8")))
        return cls(pdir, state)

    def save(self) -> None:
        self.path.write_text(self.state.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def mark(self, stage: str, **info: Any) -> None:
        info.setdefault("at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.state.stages[stage] = info
        self.save()

    # ---- config helpers ----------------------------------------------------
    def reactor(self) -> ReactorConfig:
        from .config import load_reactor

        return load_reactor(self.state.reactor_config)

    def title(self) -> TitleConfig:
        from .config import load_title

        return load_title(self.state.title_config)

    def require_geometry(self) -> Geometry:
        if self.state.geometry is None:
            raise RuntimeError("no geometry yet — run `rae detect-layout` (or provide a layout template)")
        return self.state.geometry
