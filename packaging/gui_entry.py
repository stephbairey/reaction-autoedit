"""PyInstaller entry point: point FFMPEG_BIN at the bundled ffmpeg, work in %LOCALAPPDATA%, launch."""
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    # windowed app: no console → sys.stdout/stderr are None and uvicorn's logger calls .isatty().
    # Route everything to a log file next to the user data (also our only diagnostics channel).
    _home = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ReactionAutoEdit"
    _home.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None or sys.stderr is None:
        _log = open(_home / "app.log", "a", buffering=1, encoding="utf-8", errors="replace")
        sys.stdout = sys.stdout or _log
        sys.stderr = sys.stderr or _log
    bundle = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    ff = bundle / "ffmpeg"
    if (ff / "ffmpeg.exe").exists():
        os.environ.setdefault("FFMPEG_BIN", str(ff / "ffmpeg.exe"))
        os.environ.setdefault("FFPROBE_BIN", str(ff / "ffprobe.exe"))
    torch_lib = bundle / "torch" / "lib"        # CUDA/cuDNN DLLs (GPU edition) — ctranslate2 needs them on PATH
    if torch_lib.exists():
        os.add_dll_directory(str(torch_lib))
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    home = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ReactionAutoEdit"
    home.mkdir(parents=True, exist_ok=True)
    for sub in ("configs", "templates"):
        dst = home / sub
        if not dst.exists():
            import shutil
            shutil.copytree(bundle / sub, dst)
    env = home / ".env"
    if not env.exists():
        env.write_text("# API keys (optional)\nANTHROPIC_API_KEY=\nTVDB_API_KEY=\nYOUTUBE_API_KEY=\n", encoding="utf-8")
    os.chdir(home)

from reaction_autoedit.cli import _load_dotenv  # noqa: E402
_load_dotenv()
from reaction_autoedit.gui.server import run  # noqa: E402
run(window=True)
