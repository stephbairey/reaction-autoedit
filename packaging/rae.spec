# PyInstaller spec — Reaction AutoEdit (Windows onedir bundle).
# Build (on Windows, from repo root):  pyinstaller packaging/rae.spec
# CI drops ffmpeg.exe/ffprobe.exe into packaging/ffmpeg/ first; they ship inside the bundle and
# gui-launch sets FFMPEG_BIN/FFPROBE_BIN to them.
import os
from pathlib import Path

block_cipher = None
ROOT = Path(os.getcwd())
STATIC = ROOT / "src" / "reaction_autoedit" / "gui" / "static"
FF = ROOT / "packaging" / "ffmpeg"

datas = [(str(STATIC), "reaction_autoedit/gui/static"),
         (str(ROOT / "templates"), "templates"),
         (str(ROOT / "configs"), "configs")]
binaries = []
for exe in ("ffmpeg.exe", "ffprobe.exe"):
    if (FF / exe).exists():
        binaries.append((str(FF / exe), "ffmpeg"))

hidden = ["uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
          "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on",
          "webview.platforms.edgechromium"]

a = Analysis([str(ROOT / "packaging" / "gui_entry.py")],
             pathex=[str(ROOT / "src")], datas=datas, binaries=binaries,
             hiddenimports=hidden, excludes=["tkinter", "matplotlib.tests"],
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ReactionAutoEdit",
          console=False, icon=None)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="ReactionAutoEdit")
