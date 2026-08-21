"""FastAPI app + launcher: `rae gui` serves the frontend and opens a native window (pywebview)
or the browser. Everything binds to 127.0.0.1 only."""

from __future__ import annotations

import socket
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router

STATIC = Path(__file__).parent / "static"


def create_app(work_root: Path = Path("work")) -> FastAPI:
    app = FastAPI(title="reaction-autoedit", docs_url="/api/docs")
    app.include_router(router)
    work_root.mkdir(exist_ok=True)
    # /media/<project>/... → work/<project>/... (StaticFiles handles HTTP Range for the video player)
    app.mount("/media", StaticFiles(directory=str(work_root)), name="media")
    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="app")
    return app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(port: int | None = None, window: bool = True) -> None:
    import uvicorn

    port = port or _free_port()
    app = create_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    url = f"http://127.0.0.1:{port}"

    if not window:
        print(f"reaction-autoedit GUI: {url}")
        server.run()
        return

    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        import webview  # pywebview

        webview.create_window("Reaction AutoEdit", url, width=1380, height=900)
        webview.start()
        server.should_exit = True
    except Exception:  # pywebview missing/broken (e.g. WSL) → browser fallback
        import webbrowser

        print(f"reaction-autoedit GUI: {url}  (Ctrl+C to stop)")
        webbrowser.open(url)
        t.join()
