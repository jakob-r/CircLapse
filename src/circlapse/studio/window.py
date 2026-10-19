"""The studio in a native window (pywebview, the ``app`` extra) instead of a browser tab.

The page gets real file system paths, which a browser never hands out: a folder picker
(``window.pywebview.api.pick_folder()``) and the paths of dropped files and folders, sent to
the page as a ``circlapse:drop`` event with the paths as ``detail``.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import uvicorn


def available() -> bool:
    import importlib.util

    return importlib.util.find_spec("webview") is not None


class NativeApi:
    """Exposed to the page as ``window.pywebview.api``; underscore names stay private."""

    def __init__(self) -> None:
        self._window: Any = None

    def pick_folder(self) -> str | None:
        import webview

        picked = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        return str(picked[0]) if picked else None


def serve_window(app: Any, host: str, port: int) -> None:
    """Serve ``app`` in a background thread and show it in a window until that is closed."""
    import webview
    from webview.dom import DOMEventHandler

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise SystemExit(f"could not start the server on {host}:{port}")
        time.sleep(0.05)

    api = NativeApi()
    url = f"http://{host}:{port}"
    window: Any = webview.create_window(
        "circlapse", url, js_api=api, width=1440, height=920, min_size=(800, 560)
    )
    api._window = window

    def on_drop(event: dict[str, Any]) -> None:
        files = event.get("dataTransfer", {}).get("files", [])
        paths = [f["pywebviewFullPath"] for f in files if f.get("pywebviewFullPath")]
        if paths:
            detail = json.dumps(paths)
            window.evaluate_js(
                f"window.dispatchEvent(new CustomEvent('circlapse:drop', {{detail: {detail}}}))"
            )

    def on_loaded() -> None:
        # Not preventing the default: the page's own drop handlers decide
        window.dom.document.events.drop += DOMEventHandler(on_drop, False, False)

    window.events.loaded += on_loaded
    webview.start()
    server.should_exit = True
    thread.join(timeout=5)
