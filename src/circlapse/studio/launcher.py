"""The studio's start page: make a project from a folder, dropped photos or Apple Photos, or
reopen a recent one, then serve that project's studio (see :mod:`circlapse.studio.app`).

Folder imports keep the project next to the photos (``<folder>/circlapse``, as `circlapse
make` does). Apple Photos exports and dropped photos get a folder in the library.
"""

from __future__ import annotations

import importlib.util
import json
import re
import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.types import Receive, Scope, Send

from circlapse.annotate.app import mount_frontend
from circlapse.detect.base import Detector
from circlapse.labels import IMAGE_SUFFIXES
from circlapse.organize import Photo, safe_name
from circlapse.raw import is_raw
from circlapse.studio.app import create_studio_app
from circlapse.studio.project import create_project, is_project, load_project

PROJECT_DIR = "circlapse"
RECENT_LIMIT = 12
DEFAULT_LIBRARY = Path.home() / "Pictures" / "circlapse"
DEFAULT_STATE = Path.home() / ".config" / "circlapse" / "studio.json"

ExportRange = Callable[[Path, date, date, Callable[[str], None]], dict[str, Photo]]


class ImportRequest(BaseModel):
    source: Literal["paths", "apple_photos"]
    # source "paths": one folder, or photos (and folders whose photos count)
    paths: list[str] = []
    # source "apple_photos": first and last day, inclusive
    from_date: date | None = None
    to_date: date | None = None


class ImportStatus(BaseModel):
    state: Literal["idle", "running", "done", "error"] = "idle"
    step: str = ""
    done: int = 0
    total: int = 0
    log: list[str] = []
    error: str | None = None
    # The new project, opened when the import is done
    project: str | None = None


class RecentProject(BaseModel):
    name: str
    path: str
    photos: int


class LauncherInfo(BaseModel):
    # The open project, None on the start page
    project: str | None
    recent: list[RecentProject]
    apple_photos: bool
    library: str


class OpenRequest(BaseModel):
    path: str


def project_in(folder: Path) -> Path | None:
    """The project that is ``folder`` or lives in it."""
    for candidate in (folder, folder / PROJECT_DIR):
        if is_project(candidate):
            return candidate
    return None


def _is_photo(p: Path) -> bool:
    return (
        p.is_file()
        and not p.name.startswith(".")
        and (p.suffix.lower() in IMAGE_SUFFIXES or is_raw(p))
    )


def _unique_dir(parent: Path, name: str) -> Path:
    path, i = parent / name, 2
    while path.exists():
        path = parent / f"{name}_{i}"
        i += 1
    return path


def _default_export() -> ExportRange | None:
    if importlib.util.find_spec("osxphotos") is None:
        return None
    from circlapse.apple_photos import export_range

    return lambda dest, first, last, log: export_range(dest, first, last, log=log)


class Launcher:
    """ASGI app: the launcher routes under ``/api/launcher``, everything else goes to the open
    project's studio, or to the plain frontend on the start page."""

    def __init__(
        self,
        detector: Callable[[], Detector],
        library: Path = DEFAULT_LIBRARY,
        state_file: Path = DEFAULT_STATE,
        export: ExportRange | None | Literal["default"] = "default",
        project: Path | None = None,
    ) -> None:
        self.detector = detector
        self.library = library.expanduser()
        self.state_file = state_file
        self.export = _default_export() if export == "default" else export
        self.status = ImportStatus()
        self.lock = threading.Lock()
        self.project: Path | None = None
        self.studio: Any = None
        self.start = FastAPI()
        mount_frontend(self.start)
        self.api = self._routes()
        if project is not None:
            self.open(project)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope["type"] == "lifespan" or path.startswith(("/api/launcher", "/docs", "/openapi")):
            await self.api(scope, receive, send)
        else:
            await (self.studio or self.start)(scope, receive, send)

    # ------------------------------------------------------------ projects

    def recent(self) -> list[Path]:
        try:
            paths = json.loads(self.state_file.read_text()).get("recent", [])
        except (OSError, ValueError, AttributeError):
            return []
        return [Path(p) for p in paths if isinstance(p, str) and is_project(Path(p))]

    def _remember(self, project: Path) -> None:
        paths = [project, *(p for p in self.recent() if p != project)][:RECENT_LIMIT]
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({"recent": [str(p) for p in paths]}, indent=2))
        except OSError:
            pass  # only the recent list is lost

    def open(self, project: Path) -> None:
        project = project.expanduser().resolve()
        studio = create_studio_app(project)
        self.studio, self.project = studio, project
        self._remember(project)

    def close(self) -> None:
        self.studio, self.project = None, None

    def info(self) -> LauncherInfo:
        recent = []
        for p in self.recent():
            try:
                project = load_project(p)
            except (OSError, ValueError):
                continue
            recent.append(RecentProject(name=project.name, path=str(p), photos=len(project.order)))
        return LauncherInfo(
            project=str(self.project) if self.project else None,
            recent=recent,
            apple_photos=self.export is not None,
            library=str(self.library),
        )

    # ------------------------------------------------------------ import

    def _log(self, msg: str) -> None:
        msg = msg.strip()
        self.status.log = [*self.status.log[-199:], msg]
        if m := re.match(r"\[(\d+)/(\d+)\]", msg):
            self.status.done, self.status.total = int(m[1]), int(m[2])

    def _step(self, step: str) -> None:
        self.status.step, self.status.done, self.status.total = step, 0, 0

    def _make(
        self,
        source: Path,
        project: Path,
        files: list[Path] | None = None,
        fallback: dict[str, Photo] | None = None,
    ) -> None:
        self._step("Loading the detector")
        detector = self.detector()
        self._step("Finding circles")
        created = create_project(source, project, detector, fallback, log=self._log, files=files)
        if not created.order:
            raise ValueError("no circle found in any of the photos")

    def _run(self, plan: Callable[[], Path]) -> None:
        try:
            project = plan()
            self.open(project)
            self.status.project = str(project)
            self.status.state = "done"
        except Exception as e:  # reported to the UI
            self.status.error = str(e) or type(e).__name__
            self.status.state = "error"

    def _plan(self, request: ImportRequest) -> Callable[[], Path]:
        """Check the request and return the import, which runs in the background."""
        if request.source == "apple_photos":
            export, first, last = self.export, request.from_date, request.to_date
            if export is None:
                raise HTTPException(422, "install the photos extra: uv sync --extra photos")
            if first is None or last is None or first > last:
                raise HTTPException(422, "choose a first and a last day")
            name = f"{first}_{last}" if first != last else str(first)

            def from_photos() -> Path:
                data = _unique_dir(self.library, name)
                self._step("Exporting from Apple Photos")
                fallback = export(data, first, last, self._log)
                if not fallback:
                    raise ValueError(f"Apple Photos has no photos from {first} to {last}")
                project = data / PROJECT_DIR
                self._make(data, project, fallback=fallback)
                return project

            return from_photos

        paths = [Path(p).expanduser() for p in request.paths if p.strip()]
        if not paths:
            raise HTTPException(422, "choose a folder or photos")
        if missing := [str(p) for p in paths if not p.exists()]:
            raise HTTPException(422, f"not found: {', '.join(missing)}")
        if len(paths) == 1 and paths[0].is_dir():
            folder = paths[0].resolve()
            existing = project_in(folder)
            if existing is not None:
                return lambda: existing
            if not any(_is_photo(p) for p in folder.iterdir()):
                raise HTTPException(422, f"no photos directly in {folder}")

            def from_folder() -> Path:
                project = folder / PROJECT_DIR
                self._make(folder, project)
                return project

            return from_folder

        files = []
        for p in paths:
            files += sorted(c for c in p.iterdir() if _is_photo(c)) if p.is_dir() else [p]
        files = [f.resolve() for f in files if _is_photo(f)]
        if not files:
            raise HTTPException(422, "none of these is a photo")
        parents = {f.parent for f in files}
        source = parents.pop() if len(parents) == 1 else Path("/")
        name = source.name if source.name else "photos"

        def from_files() -> Path:
            project = _unique_dir(self.library, safe_name(name))
            self._make(source, project, files=files)
            return project

        return from_files

    # ------------------------------------------------------------ routes

    def _routes(self) -> FastAPI:
        api = FastAPI(title="circlapse launcher")

        @api.get("/api/launcher")
        def get_launcher() -> LauncherInfo:
            return self.info()

        @api.post("/api/launcher/open")
        def open_project(request: OpenRequest) -> LauncherInfo:
            project = project_in(Path(request.path).expanduser())
            if project is None:
                raise HTTPException(404, f"no project in {request.path}")
            self.open(project)
            return self.info()

        @api.post("/api/launcher/close")
        def close_project() -> LauncherInfo:
            self.close()
            return self.info()

        @api.post("/api/launcher/import")
        def start_import(request: ImportRequest) -> ImportStatus:
            with self.lock:
                if self.status.state == "running":
                    raise HTTPException(409, "an import is already running")
                plan = self._plan(request)
                self.status = ImportStatus(state="running", step="Starting")
            threading.Thread(target=self._run, args=(plan,), daemon=True).start()
            return self.status

        @api.get("/api/launcher/import")
        def import_status() -> ImportStatus:
            return self.status

        @api.delete("/api/launcher/import")
        def dismiss_import() -> ImportStatus:
            with self.lock:
                if self.status.state != "running":
                    self.status = ImportStatus()
            return self.status

        return api


def _no_detector() -> Detector:
    raise RuntimeError("no detector")


def openapi_schema(project_dir: Path) -> dict[str, Any]:
    """The studio's and the launcher's API in one schema (for the frontend types)."""
    schema: dict[str, Any] = create_studio_app(project_dir).openapi()
    launcher = Launcher(detector=_no_detector, export=None).api.openapi()
    schema["paths"].update(launcher["paths"])
    schema["components"]["schemas"].update(launcher["components"]["schemas"])
    return schema
