import shutil
import time

import pytest
from fastapi.testclient import TestClient

from circlapse.organize import Photo
from circlapse.studio.launcher import Launcher
from circlapse.studio.project import load_project
from test_studio import T0, FakeDetector, write_jpeg


def wait(client):
    for _ in range(200):
        status = client.get("/api/launcher/import").json()
        if status["state"] != "running":
            return status
        time.sleep(0.02)
    raise AssertionError("import did not finish")


@pytest.fixture
def photos(tmp_path):
    src = tmp_path / "trip"
    src.mkdir()
    write_jpeg(src / "a.jpg", T0)
    write_jpeg(src / "b.jpg", T0.replace(hour=10))
    write_jpeg(src / "dark.jpg", T0, value=10)
    return src


def make(tmp_path, **kwargs):
    launcher = Launcher(
        **{
            "detector": lambda conf: FakeDetector(),
            "library": tmp_path / "library",
            "state_file": tmp_path / "state.json",
            "export": None,
            **kwargs,
        }
    )
    return launcher, TestClient(launcher)


def test_start_page_has_no_project(tmp_path):
    _, client = make(tmp_path)
    info = client.get("/api/launcher").json()
    assert info == {
        "project": None,
        "recent": [],
        "apple_photos": False,
        "library": str(tmp_path / "library"),
        "min_confidence": 0.81,
    }
    assert client.get("/api/project").status_code == 404


def test_import_folder_opens_its_project(tmp_path, photos):
    _, client = make(tmp_path)
    started = client.post("/api/launcher/import", json={"source": "paths", "paths": [str(photos)]})
    assert started.json()["state"] == "running"
    status = wait(client)
    assert status["state"] == "done", status
    assert status["project"] == str((photos / "circlapse").resolve())
    assert status["total"] == 3
    project = client.get("/api/project").json()
    assert project["name"] == "trip"
    assert len(project["order"]) == 2
    assert client.get("/api/launcher").json()["recent"][0]["photos"] == 2


def test_import_of_a_project_folder_reopens_it(tmp_path, photos):
    launcher, client = make(tmp_path)
    client.post("/api/launcher/import", json={"source": "paths", "paths": [str(photos)]})
    wait(client)
    client.post("/api/launcher/close")
    assert client.get("/api/launcher").json()["project"] is None
    client.delete("/api/launcher/import")
    client.post("/api/launcher/import", json={"source": "paths", "paths": [str(photos)]})
    assert wait(client)["state"] == "done"
    assert launcher.project == (photos / "circlapse").resolve()


def test_dropped_photos_get_a_library_folder(tmp_path, photos):
    _, client = make(tmp_path)
    paths = [str(photos / "a.jpg"), str(photos / "dark.jpg")]
    client.post("/api/launcher/import", json={"source": "paths", "paths": paths})
    status = wait(client)
    assert status["state"] == "done", status
    project = load_project(tmp_path / "library" / "trip")
    assert project.order == ["20250719-090000_a.jpg"]
    assert not (photos / "circlapse").exists()


def test_import_errors(tmp_path, photos):
    _, client = make(tmp_path)
    missing = client.post("/api/launcher/import", json={"source": "paths", "paths": ["/nope"]})
    assert missing.status_code == 422
    empty = tmp_path / "empty"
    empty.mkdir()
    res = client.post("/api/launcher/import", json={"source": "paths", "paths": [str(empty)]})
    assert res.status_code == 422
    res = client.post("/api/launcher/import", json={"source": "apple_photos"})
    assert "photos extra" in res.json()["detail"]
    dark = tmp_path / "dark"
    dark.mkdir()
    write_jpeg(dark / "x.jpg", T0, value=10)
    client.post("/api/launcher/import", json={"source": "paths", "paths": [str(dark)]})
    status = wait(client)
    assert status["state"] == "error"
    assert "no circle" in status["error"]
    assert not (dark / "circlapse").exists()


def test_apple_photos_export_goes_to_the_library(tmp_path):
    calls = []

    def export(dest, first, last, log):
        calls.append((first, last))
        dest.mkdir(parents=True)
        write_jpeg(dest / "IMG_1.jpg")
        log("[1/1] exported IMG_1.jpg")
        return {"IMG_1.jpg": Photo("IMG_1.jpg", T0, None, None)}

    _, client = make(tmp_path, export=export)
    assert client.get("/api/launcher").json()["apple_photos"]
    body = {"source": "apple_photos", "from_date": "2025-07-19", "to_date": "2025-07-20"}
    client.post("/api/launcher/import", json=body)
    status = wait(client)
    assert status["state"] == "done", status
    project = load_project(tmp_path / "library" / "2025-07-19_2025-07-20" / "circlapse")
    # The time comes from Apple Photos, the file has no EXIF date
    assert project.order == ["20250719-090000_IMG_1.jpg"]
    assert calls == [(T0.date(), T0.date().replace(day=20))]


def test_open_recent_and_forget_missing(tmp_path, photos):
    launcher, client = make(tmp_path)
    client.post("/api/launcher/import", json={"source": "paths", "paths": [str(photos)]})
    wait(client)
    client.post("/api/launcher/close")
    recent = client.get("/api/launcher").json()["recent"]
    assert [r["path"] for r in recent] == [str((photos / "circlapse").resolve())]
    assert client.post("/api/launcher/open", json={"path": str(photos)}).json()["project"]
    assert client.post("/api/launcher/open", json={"path": str(tmp_path)}).status_code == 404
    (photos / "circlapse" / "project.json").unlink()
    assert launcher.info().recent == []


def test_import_uses_the_requested_confidence(tmp_path, photos):
    used = []

    def detector(conf):
        used.append(conf)
        return FakeDetector()

    _, client = make(tmp_path, detector=detector)
    body = {"source": "paths", "paths": [str(photos)], "min_confidence": 0.9}
    client.post("/api/launcher/import", json=body)
    assert wait(client)["state"] == "done"
    client.post("/api/launcher/close")
    client.delete("/api/launcher/import")
    other = tmp_path / "other"
    shutil.copytree(photos, other, ignore=shutil.ignore_patterns("circlapse"))
    client.post("/api/launcher/import", json={"source": "paths", "paths": [str(other)]})
    assert wait(client)["state"] == "done"
    assert used == [0.9, 0.81]
    bad = client.post("/api/launcher/import", json={**body, "min_confidence": 1.5})
    assert bad.status_code == 422
