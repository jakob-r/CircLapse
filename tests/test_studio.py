import shutil
import time
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from circlapse.labels import ScoredEllipse, load_labels, save_labels
from circlapse.studio.app import create_studio_app
from circlapse.studio.project import create_project, is_project, load_project

T0 = datetime(2025, 7, 19, 9, 0)


def write_jpeg(path, time=None, value=200):
    rng = np.random.default_rng(value)
    pixels = np.clip(rng.normal(value, 20, (120, 160, 3)), 0, 255).astype(np.uint8)
    exif = Image.Exif()
    if time is not None:
        exif.get_ifd(0x8769)[36867] = time.strftime("%Y:%m:%d %H:%M:%S")
    Image.fromarray(pixels).save(path, exif=exif)


class FakeDetector:
    name = "fake"

    def detect(self, image):
        if image.mean() < 128:
            return []
        return [ScoredEllipse(cx=80, cy=60, a=30, b=28, angle_deg=10, score=0.9)]


@pytest.fixture
def project(tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    write_jpeg(src / "late.jpg", T0 + timedelta(days=1))
    write_jpeg(src / "early.jpg", T0)
    write_jpeg(src / "undated.jpg")
    write_jpeg(src / "dark.jpg", T0, value=10)
    create_project(src, src / "circlapse", FakeDetector(), log=lambda _: None)
    return src, src / "circlapse"


def test_create_project_copies_candidates_in_date_order(project):
    src, proj = project
    assert sorted(p.name for p in src.iterdir() if p.is_file()) == [
        "dark.jpg",
        "early.jpg",
        "late.jpg",
        "undated.jpg",
    ]
    p = load_project(proj)
    assert p.order == ["20250719-090000_early.jpg", "20250720-090000_late.jpg", "undated.jpg"]
    assert p.name == "photos"
    assert not (proj / "dark.jpg").exists()
    labels = load_labels(proj / "labels.json")
    assert labels["undated.jpg"].prominent_ellipse.score == 0.9


def test_project_api_reorder_and_exclude(project):
    _, proj = project
    c = TestClient(create_studio_app(proj))
    info = c.get("/api/project").json()
    assert info["frames"] == 3
    assert all(ph["usable"] and ph["preview"] for ph in info["photos"])
    order = list(reversed(info["order"]))
    r = c.put(
        "/api/project",
        json={"order": order, "excluded": [order[0]], "settings": info["settings"]},
    )
    assert r.status_code == 200
    assert r.json()["frames"] == 2
    assert load_project(proj).order == order
    assert [i["name"] for i in c.get("/api/images").json()] == order
    bad = {"order": order[:2], "excluded": [], "settings": info["settings"]}
    assert c.put("/api/project", json=bad).status_code == 422


def test_label_skip_becomes_an_exclusion(project):
    _, proj = project
    labels = load_labels(proj / "labels.json")
    labels["undated.jpg"] = labels["undated.jpg"].model_copy(update={"status": "skip"})
    save_labels(proj / "labels.json", labels)
    info = TestClient(create_studio_app(proj)).get("/api/project").json()
    assert info["excluded"] == ["undated.jpg"]
    photo = next(ph for ph in info["photos"] if ph["name"] == "undated.jpg")
    assert photo["usable"]
    assert load_labels(proj / "labels.json")["undated.jpg"].status == "labeled"
    assert load_project(proj).excluded == ["undated.jpg"]


def test_excluded_photos_do_not_limit_the_crop(project):
    _, proj = project
    labels = load_labels(proj / "labels.json")
    near_border = ScoredEllipse(cx=35, cy=60, a=30, b=28, angle_deg=10, score=0.9)
    labels["undated.jpg"] = labels["undated.jpg"].model_copy(update={"ellipses": [near_border]})
    save_labels(proj / "labels.json", labels)
    c = TestClient(create_studio_app(proj))
    info = c.get("/api/project").json()
    update = {"order": info["order"], "excluded": ["undated.jpg"], "settings": info["settings"]}
    excluded = c.put("/api/project", json=update).json()
    assert excluded["extent"] > info["extent"]
    assert excluded["frames"] == info["frames"] - 1


def test_project_picks_up_new_and_deleted_files(project):
    _, proj = project
    (proj / "undated.jpg").unlink()
    write_jpeg(proj / "added.jpg", T0 + timedelta(days=5))
    c = TestClient(create_studio_app(proj))
    assert c.get("/api/project").json()["order"] == [
        "20250719-090000_early.jpg",
        "20250720-090000_late.jpg",
        "added.jpg",
    ]


def test_preview_is_an_aligned_jpeg(project):
    _, proj = project
    c = TestClient(create_studio_app(proj))
    url = c.get("/api/project").json()["photos"][0]["preview"]
    r = c.get(url)
    assert r.headers["content-type"] == "image/jpeg"
    with Image.open(
        proj
        / ".thumbs"
        / "preview"
        / next(p.name for p in (proj / ".thumbs" / "preview").iterdir())
    ) as im:
        assert im.size == (320, 320)
    assert c.get("/api/preview/nope.jpg").status_code == 404


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_export(project):
    _, proj = project
    assert is_project(proj)
    c = TestClient(create_studio_app(proj))
    settings = {**c.get("/api/project").json()["settings"], "size": 64, "formats": ["mp4"]}
    assert c.post("/api/export", json=settings).json()["state"] == "running"
    for _ in range(200):
        status = c.get("/api/export").json()
        if status["state"] != "running":
            break
        time.sleep(0.05)
    assert status["state"] == "done", status
    assert status["outputs"] == ["photos.mp4"]
    r = c.get("/api/export/file/photos.mp4")
    assert r.status_code == 200 and len(r.content) > 0
    assert c.get("/api/export/file/..%2Fproject.json").status_code == 404
    assert c.post("/api/export", json={**settings, "formats": []}).status_code == 422


@pytest.mark.parametrize("by", ["color", "geometry"])
def test_sorted_order_keeps_every_photo(project, by):
    _, proj = project
    c = TestClient(create_studio_app(proj))
    order = c.get("/api/project").json()["order"]
    r = c.get("/api/sorted", params={"by": by})
    assert r.status_code == 200
    assert sorted(r.json()["order"]) == sorted(order)
    assert c.get("/api/sorted", params={"by": "size"}).status_code == 422
