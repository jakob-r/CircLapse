import pytest
from fastapi.testclient import TestClient
from PIL import Image

from circlapse.annotate.app import create_app
from circlapse.labels import load_labels


@pytest.fixture
def client(tmp_path):
    Image.new("RGB", (300, 200), "gray").save(tmp_path / "a.jpg")
    im = Image.new("RGB", (300, 200), "gray")
    exif = im.getexif()
    exif[0x0112] = 6  # rotated 90°: displayed as 200x300
    im.save(tmp_path / "rotated.jpg", exif=exif)
    return TestClient(create_app(tmp_path)), tmp_path


def test_list_images_reports_exif_corrected_size(client):
    c, _ = client
    images = {i["name"]: i for i in c.get("/api/images").json()}
    assert (images["a.jpg"]["width"], images["a.jpg"]["height"]) == (300, 200)
    assert (images["rotated.jpg"]["width"], images["rotated.jpg"]["height"]) == (200, 300)
    assert images["a.jpg"]["status"] == "unlabeled"


def test_thumb_and_image(client):
    c, _ = client
    assert c.get("/api/thumb/a.jpg").headers["content-type"] == "image/jpeg"
    assert c.get("/api/image/rotated.jpg").status_code == 200
    assert c.get("/api/image/nope.jpg").status_code == 404


def test_put_label_persists(client):
    c, path = client
    label = {
        "ellipses": [{"cx": 150, "cy": 100, "a": 40, "b": 60, "angle_deg": 0}],
        "prominent": 0,
        "status": "labeled",
    }
    r = c.put("/api/labels/a.jpg", json=label)
    assert r.status_code == 200
    stored = load_labels(path / "labels.json")["a.jpg"]
    assert stored.ellipses[0].a == 60  # normalized
    assert c.get("/api/labels/a.jpg").json()["status"] == "labeled"
    assert c.put("/api/labels/a.jpg", json={**label, "prominent": 3}).status_code == 422


def test_fit(client):
    c, _ = client
    pts = [[110, 100], [150, 140], [190, 100], [150, 60], [178.28, 128.28]]
    e = c.post("/api/fit", json={"points": pts}).json()
    assert e["cx"] == pytest.approx(150, abs=0.5)
    assert c.post("/api/fit", json={"points": pts[:3]}).status_code == 422
