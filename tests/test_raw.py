import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from circlapse import raw
from circlapse.organize import read_photo
from circlapse.studio import project as project_module

# A camera RAW file to decode for real; the test is skipped without one.
SAMPLE_RAW = os.environ.get("CIRCLAPSE_TEST_RAW")


def test_raw_suffixes():
    assert raw.is_raw(Path("IMG_0001.ARW"))
    assert raw.is_raw(Path("x.cr3"))
    assert not raw.is_raw(Path("x.jpg"))


def test_exif_time_and_gps_survive_the_jpeg(tmp_path):
    # exifread reads any EXIF container, so a JPEG stands in for a RAW file here
    exif = Image.Exif()
    exif.get_ifd(0x8769)[36867] = "2025:07:19 09:30:00"
    exif.get_ifd(0x8825).update({1: "N", 2: (51.0, 9.0, 0.0), 3: "W", 4: (14.0, 59.0, 24.0)})
    src = tmp_path / "shot.jpg"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(src, exif=exif)

    dst = tmp_path / "copy.jpg"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(dst, exif=raw._exif(src))

    photo = read_photo(dst)
    assert photo.time == datetime(2025, 7, 19, 9, 30)
    assert photo.lat == pytest.approx(51.15)
    assert photo.lon == pytest.approx(-14.99, abs=1e-3)


@pytest.mark.skipif(not SAMPLE_RAW, reason="set CIRCLAPSE_TEST_RAW to a camera RAW file")
def test_raw_to_jpeg_decodes_a_camera_file(tmp_path):
    dst = tmp_path / "decoded.jpg"
    raw.raw_to_jpeg(Path(SAMPLE_RAW), dst)
    with Image.open(dst) as im:
        assert im.format == "JPEG"
        assert im.size[0] > 1000 and im.size[1] > 1000
        assert im.getexif().get(0x0112, 1) == 1  # pixels are already upright


def test_stage_converts_raw_and_links_images(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(src / "a.jpg")
    (src / "a.arw").write_bytes(b"raw")  # same name as a JPEG: skipped
    (src / "b.ARW").write_bytes(b"raw")
    (src / "notes.txt").write_text("not a photo")
    (src / ".hidden.arw").write_bytes(b"raw")

    def fake_convert(path, dst):
        Image.fromarray(np.full((8, 8, 3), 90, np.uint8)).save(dst)

    monkeypatch.setattr(project_module, "raw_to_jpeg", fake_convert)
    logs = []
    project_module._stage(src, work, logs.append)

    assert sorted(p.name for p in work.iterdir()) == ["a.jpg", "b.jpg"]
    assert (work / "a.jpg").resolve() == (src / "a.jpg").resolve()
    assert np.asarray(Image.open(work / "b.jpg"))[0, 0, 0] == 90
    assert any("a.arw" in m and "skipped" in m for m in logs)
