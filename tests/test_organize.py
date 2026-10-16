from datetime import datetime, timedelta
from fractions import Fraction

import numpy as np
import pytest
from PIL import Image

from circlapse.labels import ScoredEllipse, load_labels
from circlapse.organize import (
    Photo,
    apply_moves,
    chain_clusters,
    cluster_name,
    detect_all,
    plan_moves,
    read_photo,
    safe_name,
)

T0 = datetime(2025, 7, 19, 9, 0)
GAP = timedelta(hours=6)


def photo(name, hours, lat=None, lon=None):
    return Photo(name, T0 + timedelta(hours=hours), lat, lon)


def write_jpeg(path, time=None, gps=None, value=0):
    im = Image.fromarray(np.full((32, 32, 3), value, np.uint8))
    exif = Image.Exif()
    if time is not None:
        exif.get_ifd(0x8769)[36867] = time.strftime("%Y:%m:%d %H:%M:%S")
    if gps is not None:
        lat, lon = gps

        def dms(x):
            x = abs(x)
            d, m = int(x), int(x * 60) % 60
            return (Fraction(d), Fraction(m), Fraction(round((x * 3600) % 60 * 100), 100))

        exif.get_ifd(0x8825).update(
            {1: "N" if lat >= 0 else "S", 2: dms(lat), 3: "E" if lon >= 0 else "W", 4: dms(lon)}
        )
    im.save(path, exif=exif)


def test_read_photo_exif(tmp_path):
    write_jpeg(tmp_path / "a.jpg", T0, (51.152, -14.987))
    write_jpeg(tmp_path / "b.jpg")
    a = read_photo(tmp_path / "a.jpg")
    assert a.time == T0
    assert abs(a.lat - 51.152) < 1e-3 and abs(a.lon + 14.987) < 1e-3
    b = read_photo(tmp_path / "b.jpg")
    assert b.time is None and not b.has_gps


def test_chain_clusters_by_time():
    # chained: each gap below 6h although first to last is 11h
    photos = [photo("c", 11), photo("a", 0), photo("b", 5), photo("d", 30), Photo("x")]
    clusters = chain_clusters(photos, GAP, max_km=20)
    assert [[p.name for p in c] for c in clusters] == [["a", "b", "c"], ["d"]]


def test_chain_clusters_by_location_skips_photos_without_gps():
    photos = [
        photo("berlin1", 0, 52.52, 13.40),
        photo("nogps", 1),
        photo("berlin2", 2, 52.50, 13.45),
        photo("potsdam", 3, 52.39, 13.06),  # ~30 km from Berlin
    ]
    clusters = chain_clusters(photos, GAP, max_km=20)
    assert [[p.name for p in c] for c in clusters] == [["berlin1", "nogps", "berlin2"], ["potsdam"]]


def test_chain_clusters_merges_nights_at_the_same_place():
    night = timedelta(hours=36)
    photos = [
        photo("day1", 0, 50.85, 4.35),
        photo("day1-camera", 3),  # no GPS: the last geotagged photo before the night counts
        photo("day2-camera", 24),  # no GPS: the first geotagged photo after the night counts
        photo("day2", 25, 50.83, 4.36),  # ~2 km away
        photo("day4", 72, 50.83, 4.36),  # pause longer than 36 h
        photo("antwerp", 96, 51.22, 4.40),  # ~43 km away
        photo("nogps-a", 120),
        photo("nogps-b", 132),  # no GPS on either side: stays separate
    ]
    clusters = chain_clusters(photos, GAP, 20, night)
    assert [[p.name for p in c] for c in clusters] == [
        ["day1", "day1-camera", "day2-camera", "day2"],
        ["day4"],
        ["antwerp"],
        ["nogps-a"],
        ["nogps-b"],
    ]
    assert len(chain_clusters(photos, GAP, 20)) == 6


def test_cluster_name():
    cluster = [photo("a", 0, 1, 1), photo("b", 20, 2, 2), photo("c", 30, 2, 2), photo("d", 31)]
    places = {1: "Bad Muskau", 2: "Görlitz / Zgorzelec"}
    assert cluster_name(cluster, None) == "2025-07-19_2025-07-20"
    assert cluster_name(cluster, lambda lat, lon: places[lat]) == (
        "2025-07-19_2025-07-20_Görlitz-Zgorzelec"
    )
    assert safe_name("  ../..  ") == ""


def test_plan_moves_unique_folders_and_undated():
    accepted = [photo("a.jpg", 0), photo("b.jpg", 7), Photo("x.jpg")]
    moves, folders = plan_moves(accepted, [Photo("r.jpg")], GAP, 20, None)
    assert {m.src: m.dst for m in moves} == {
        "r.jpg": "rejected/r.jpg",
        "a.jpg": "2025-07-19_2025-07-19/20250719-090000_a.jpg",
        "b.jpg": "2025-07-19_2025-07-19_2/20250719-160000_b.jpg",
        "x.jpg": "undated/x.jpg",
    }
    assert set(folders["undated"]) == {"x.jpg"}


class FakeDetector:
    name = "fake"

    def detect(self, image):
        if image.mean() < 128:
            return []
        return [ScoredEllipse(cx=16, cy=16, a=10, b=8, angle_deg=0, score=0.9)]


def test_organize_end_to_end(tmp_path):
    write_jpeg(tmp_path / "good1.jpg", T0, value=200)
    write_jpeg(tmp_path / "good2.jpg", T0 + timedelta(days=2), value=200)
    write_jpeg(tmp_path / "bad.jpg", T0, value=0)
    photos = detect_all(tmp_path, FakeDetector(), log=lambda _: None)
    accepted = [p for p in photos if p.label.ellipses]
    rejected = [p for p in photos if not p.label.ellipses]
    moves, folders = plan_moves(accepted, rejected, GAP, 20, None)
    apply_moves(tmp_path, moves, folders)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.jpg")) == [
        "2025-07-19_2025-07-19/20250719-090000_good1.jpg",
        "2025-07-21_2025-07-21/20250721-090000_good2.jpg",
        "rejected/bad.jpg",
    ]
    labels = load_labels(tmp_path / "2025-07-19_2025-07-19" / "labels.json")
    assert labels["20250719-090000_good1.jpg"].prominent_ellipse.score == 0.9


def test_detect_all_uses_fallback_metadata(tmp_path):
    write_jpeg(tmp_path / "exif.jpg", T0, value=200)
    write_jpeg(tmp_path / "plain.jpg", value=200)
    later = T0 + timedelta(hours=1)
    fallback = {
        "exif.jpg": Photo("exif.jpg", later, 1.0, 2.0),
        "plain.jpg": Photo("plain.jpg", later, 3.0, 4.0),
    }
    photos = {p.name: p for p in detect_all(tmp_path, FakeDetector(), lambda _: None, fallback)}
    assert (photos["exif.jpg"].time, photos["exif.jpg"].lat) == (T0, 1.0)
    assert (photos["plain.jpg"].time, photos["plain.jpg"].lon) == (later, 4.0)


def test_export_range_from_apple_photos(tmp_path, monkeypatch):
    from datetime import UTC, date

    osxphotos = pytest.importorskip("osxphotos")
    from circlapse.apple_photos import export_range

    class Info:
        def __init__(self, name, location):
            self.original_filename = name
            self.date = T0.replace(tzinfo=UTC)
            self.location = location

    class Results:
        def __init__(self, exported):
            self.exported, self.error = exported, []

    class Exporter:
        def __init__(self, info):
            self.info = info

        def export(self, dest, options):
            assert options.convert_to_jpeg and options.download_missing
            if self.info.original_filename == "gone.heic":
                return Results([])
            return Results([str(dest / self.info.original_filename.replace(".heic", ".jpeg"))])

    class DB:
        def photos(self, images, movies, from_date, to_date):
            assert images and not movies
            assert (from_date, to_date) == (datetime(2025, 7, 19), datetime(2025, 7, 21))
            return [Info("a.heic", (52.5, 13.4)), Info("gone.heic", (None, None))]

    monkeypatch.setattr(osxphotos, "PhotoExporter", Exporter)
    meta = export_range(
        tmp_path / "out", date(2025, 7, 19), date(2025, 7, 20), lambda _: None, db=DB()
    )
    assert list(meta) == ["a.jpeg"]
    assert (meta["a.jpeg"].time, meta["a.jpeg"].lat) == (T0, 52.5)
