"""Organize a dump of photos: reject photos without a good ellipse, cluster the rest into trips.

Clusters are chains in capture order: a photo joins the current cluster when it was taken at
most ``max_gap`` after the previous photo and, if both it and the cluster's last geotagged
photo have GPS, at most ``max_km`` away from it. Photos without GPS only use the time rule.
Neighbouring chains separated by a longer pause (a night) are merged again when the pause is
at most ``max_gap_same_place`` and the last geotagged photo before it and the first one after
it are at most ``max_km`` apart; without GPS on both sides they stay separate.
Kept photos are renamed ``YYYYMMDD-HHMMSS_<original>`` so name order is capture order, and
every cluster folder gets a ``labels.json`` with the detected ellipses.
"""

from __future__ import annotations

import math
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

from circlapse.detect.base import Detector
from circlapse.labels import ImageLabel, list_images, load_image, save_labels

REJECTED_DIR = "rejected"
UNDATED_DIR = "undated"

EXIF_IFD = 0x8769
GPS_IFD = 0x8825
DATETIME_ORIGINAL = 36867
DATETIME_DIGITIZED = 36868
DATETIME = 306

Geocoder = Callable[[float, float], str | None]


@dataclass
class Photo:
    name: str
    time: datetime | None = None
    lat: float | None = None
    lon: float | None = None
    label: ImageLabel = field(default_factory=ImageLabel)

    @property
    def has_gps(self) -> bool:
        return self.lat is not None and self.lon is not None


@dataclass
class Move:
    src: str
    dst: str  # relative to the input directory


def _parse_exif_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip("\x00 "), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _dms_to_deg(dms: object, ref: object) -> float | None:
    try:
        d, m, s = (float(x) for x in dms)  # type: ignore[attr-defined]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def read_photo(path: Path) -> Photo:
    """Capture time and GPS position from EXIF; time falls back to None, not the mtime."""
    photo = Photo(path.name)
    try:
        with Image.open(path) as im:
            exif = im.getexif()
    except OSError:
        return photo
    sub = exif.get_ifd(EXIF_IFD)
    for value in (sub.get(DATETIME_ORIGINAL), sub.get(DATETIME_DIGITIZED), exif.get(DATETIME)):
        if (t := _parse_exif_time(value)) is not None:
            photo.time = t
            break
    gps = exif.get_ifd(GPS_IFD)
    if 2 in gps and 4 in gps:
        lat, lon = _dms_to_deg(gps[2], gps.get(1)), _dms_to_deg(gps[4], gps.get(3))
        # (0, 0) is what many cameras write without a fix
        if lat is not None and lon is not None and (lat, lon) != (0.0, 0.0):
            photo.lat, photo.lon = lat, lon
    return photo


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _distance_km(a: Photo, b: Photo) -> float:
    assert a.lat is not None and a.lon is not None and b.lat is not None and b.lon is not None
    return haversine_km(a.lat, a.lon, b.lat, b.lon)


def chain_clusters(
    photos: Iterable[Photo],
    max_gap: timedelta,
    max_km: float,
    max_gap_same_place: timedelta | None = None,
) -> list[list[Photo]]:
    """Split dated photos into chains in capture order (see module docstring)."""
    ordered = sorted((p for p in photos if p.time is not None), key=lambda p: (p.time, p.name))
    clusters: list[list[Photo]] = []
    last_gps: Photo | None = None
    for p in ordered:
        cluster = clusters[-1] if clusters else None
        split = cluster is None
        if cluster is not None:
            assert p.time is not None and cluster[-1].time is not None
            split = p.time - cluster[-1].time > max_gap
            if not split and p.has_gps and last_gps is not None:
                split = _distance_km(p, last_gps) > max_km
        if split:
            clusters.append([p])
            last_gps = None
        else:
            clusters[-1].append(p)
        if p.has_gps:
            last_gps = p
    if max_gap_same_place is None:
        return clusters
    merged = clusters[:1]
    for cluster in clusters[1:]:
        prev = merged[-1]
        assert prev[-1].time is not None and cluster[0].time is not None
        before = next((p for p in reversed(prev) if p.has_gps), None)
        after = next((p for p in cluster if p.has_gps), None)
        if (
            cluster[0].time - prev[-1].time <= max_gap_same_place
            and before is not None
            and after is not None
            and _distance_km(before, after) <= max_km
        ):
            prev.extend(cluster)
        else:
            merged.append(cluster)
    return merged


def offline_geocoder() -> Geocoder | None:
    """City name lookup via the offline ``reverse_geocode`` package (``geo`` extra), if present."""
    try:
        import reverse_geocode
    except ImportError:
        return None

    def lookup(lat: float, lon: float) -> str | None:
        # GeoNames lists districts of big cities as places of their own ("Mitte" for Berlin)
        place = reverse_geocode.get((lat, lon), min_population=300_000)
        if haversine_km(lat, lon, place["latitude"], place["longitude"]) > 20:
            place = reverse_geocode.get((lat, lon), min_population=5_000)
        city = place.get("city")
        return str(city) if city else None

    return lookup


def safe_name(text: str) -> str:
    """File-name safe version of a place name: keeps letters and digits, joins words with '-'."""
    return "-".join(re.findall(r"\w+", text, flags=re.UNICODE)).strip("_-")


def cluster_name(cluster: list[Photo], geocoder: Geocoder | None) -> str:
    """``YYYY-MM-DD_YYYY-MM-DD[_Place]`` with the most common place among geotagged photos."""
    first, last = cluster[0].time, cluster[-1].time
    assert first is not None and last is not None
    name = f"{first:%Y-%m-%d}_{last:%Y-%m-%d}"
    if geocoder is not None:
        places = Counter(
            place
            for p in cluster
            if p.lat is not None and p.lon is not None
            if (place := geocoder(p.lat, p.lon))
        )
        if places and (place := safe_name(places.most_common(1)[0][0])):
            name += f"_{place}"
    return name


def dated_name(photo: Photo) -> str:
    """``YYYYMMDD-HHMMSS_<original>`` so that name order is capture order; undated keep theirs."""
    return f"{photo.time:%Y%m%d-%H%M%S}_{photo.name}" if photo.time else photo.name


def _unique(name: str, taken: set[str]) -> str:
    candidate, i = name, 2
    while candidate in taken:
        candidate = f"{name}_{i}"
        i += 1
    taken.add(candidate)
    return candidate


def plan_moves(
    accepted: list[Photo],
    rejected: list[Photo],
    max_gap: timedelta,
    max_km: float,
    geocoder: Geocoder | None,
    max_gap_same_place: timedelta | None = None,
) -> tuple[list[Move], dict[str, dict[str, ImageLabel]]]:
    """Moves for every photo and the ``labels.json`` content of every cluster folder."""
    moves = [Move(p.name, f"{REJECTED_DIR}/{p.name}") for p in rejected]
    folders: dict[str, dict[str, ImageLabel]] = {}
    taken: set[str] = set()
    groups = [
        (cluster_name(c, geocoder), c)
        for c in chain_clusters(accepted, max_gap, max_km, max_gap_same_place)
    ]
    if undated := sorted((p for p in accepted if p.time is None), key=lambda p: p.name):
        groups.append((UNDATED_DIR, undated))
    for name, photos in groups:
        folder = _unique(name, taken)
        labels = folders.setdefault(folder, {})
        for p in photos:
            new = dated_name(p)
            moves.append(Move(p.name, f"{folder}/{new}"))
            labels[new] = p.label
    return moves, folders


def detect_all(
    directory: Path,
    detector: Detector,
    log: Callable[[str], None] = print,
    fallback: dict[str, Photo] | None = None,
) -> list[Photo]:
    """EXIF metadata and detected ellipses for every image directly in ``directory``.

    ``fallback`` provides time and position by file name where EXIF has none.
    """
    photos = []
    names = list_images(directory)
    for i, name in enumerate(names, 1):
        path = directory / name
        photo = read_photo(path)
        if fallback and (known := fallback.get(name)):
            if photo.time is None:
                photo.time = known.time
            if not photo.has_gps:
                photo.lat, photo.lon = known.lat, known.lon
        try:
            ellipses = detector.detect(load_image(path))
        except OSError as e:
            log(f"[{i}/{len(names)}] {name}: unreadable ({e})")
            ellipses = []
        photo.label = ImageLabel(
            ellipses=ellipses,
            prominent=0 if ellipses else None,
            status="labeled" if ellipses else "no_ellipse",
            source="detector",
        )
        score = ellipses[0].score if ellipses else None
        result = f"score {score:.2f}" if score is not None else "rejected"
        log(f"[{i}/{len(names)}] {name}: {result}")
        photos.append(photo)
    return photos


def apply_moves(
    directory: Path, moves: list[Move], folders: dict[str, dict[str, ImageLabel]]
) -> None:
    for move in moves:
        dst = directory / move.dst
        if dst.exists():
            raise FileExistsError(dst)
    for move in moves:
        dst = directory / move.dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(directory / move.src, dst)
    for folder, labels in folders.items():
        save_labels(directory / folder / "labels.json", labels)
