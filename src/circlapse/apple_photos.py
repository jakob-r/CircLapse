"""Export a date range from the macOS Photos library (needs the ``photos`` extra).

The library is only read through osxphotos; nothing in it is changed. Originals are exported,
HEIC and other formats converted to JPEG, and photos that only live in iCloud downloaded
through Photos. Photos' own date and position are returned for every exported file, so
images without EXIF (screenshots, messenger images) can still be clustered.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from circlapse.organize import Photo


def export_range(
    dest: Path,
    from_date: date,
    to_date: date,
    log: Callable[[str], None] = print,
    db: Any = None,
) -> dict[str, Photo]:
    """Export photos taken from ``from_date`` to ``to_date`` (inclusive) into ``dest``.

    Returns Photos' metadata by exported file name. ``db`` replaces the system library (tests).
    """
    from osxphotos import ExportOptions, PhotoExporter, PhotosDB

    photosdb = db if db is not None else PhotosDB()
    photos = photosdb.photos(
        images=True,
        movies=False,
        from_date=datetime.combine(from_date, time()),
        to_date=datetime.combine(to_date + timedelta(days=1), time()),
    )
    options = ExportOptions(convert_to_jpeg=True, download_missing=True)
    dest.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Photo] = {}
    for i, info in enumerate(photos, 1):
        results = PhotoExporter(info).export(dest, options=options)
        if not results.exported:
            log(f"[{i}/{len(photos)}] {info.original_filename}: not exported {results.error}")
            continue
        lat, lon = info.location
        for path in results.exported:
            meta[Path(path).name] = Photo(Path(path).name, info.date.replace(tzinfo=None), lat, lon)
        log(f"[{i}/{len(photos)}] exported {info.original_filename}")
    return meta
