"""Camera RAW files (Sony ARW, Canon CR2/CR3, Nikon NEF, Fuji RAF, Adobe DNG, ...) as JPEG.

Pixels are decoded by LibRaw through rawpy, already rotated to the camera's orientation. PIL
cannot open RAW files, so capture time and GPS are read with exifread and written into the JPEG,
where the rest of the pipeline reads them like any other photo.
"""

from __future__ import annotations

from pathlib import Path

import exifread
import rawpy
from PIL import Image

RAW_SUFFIXES = {".arw", ".cr2", ".cr3", ".nef", ".raf", ".dng", ".orf", ".rw2"}
JPEG_SUFFIXES = {".jpg", ".jpeg"}

EXIF_IFD = 0x8769
GPS_IFD = 0x8825
DATETIME = 306
DATETIME_ORIGINAL = 36867
DATETIME_DIGITIZED = 36868


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_SUFFIXES


def raw_to_jpeg(src: Path, dst: Path, quality: int = 95) -> None:
    """Decode the RAW file ``src`` into the JPEG ``dst`` with its capture time and GPS."""
    with rawpy.imread(str(src)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
    Image.fromarray(rgb).save(dst, "JPEG", quality=quality, exif=_exif(src))


def _exif(src: Path) -> Image.Exif:
    with src.open("rb") as f:
        tags = exifread.process_file(f, details=False)
    exif = Image.Exif()
    if "Image DateTime" in tags:
        exif[DATETIME] = str(tags["Image DateTime"])
    sub = exif.get_ifd(EXIF_IFD)
    for key, tag in (
        ("EXIF DateTimeOriginal", DATETIME_ORIGINAL),
        ("EXIF DateTimeDigitized", DATETIME_DIGITIZED),
    ):
        if key in tags:
            sub[tag] = str(tags[key])
    gps = exif.get_ifd(GPS_IFD)
    for ref_key, value_key, ref_tag, value_tag in (
        ("GPS GPSLatitudeRef", "GPS GPSLatitude", 1, 2),
        ("GPS GPSLongitudeRef", "GPS GPSLongitude", 3, 4),
    ):
        if value_key in tags and ref_key in tags:
            gps[ref_tag] = str(tags[ref_key])
            gps[value_tag] = tuple(float(x) for x in tags[value_key].values)
    return exif
