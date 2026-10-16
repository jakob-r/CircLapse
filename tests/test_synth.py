import numpy as np

from circlapse.labels import load_labels
from circlapse.synth import SynthConfig, generate, generate_dataset


def test_label_matches_rendered_disk():
    """Clean filled disks: the labeled ellipse coincides with the rendered pixels."""
    cfg = SynthConfig(
        size=256, perspective=0.0, noise=False, max_distractors=0, style_weights=(1, 0, 0)
    )
    for seed in range(5):
        img, label = generate(np.random.default_rng(seed), cfg)
        e = label.prominent_ellipse
        assert e is not None
        mask = e.mask(img.shape)
        inner, outer = mask, ~mask
        # pixel colors are uniform inside the disk and differ from the boundary ring outside
        ring = _grow(e, 1.15).mask(img.shape) & outer
        inside_color = np.median(img[_grow(e, 0.85).mask(img.shape) & inner], axis=0)
        assert np.abs(img[ring].astype(float) - inside_color).sum(axis=1).mean() > 20


def _grow(e, f):
    return type(e)(cx=e.cx, cy=e.cy, a=e.a * f, b=e.b * f, angle_deg=e.angle_deg)


def test_prominent_ellipse_is_large_enough():
    cfg = SynthConfig(size=256)
    for seed in range(200):
        img, label = generate(np.random.default_rng(seed), cfg)
        assert img.shape == (256, 256, 3) and img.dtype == np.uint8
        assert label.prominent == 0
        # perspective may shrink it slightly below the sampling minimum
        e = label.ellipses[0]
        assert e.rel_size(img.shape) > 0.25
        x0, y0, x1, y1 = e.bbox()
        assert min(x0, y0) >= 0.1 * 256 - 1e-6 and max(x1, y1) <= 0.9 * 256 + 1e-6


def test_generate_dataset(tmp_path):
    generate_dataset(tmp_path, 3, SynthConfig(size=64), seed=1)
    labels = load_labels(tmp_path / "labels.json")
    assert sorted(labels) == [f"synth_{i:06d}.jpg" for i in range(3)]
    assert all((tmp_path / n).exists() for n in labels)


def test_eccentricity_is_capped():
    cfg = SynthConfig(size=256, max_eccentricity=0.6)
    for seed in range(200):
        _, label = generate(np.random.default_rng(seed), cfg)
        assert all(e.eccentricity <= 0.6 + 1e-6 for e in label.ellipses)
