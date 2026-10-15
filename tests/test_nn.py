import numpy as np
import pytest

from circlapse.ellipse import Ellipse

pytest.importorskip("torch")
pytest.importorskip("timm")

from circlapse.nn.data import (  # noqa: E402
    Augment,
    augment,
    decode,
    encode,
    letterbox,
    target_mask,
    transform_ellipse,
)
from circlapse.nn.infer import fit_mask  # noqa: E402


@pytest.mark.parametrize("angle", [0, 45, 90, 170])
def test_encode_decode_round_trip(angle):
    e = Ellipse(cx=100, cy=150, a=80, b=50, angle_deg=angle)
    d = decode(encode(e, 384), 384)
    assert e.iou(d) > 0.999
    assert d.angle_deg == pytest.approx(angle, abs=1e-3)


def test_encoding_is_continuous_for_circles():
    near = [Ellipse(cx=0, cy=0, a=50.01, b=50, angle_deg=t) for t in (0, 60, 120)]
    codes = np.stack([encode(e, 100) for e in near])
    assert np.ptp(codes, axis=0).max() < 1e-3


def test_letterbox_maps_ellipse():
    img = np.zeros((300, 600, 3), np.uint8)
    e = Ellipse(cx=300, cy=150, a=100, b=60, angle_deg=30)
    canvas, h = letterbox(img, 200)
    c = transform_ellipse(e, h)
    assert canvas.shape == (200, 200, 3)
    assert (c.cx, c.cy, c.a, c.b) == pytest.approx((100, 100, 100 / 3, 20), abs=1e-3)
    back = transform_ellipse(c, np.linalg.inv(h))
    assert back.iou(e) > 0.999


def test_augment_keeps_labels_on_pixels():
    """A drawn disk and its label stay aligned through the random homography."""
    import cv2

    img = np.zeros((400, 600, 3), np.uint8)
    e = Ellipse(cx=320, cy=180, a=120, b=80, angle_deg=20)
    cv2.ellipse(img, e.to_cv_box(), (255, 255, 255), -1)
    aug = Augment(photometric=False)
    for seed in range(5):
        out, (t,) = augment(np.random.default_rng(seed), img, [e], 256, aug)
        drawn = out[..., 0] > 127
        assert (drawn != t.mask(drawn.shape)).mean() < 0.01


def test_fit_mask_recovers_target():
    e = Ellipse(cx=180, cy=200, a=120, b=70, angle_deg=35)
    m = target_mask(e, 384)
    prob = cv2_resize(m, 384)
    fitted, fit_iou = fit_mask(prob)
    assert fitted is not None
    assert fitted.iou(e) > 0.97
    assert fit_iou > 0.95
    assert fit_mask(np.zeros((64, 64), np.float32)) == (None, 0.0)


def cv2_resize(m, size):
    import cv2

    return cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR)
