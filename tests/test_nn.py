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


def _encoded(*es, size=384):
    import torch

    return torch.from_numpy(np.stack([encode(e, size) for e in es]))


def test_loss_terms_match_metric():
    import torch

    from circlapse.metrics import rectification_error
    from circlapse.nn.train import _shape, rectification_distance

    gt = Ellipse(cx=200, cy=180, a=90, b=70, angle_deg=30)
    pred = Ellipse(cx=210, cy=175, a=100, b=60, angle_deg=50)
    (_, r_p, v_p), (_, r_g, v_g) = _shape(_encoded(pred)), _shape(_encoded(gt))
    rect = -torch.expm1(-rectification_distance(v_p, v_g)).item()
    assert rect == pytest.approx(rectification_error(gt, pred), abs=1e-4)
    assert (r_p / r_g).item() == pytest.approx(pred.mean_radius / gt.mean_radius, rel=1e-4)


def test_loss_accepts_sister_ring_but_prefers_prominent():
    import torch

    from circlapse.nn.data import target_mask
    from circlapse.nn.train import TARGET_SWITCH_COST, loss_fn

    prominent = Ellipse(cx=190, cy=200, a=100, b=90, angle_deg=10)
    ring = prominent.scaled(0.7).model_copy(update={"cx": 190, "cy": 200})
    other = Ellipse(cx=90, cy=90, a=40, b=30, angle_deg=0)
    masks = torch.from_numpy(np.stack([target_mask(e, 384) for e in (prominent, ring, other)]))
    batch = {"masks": masks[None], "params": _encoded(prominent, ring, other)[None]}

    def loss(pred, valid=(True, True, False)):
        logits = (torch.from_numpy(target_mask(pred, 384))[None, None] - 0.5) * 40
        b = {**batch, "valid": torch.tensor([valid])}
        return loss_fn(logits, _encoded(pred), b)

    on_prominent, parts = loss(prominent)
    on_ring, ring_parts = loss(ring)
    assert parts["prominent"] == 1 and ring_parts["prominent"] == 0
    assert on_prominent < on_ring < on_prominent + TARGET_SWITCH_COST + 0.05
    assert loss(other)[0] > on_ring + 0.5  # invalid target: not accepted
    assert loss(ring, (True, False, False))[0] > on_ring + 0.5


def test_synth_targets_skip_distractors():
    from circlapse.nn.data import concentric

    prominent = Ellipse(cx=200, cy=200, a=100, b=95, angle_deg=0)
    wall = Ellipse(cx=201, cy=200, a=130, b=124, angle_deg=0)
    distractor = Ellipse(cx=80, cy=300, a=30, b=25, angle_deg=40)
    assert concentric(prominent, [wall, distractor]) == [wall]
