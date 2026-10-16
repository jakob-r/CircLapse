import numpy as np
import pytest

from circlapse.ellipse import Ellipse


def test_normalizes_axes_and_angle():
    e = Ellipse(cx=0, cy=0, a=10, b=20, angle_deg=170)
    assert (e.a, e.b) == (20, 10)
    assert e.angle_deg == pytest.approx(80)
    assert Ellipse(cx=0, cy=0, a=5, b=3, angle_deg=-30).angle_deg == pytest.approx(150)


@pytest.mark.parametrize("angle", [0, 33, 90, 145])
def test_fit_round_trip(angle):
    e = Ellipse(cx=120, cy=80, a=60, b=25, angle_deg=angle)
    f = Ellipse.from_points(e.sample_points(40))
    assert f.cx == pytest.approx(e.cx, abs=0.1)
    assert f.cy == pytest.approx(e.cy, abs=0.1)
    assert f.a == pytest.approx(e.a, abs=0.1)
    assert f.b == pytest.approx(e.b, abs=0.1)
    assert e.iou(f) > 0.99


def test_fit_needs_five_points():
    with pytest.raises(ValueError):
        Ellipse.from_points([[0, 0], [1, 1], [2, 0], [1, -1]])


def test_cv_box_round_trip():
    e = Ellipse(cx=10, cy=20, a=8, b=3, angle_deg=30)
    assert Ellipse.from_cv_box(e.to_cv_box()) == e


def test_iou():
    e = Ellipse(cx=0, cy=0, a=10, b=10)
    assert e.iou(e) == pytest.approx(1.0)
    assert e.iou(Ellipse(cx=100, cy=0, a=10, b=10)) == 0.0
    # concentric circles: IoU = (r1/r2)^2
    assert e.iou(Ellipse(cx=0, cy=0, a=5, b=5)) == pytest.approx(0.25, abs=0.01)


def test_mask_matches_contains():
    e = Ellipse(cx=50, cy=40, a=30, b=12, angle_deg=20)
    mask = e.mask((80, 100))
    ys, xs = np.mgrid[0:80, 0:100]
    expected = e.contains(xs, ys)
    assert (mask != expected).mean() < 0.02


def test_metrics_helpers():
    e = Ellipse(cx=0, cy=0, a=20, b=5)
    assert e.mean_radius == pytest.approx(10)
    assert e.center_error(Ellipse(cx=10, cy=0, a=20, b=5)) == pytest.approx(1.0)
    assert e.rel_size((120, 200, 3)) == pytest.approx(40 / 120)


def test_rounded_keeps_area_and_center():
    e = Ellipse(cx=10, cy=20, a=50, b=40, angle_deg=30)
    assert e.rounded(0.0).aspect == pytest.approx(e.aspect)
    assert e.rounded(1.0).aspect == pytest.approx(1.0)
    half = e.rounded(0.5)
    assert e.aspect < half.aspect < 1.0
    for r in (e.rounded(0.0), half, e.rounded(1.0)):
        assert r.a * r.b == pytest.approx(e.a * e.b)
        assert (r.cx, r.cy) == (e.cx, e.cy)
