import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

from circlapse.ellipse import Ellipse
from circlapse.labels import ImageLabel, ScoredEllipse
from circlapse.render import (
    RenderConfig,
    choose_extent,
    max_extent,
    plan_frames,
    render,
    warp,
    warp_matrix,
)


def test_warp_matrix_maps_ellipse_to_centered_circle():
    e = Ellipse(cx=300, cy=200, a=120, b=90, angle_deg=30)
    m = warp_matrix(e, extent=2.0, size=400)
    pts = e.sample_points(32)
    out = pts @ m[:, :2].T + m[:, 2]
    r = np.hypot(out[:, 0] - 200, out[:, 1] - 200)
    np.testing.assert_allclose(r, 100, rtol=1e-6)


def test_max_extent_circle():
    # Circle of radius 100 at (150, 200) in a 400x600 (h x w) photo: 50 px to the left edge
    e = Ellipse(cx=150, cy=200, a=100, b=100)
    assert max_extent(e, (400, 600)) == pytest.approx(1.5)


def test_max_extent_frame_lies_inside_photo():
    e = Ellipse(cx=320, cy=260, a=110, b=95, angle_deg=70)
    shape = (500, 640)
    s = max_extent(e, shape)
    m = warp_matrix(e, s, 100)
    inv = cv2.invertAffineTransform(m)
    corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    src = corners @ inv[:, :2].T + inv[:, 2]
    assert src.min() >= -1e-6
    assert (src[:, 0] <= 640 + 1e-6).all() and (src[:, 1] <= 500 + 1e-6).all()
    assert (
        np.isclose(src[:, 0], 640).any()
        or np.isclose(src[:, 1], 500).any()
        or np.isclose(src, 0).any()
    )


def test_choose_extent_keep_and_caps():
    cfg = RenderConfig(keep=1.0, max_extent=3.0, min_extent=1.0)
    assert choose_extent([2.0, 1.5, 0.8], cfg) == 1.5  # 0.8 can't show the whole circle
    assert choose_extent([2.5, 2.0, 1.2, 4.0], RenderConfig(keep=0.75)) == 2.0
    assert choose_extent([5.0, 4.0], cfg) == 3.0
    with pytest.raises(ValueError):
        choose_extent([0.5], cfg)


def test_plan_frames_drops_tight_and_unlabeled():
    def label(cx):
        return ImageLabel(ellipses=[ScoredEllipse(cx=cx, cy=200, a=100, b=100)], prominent=0)

    labels = {"a.jpg": label(300), "b.jpg": label(120), "c.jpg": label(250)}
    shapes = {n: (400, 600) for n in ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]}
    frames, dropped, extent = plan_frames(labels, shapes, RenderConfig(keep=0.6))
    assert [f.name for f in frames] == ["a.jpg", "c.jpg"]
    assert dropped == ["b.jpg", "d.jpg"]
    assert extent == pytest.approx(2.0)


def test_warp_centers_the_disc():
    img = np.zeros((300, 400, 3), np.uint8)
    e = Ellipse(cx=250, cy=140, a=80, b=60, angle_deg=20)
    img[e.mask(img.shape)] = 255
    frame = warp(img, e, extent=1.5, size=150)
    ys, xs = np.nonzero(frame[..., 0] > 127)
    assert xs.mean() == pytest.approx(75, abs=1)
    assert ys.mean() == pytest.approx(75, abs=1)
    assert len(xs) == pytest.approx(np.pi * 50**2, rel=0.03)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_render_writes_videos(tmp_path):
    rng = np.random.default_rng(0)
    labels = {}
    for i in range(3):
        img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        e = ScoredEllipse(cx=80 + 5 * i, cy=60, a=40, b=35, angle_deg=10 * i)
        Image.fromarray(img).save(tmp_path / f"{i}.png")
        labels[f"{i}.png"] = ImageLabel(ellipses=[e], prominent=0)
    out = tmp_path / "out" / "video"
    result = render(tmp_path, labels, out, RenderConfig(size=64), frames_out=tmp_path / "frames")
    assert [p.name for p in result.outputs] == ["video.mp4", "video.webm", "video.gif"]
    assert all(p.stat().st_size > 0 for p in result.outputs)
    assert len(list((tmp_path / "frames").glob("*.png"))) == 3
