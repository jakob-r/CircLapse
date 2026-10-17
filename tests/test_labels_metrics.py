import pytest

from circlapse.ellipse import Ellipse
from circlapse.labels import Dataset, ImageLabel, ScoredEllipse, load_labels, save_labels
from circlapse.metrics import evaluate, rectification_error, tolerance_score

E = ScoredEllipse(cx=50, cy=50, a=30, b=20, angle_deg=10)


def test_labels_round_trip(tmp_path):
    labels = {
        "b.jpg": ImageLabel(ellipses=[E], prominent=0, status="labeled"),
        "a.jpg": ImageLabel(status="no_ellipse"),
    }
    path = tmp_path / "labels.json"
    save_labels(path, labels)
    assert load_labels(path) == labels
    assert list(load_labels(path)) == ["a.jpg", "b.jpg"]
    assert not list(tmp_path.glob("*.tmp"))


def test_dataset_rejects_paths_outside(tmp_path):
    (tmp_path / "img.jpg").write_bytes(b"")
    (tmp_path / ".hidden.jpg").write_bytes(b"")
    ds = Dataset(tmp_path)
    assert ds.images() == ["img.jpg"]
    for bad in ["../x.jpg", "missing.jpg"]:
        try:
            ds.image_path(bad)
        except FileNotFoundError:
            continue
        raise AssertionError(bad)


def test_evaluate_outcomes():
    far = ScoredEllipse(cx=200, cy=200, a=30, b=20)
    inner = ScoredEllipse(cx=50, cy=50, a=24, b=16, angle_deg=10)  # concentric, 0.8x
    wall = ScoredEllipse(cx=50, cy=50, a=45, b=30, angle_deg=10)  # concentric, 1.5x
    gt = {
        "hit": ImageLabel(ellipses=[E], prominent=0, status="labeled"),
        "other": ImageLabel(ellipses=[E, wall], prominent=0, status="labeled"),
        "good": ImageLabel(ellipses=[E], prominent=0, status="labeled"),
        "miss": ImageLabel(ellipses=[E], prominent=0, status="labeled"),
        "dropped": ImageLabel(ellipses=[E], prominent=0, status="labeled"),
        "fp": ImageLabel(status="no_ellipse"),
        "ignored": ImageLabel(status="unlabeled"),
    }
    pred = {
        "hit": ImageLabel(ellipses=[E], prominent=0, status="labeled", runtime_s=0.5),
        "other": ImageLabel(ellipses=[wall], prominent=0, status="labeled"),
        "good": ImageLabel(ellipses=[inner], prominent=0, status="labeled"),
        "miss": ImageLabel(ellipses=[far], prominent=0, status="labeled"),
        "fp": ImageLabel(ellipses=[E], prominent=0, status="labeled"),
    }
    summary = evaluate(gt, pred)
    outcomes = {r.name: r.outcome for r in summary.results}
    assert outcomes == {
        "hit": "hit",
        "other": "hit",
        "good": "good",
        "miss": "miss",
        "dropped": "dropped",
        "fp": "false_positive",
    }
    other = next(r for r in summary.results if r.name == "other")
    assert other.matched == 1 and not other.strict_hit
    assert summary.hit_rate == 2 / 5
    assert summary.good_rate == 3 / 5
    assert summary.strict_hit_rate == 1 / 5
    assert "good rate" in summary.format()


def test_rectification_error():
    e = Ellipse(cx=0, cy=0, a=30, b=15, angle_deg=40)
    # independent of position and scale
    assert rectification_error(e, Ellipse(cx=9, cy=-3, a=60, b=30, angle_deg=40)) == pytest.approx(
        0, abs=1e-9
    )
    # predicting a circle leaves the annotated aspect ratio untouched
    circle = Ellipse(cx=0, cy=0, a=10, b=10)
    assert rectification_error(e, circle) == pytest.approx(0.5)
    # for near-circles the angle hardly matters
    near = Ellipse(cx=0, cy=0, a=100, b=99, angle_deg=0)
    assert rectification_error(near, near.model_copy(update={"angle_deg": 90})) < 0.03
    # but for eccentric ellipses it does
    assert rectification_error(e, e.model_copy(update={"angle_deg": 130})) > 0.5


def test_tolerance_score():
    e = Ellipse(cx=0, cy=0, a=20, b=20)
    assert tolerance_score(e, e) == 0
    assert tolerance_score(e, Ellipse(cx=1, cy=0, a=22, b=22)) <= 1
    assert tolerance_score(e, Ellipse(cx=5, cy=0, a=20, b=20)) > 1
    assert tolerance_score(e, Ellipse(cx=0, cy=0, a=40, b=40)) > 1


def test_combine_heads_keeps_mask_only_when_params_agree():
    from circlapse.detect.base import combine_heads

    params = Ellipse(cx=100, cy=100, a=50, b=48, angle_deg=0)
    tight = Ellipse(cx=102, cy=100, a=46, b=45, angle_deg=5)  # same ring, slightly smaller
    other_ring = Ellipse(cx=100, cy=100, a=20, b=19, angle_deg=0)  # inner sister ring
    elsewhere = Ellipse(cx=200, cy=100, a=50, b=48, angle_deg=0)
    assert combine_heads(tight, params) == tight
    assert combine_heads(other_ring, params) == params
    assert combine_heads(elsewhere, params) == params
    assert combine_heads(None, params) == params
