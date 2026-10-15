from circlapse.labels import Dataset, ImageLabel, ScoredEllipse, load_labels, save_labels
from circlapse.metrics import evaluate

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
    gt = {
        "hit": ImageLabel(ellipses=[E], prominent=0, status="labeled"),
        "miss": ImageLabel(ellipses=[E], prominent=0, status="labeled"),
        "dropped": ImageLabel(ellipses=[E], prominent=0, status="labeled"),
        "fp": ImageLabel(status="no_ellipse"),
        "ignored": ImageLabel(status="unlabeled"),
    }
    pred = {
        "hit": ImageLabel(ellipses=[E], prominent=0, status="labeled", runtime_s=0.5),
        "miss": ImageLabel(ellipses=[far], prominent=0, status="labeled"),
        "fp": ImageLabel(ellipses=[E], prominent=0, status="labeled"),
    }
    summary = evaluate(gt, pred)
    outcomes = {r.name: r.outcome for r in summary.results}
    assert outcomes == {"hit": "hit", "miss": "miss", "dropped": "dropped", "fp": "false_positive"}
    assert summary.hit_rate == 1 / 3
    assert "hit rate" in summary.format()
