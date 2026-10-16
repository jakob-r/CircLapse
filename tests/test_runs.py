import json

from circlapse.ellipse import Ellipse
from circlapse.labels import ImageLabel, ScoredEllipse, save_labels
from circlapse.runs import RESULTS, record


def _label(e: Ellipse) -> ImageLabel:
    return ImageLabel(ellipses=[ScoredEllipse(**e.model_dump())], prominent=0, status="labeled")


def test_record_scores_run_and_updates_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    circle = Ellipse(cx=100, cy=100, a=50, b=50, angle_deg=0)
    squashed = Ellipse(cx=100, cy=100, a=56, b=44, angle_deg=0)  # same area, IoU < 0.9
    save_labels(tmp_path / "gt.json", {"a.jpg": _label(circle), "unused.jpg": _label(circle)})
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    save_labels(run / "cv_mask.json", {"a.jpg": _label(squashed)})
    (run / "meta.json").write_text(json.dumps({"run": "r1", "commit": "abc", "dirty": True}))

    row = record(run, tmp_path / "gt.json")
    scores = json.loads((run / "meta.json").read_text())["scores"]
    assert scores["photos"] == 1  # only photos the run predicted
    assert scores["mask"]["hit"] == 0.0 and scores["mask_rounded"]["hit"] == 1.0
    assert "abc+dirty" in row

    record(run, tmp_path / "gt.json")  # re-recording replaces the row
    assert RESULTS.read_text().count("| r1 |") == 1
