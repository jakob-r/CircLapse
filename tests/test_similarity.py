import numpy as np

from circlapse.studio.similarity import geometry_features, similarity_path


def path_length(x, path):
    return sum(np.linalg.norm(x[a] - x[b]) for a, b in zip(path, path[1:], strict=False))


def test_similarity_path_follows_a_line():
    rng = np.random.default_rng(0)
    t = rng.permutation(30).astype(float)
    x = np.stack([t, 0.01 * rng.normal(size=30)], axis=1)
    path = similarity_path(x)
    assert sorted(path) == list(range(30))
    values = t[path]
    assert list(values) in (sorted(values), sorted(values, reverse=True))


def test_similarity_path_small_inputs():
    assert similarity_path(np.zeros((0, 3))) == []
    assert similarity_path(np.zeros((2, 3))) == [0, 1]


def spokes(n, rotation=0.0):
    y, x = np.mgrid[:96, :96] - 47.5
    angle = np.arctan2(y, x) + rotation
    gray = (127 + 120 * np.sign(np.sin(n * angle))).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def test_geometry_features_ignore_rotation():
    a = geometry_features(spokes(6))
    rotated = geometry_features(spokes(6, rotation=0.4))
    other = geometry_features(spokes(11))
    assert np.linalg.norm(a - rotated) < 0.5 * np.linalg.norm(a - other)
