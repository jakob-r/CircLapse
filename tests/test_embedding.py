import numpy as np
import pytest

from circlapse.studio import embedding


def weights_cached():
    if not embedding.available():
        return False
    from huggingface_hub import try_to_load_from_cache

    repo = f"timm/{embedding.MODEL}"
    return isinstance(try_to_load_from_cache(repo, "model.safetensors"), str)


@pytest.mark.skipif(not weights_cached(), reason="needs torch, timm and cached DINOv2 weights")
def test_embed_gives_unit_rows_that_tell_frames_apart():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (embedding.SIZE, embedding.SIZE, 3), dtype=np.uint8)
    flat = np.full_like(noise, 128)
    e = embedding.embed([noise, noise, flat])
    assert e.shape[0] == 3
    assert np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)
    assert np.linalg.norm(e[0] - e[1]) < 1e-3 < np.linalg.norm(e[0] - e[2])


def test_embed_nothing():
    assert embedding.embed([]).shape[0] == 0
