"""Допоміжні функції розпакування NPZ."""
import numpy as np
import pytest

from cxr_xai.pipeline import extract_npz


def _save(path, images, saver=np.savez):
    n = len(images)
    saver(path, image=images, image_name=np.array([f"a{i}" for i in range(n)]),
          image_label=np.zeros(n, dtype=int))
    return path


def test_npz_helpers(tmp_path):
    m = extract_npz
    assert m.name_prefix("COVID-123.png") == "COVID"
    assert m.name_prefix("CHNCXR_0001_0.png") == "CHNCXR"
    assert m.name_prefix("Viral Pneumonia-12") == "Viral Pneumonia"
    assert m.safe_name("a/b\\c?.png") == "a_b_c_"
    imgs = np.random.default_rng(0).random((4, 8, 8, 1)).astype(np.float32)
    rule, f, _ = m.decide_scaling(m.NpyStream(_save(tmp_path / "x.npz", imgs)), chunk=3)
    assert rule == "x255"
    out = m.to_uint8(imgs[0], rule, f)
    assert out.shape == (8, 8) and out.dtype == np.uint8
    rgb = np.repeat((imgs[0] * 255).astype(np.uint8), 3, axis=-1)
    assert m.to_uint8(rgb, "none", 1.0).shape == (8, 8)


@pytest.mark.parametrize("saver", [np.savez, np.savez_compressed])
def test_npy_stream_matches_full_load(tmp_path, saver):
    imgs = np.random.default_rng(1).random((37, 6, 5, 3)) * 1000
    s = extract_npz.NpyStream(_save(tmp_path / "x.npz", imgs, saver))
    assert len(s) == 37 and s.shape == imgs.shape and s.dtype == imgs.dtype
    starts, blocks = zip(*s.chunks(5), strict=True)
    assert list(starts) == list(range(0, 37, 5))
    assert np.array_equal(np.concatenate(blocks), imgs)
    rule, factor, gmax = extract_npz.decide_scaling(s, chunk=5)
    assert rule == "scale_by_max" and gmax == imgs.max() and factor == 255.0 / imgs.max()
