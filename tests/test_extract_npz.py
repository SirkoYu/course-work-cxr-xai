"""Допоміжні функції розпакування NPZ."""
import numpy as np

from cxr_xai.pipeline import extract_npz


def test_npz_helpers():
    m = extract_npz
    assert m.name_prefix("COVID-123.png") == "COVID"
    assert m.name_prefix("CHNCXR_0001_0.png") == "CHNCXR"
    assert m.name_prefix("Viral Pneumonia-12") == "Viral Pneumonia"
    assert m.safe_name("a/b\\c?.png") == "a_b_c_"
    imgs = np.random.default_rng(0).random((4, 8, 8, 1)).astype(np.float32)
    rule, f = m.decide_scaling(imgs)
    assert rule == "x255"
    out = m.to_uint8(imgs[0], rule, f)
    assert out.shape == (8, 8) and out.dtype == np.uint8
    rgb = np.repeat((imgs[0] * 255).astype(np.uint8), 3, axis=-1)
    assert m.to_uint8(rgb, "none", 1.0).shape == (8, 8)
