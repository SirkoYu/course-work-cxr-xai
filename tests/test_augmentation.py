"""Аугментація не повинна додавати штучних чорних ділянок, яких немає на валідації/тесті."""
import numpy as np

from cxr_xai.data import loading


def test_augmentation_does_not_add_black_fill():
    x = np.full((8, 64, 64, 3), 200.0, dtype="float32")
    out = np.asarray(loading.build_augmenter(seed=1)(x, training=True))
    assert out.min() > 100.0
