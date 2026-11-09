"""Попередня обробка й аугментація не повинні давати моделі «підказок», не пов'язаних із хворобою."""
import io

import numpy as np
from PIL import Image

from cxr_xai.data import loading


def test_color_images_become_grayscale():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 200, 120, 60
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    out = np.asarray(loading.decode_and_resize(buf.getvalue(), 32))
    assert np.allclose(out[..., 0], out[..., 1]) and np.allclose(out[..., 1], out[..., 2])


def test_augmentation_does_not_add_black_fill():
    x = np.full((8, 64, 64, 3), 200.0, dtype="float32")
    out = np.asarray(loading.build_augmenter(seed=1)(x, training=True))
    assert out.min() > 100.0
