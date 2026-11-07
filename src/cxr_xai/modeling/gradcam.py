"""
Grad-CAM (Gradient-weighted Class Activation Mapping) для моделі на основі Xception.

Алгоритм:
  1) A^k - карти ознак останнього згорткового шару (вихід основи Xception,
     шар block14_sepconv2_act, розмір 10x10x2048 для входу 299x299);
  2) y^c - оцінка класу c на виході моделі;
  3) ваги каналів  α_k^c = (1/Z) Σ_ij ∂y^c / ∂A^k_ij  (усереднення градієнтів);
  4) L^c = ReLU( Σ_k α_k^c A^k ) - теплова карта, нормується до [0, 1]
     і масштабується до розміру зображення.

Модель складається з лінійного ланцюжка шарів, тому прямий прохід розбивається
на частину «до основи», саму основу і «голову» - це дає доступ до виходу
останнього згорткового шару без перебудови вкладеної моделі.
Обмеження: роздільність карти низька (10x10), вона показує області, що вплинули
на оцінку, але не є сегментацією патології.
"""
from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf
from matplotlib import colormaps

from cxr_xai.modeling.builder import get_backbone


def _split_layers(model: keras.Model):
    backbone = get_backbone(model)
    layers_list = [layer for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)]
    idx = layers_list.index(backbone)
    return layers_list[:idx], backbone, layers_list[idx + 1:]


def compute_gradcam(model: keras.Model, image: np.ndarray, class_index: int | None = None):
    """
    image: (H, W, 3) або (1, H, W, 3), значення 0..255 (як на вході моделі).
    Повертає (heatmap (h, w) у [0, 1], вектор оцінок softmax, class_index).
    """
    x = tf.convert_to_tensor(image, dtype=tf.float32)
    if x.shape.rank == 3:
        x = x[tf.newaxis]
    pre, backbone, head = _split_layers(model)
    for layer in pre:
        x = layer(x)
    with tf.GradientTape() as tape:
        conv = backbone(x, training=False)
        tape.watch(conv)
        y = conv
        for layer in head:
            y = layer(y, training=False)
        if class_index is None:
            class_index = int(tf.argmax(y[0]))
        score = y[:, class_index]
    grads = tape.gradient(score, conv)                       # (1, h, w, C)
    weights = tf.reduce_mean(grads, axis=(1, 2))             # (1, C)
    cam = tf.reduce_sum(conv * weights[:, tf.newaxis, tf.newaxis, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32), y.numpy()[0].astype(np.float64), class_index


def resize_heatmap(heatmap: np.ndarray, height: int, width: int) -> np.ndarray:
    hm = tf.image.resize(heatmap[..., np.newaxis], (height, width), method="bilinear").numpy()[..., 0]
    return np.clip(hm, 0.0, 1.0)


def overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4,
            cmap: str = "jet") -> np.ndarray:
    """Накладає теплову карту на зображення (uint8, H x W x 3)."""
    img = np.asarray(image_rgb, dtype=np.float32)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    hm = resize_heatmap(heatmap, img.shape[0], img.shape[1])
    colored = colormaps[cmap](hm)[..., :3] * 255.0
    out = (1.0 - alpha) * img + alpha * colored
    return np.clip(out, 0, 255).astype(np.uint8)


def border_activation_fraction(heatmap: np.ndarray, border: float = 0.10, size: int = 100) -> float:
    """
    Частка «маси» теплової карти в периферійній смузі зображення шириною border.
    Евристика для виявлення «хибних підказок» (shortcut learning): написи, маркери
    і рамки зазвичай розташовані біля країв знімка. Площа смуги 10 % становить 36 %
    від площі зображення, тож значення суттєво більше 0,36 свідчить про зміщення уваги до країв.
    """
    hm = resize_heatmap(heatmap, size, size)
    total = hm.sum()
    if total <= 0:
        return float("nan")
    b = max(1, int(round(border * size)))
    mask = np.ones_like(hm, dtype=bool)
    mask[b:-b, b:-b] = False
    return float(hm[mask].sum() / total)
