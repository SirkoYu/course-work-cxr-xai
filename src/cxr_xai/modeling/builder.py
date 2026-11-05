"""
Побудова моделі на основі Xception (перенесення знань з ImageNet)
та розморожування верхніх блоків для fine-tuning.

Архітектура (лінійний ланцюжок шарів):
    image (0..255) -> Rescaling [-1, 1] -> Xception (без верхніх шарів)
    -> GlobalAveragePooling2D -> Dropout -> Dense(K, softmax)
"""
from __future__ import annotations

import keras
import numpy as np
from keras import layers

from cxr_xai import constants
from cxr_xai.settings import settings


def build_model(num_classes: int = len(constants.CLASS_NAMES), img_size: int = settings.img_size,
                weights: str | None = settings.backbone_weights, dropout: float = settings.dropout):
    """Повертає (model, backbone). Основа (backbone) заморожена - етап 1."""
    w = None if weights in (None, "none", "None", "") else weights
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    # Xception preprocess_input: x / 127.5 - 1. Реалізовано шаром, щоб нормалізація
    # зберігалася разом із моделлю і не могла бути пропущена в інтерфейсі.
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)
    backbone = keras.applications.Xception(include_top=False, weights=w,
                                           input_shape=(img_size, img_size, 3))
    backbone.trainable = False
    # training=False: шари BatchNormalization основи завжди працюють у режимі виведення
    # (використовують накопичені статистики), зокрема й під час fine-tuning.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    model = keras.Model(inputs, outputs, name="cxr_xception")
    return model, backbone


def get_backbone(model: keras.Model) -> keras.Model:
    """Знаходить вкладену модель-основу (Xception) усередині моделі."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("У моделі не знайдено вкладеної основи (Xception)")


def unfreeze_from_block(model: keras.Model, block: int = settings.fine_tune_from_block) -> dict:
    """
    Розморожує шари Xception, що йдуть після блоку (block-1), тобто блоки block..14
    разом із їхніми залишковими (residual) згортками. Шари BatchNormalization
    лишаються замороженими - це стабілізує fine-tuning на невеликому наборі даних.
    Повертає відомості для звіту.
    """
    backbone = get_backbone(model)
    backbone.trainable = True
    names = [layer.name for layer in backbone.layers]
    prev = [i for i, n in enumerate(names) if n.startswith(f"block{block - 1}_")]
    if not prev:
        raise ValueError(f"У Xception не знайдено шарів блоку {block - 1}")
    start = max(prev) + 1
    unfrozen = []
    for i, layer in enumerate(backbone.layers):
        trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = trainable
        if trainable and layer.weights:
            unfrozen.append(layer.name)
    return {"first_unfrozen_index": start, "unfrozen_layers_with_weights": unfrozen}


def count_params(model: keras.Model) -> dict:
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(sum(np.prod(w.shape) for w in model.weights))
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}
