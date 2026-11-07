"""
Середовище виконання: seed, відомості про версії, налаштування TensorFlow.
Модуль не імпортує TensorFlow під час завантаження, тож придатний для кроків без GPU.
"""
from __future__ import annotations

import os
import platform
import random
import sys
from datetime import datetime
from importlib import metadata

import numpy as np

from cxr_xai.settings import settings


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_global_seed(seed: int = settings.seed) -> None:
    """Фіксує seed для random, numpy і (якщо імпортовано) Keras/TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - глобальний seed потрібен саме для застарілого API
    if "keras" in sys.modules or "tensorflow" in sys.modules:
        import keras
        keras.utils.set_random_seed(seed)


def environment_info() -> dict:
    """Версії Python і бібліотек - для розділу про середовище та відтворюваність."""
    packages = ["tensorflow", "tensorflow-cpu", "keras", "numpy", "pandas", "scikit-learn",
                "pillow", "imagehash", "matplotlib", "streamlit"]
    versions = {}
    for p in packages:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }
    if "tensorflow" in sys.modules:
        import tensorflow as tf
        info["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
        try:
            info["gpu_details"] = [tf.config.experimental.get_device_details(d)
                                   for d in tf.config.list_physical_devices("GPU")]
        except Exception:  # noqa: BLE001 - деталі GPU не критичні
            pass
    return info


def configure_tensorflow() -> None:
    """Вмикає поступове виділення пам'яті GPU і (за потреби) mixed precision."""
    import keras
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if settings.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
