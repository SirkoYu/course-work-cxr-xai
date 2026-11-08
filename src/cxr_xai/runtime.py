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


def preload_cuda_libraries() -> None:
    """
    Linux/WSL: попередньо завантажує бібліотеки CUDA з pip-пакетів nvidia-* (extra «gpu»).
    TensorFlow 2.21 шукає їх за власним RUNPATH, де немає, наприклад, nvidia/cusolver/lib,
    тому без цього GPU відключається («Cannot dlopen some GPU libraries»). Уже завантажену
    бібліотеку dlopen знаходить за її soname, тож LD_LIBRARY_PATH задавати не потрібно.
    Має викликатися до першого звернення TensorFlow до GPU. Без пакетів nvidia - нічого не робить.
    """
    if sys.platform != "linux":
        return
    import ctypes
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    for root in spec.submodule_search_locations:
        for lib in sorted(Path(root).glob("*/lib/lib*.so*")):
            if lib.name.startswith("libnvblas"):   # перехоплює BLAS-виклики, TensorFlow не потрібна
                continue
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass   # не критично: TensorFlow спробує знайти бібліотеку сам


def configure_tensorflow() -> str:
    """
    Вмикає поступове виділення пам'яті GPU і (за потреби) mixed precision.
    Якщо GPU недоступний, TensorFlow працює на CPU - функція лише явно про це повідомляє.
    Повертає "GPU" або "CPU".
    """
    preload_cuda_libraries()
    import keras
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if gpus:
        names = []
        for gpu in gpus:
            try:
                names.append(tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name))
            except Exception:  # noqa: BLE001 - назва пристрою не критична
                names.append(gpu.name)
        print(f"Пристрій: GPU ({', '.join(names)})")
    else:
        print("[УВАГА] GPU недоступний - обчислення виконуються на CPU (значно повільніше). "
              "Для GPU запускайте в WSL2/Linux: uv sync --extra gpu")
    if settings.mixed_precision:
        if gpus:
            keras.mixed_precision.set_global_policy("mixed_float16")
        else:
            print("[УВАГА] mixed_precision вимкнено: на CPU float16 лише сповільнює обчислення")
    return "GPU" if gpus else "CPU"
