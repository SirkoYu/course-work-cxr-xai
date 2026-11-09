"""
Конвеєр завантаження даних (tf.data).

Одна й та сама функція decode_and_resize використовується і під час навчання,
і в інтерфейсі Streamlit - це виключає розбіжність попередньої обробки
між навчанням і застосуванням моделі.

Нормалізація пікселів до [-1, 1] (як вимагає Xception) виконується
всередині моделі шаром Rescaling, тому сюди подаються значення 0..255.
"""
from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from cxr_xai.settings import settings

AUTOTUNE = tf.data.AUTOTUNE


def decode_and_resize(raw_bytes, img_size: int = settings.img_size) -> tf.Tensor:
    """
    Байти файлу (PNG/JPEG/BMP) -> тензор (img_size, img_size, 3), float32, 0..255.
    Будь-яке зображення спершу зводиться до градацій сірого, які потім дублюються в 3 канали
    (Xception очікує RGB-вхід). Кольоровими в датасеті є лише частина знімків Tuberculosis
    і COVID-19, тож відтінок інакше став би «підказкою» класу, а не ознакою хвороби.
    antialias=True зменшує артефакти при сильному зменшенні великих знімків.
    """
    img = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    img = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img))
    img = tf.image.resize(img, (img_size, img_size), method="bilinear", antialias=True)
    return tf.clip_by_value(img, 0.0, 255.0)


def load_image(path: tf.Tensor, img_size: int = settings.img_size) -> tf.Tensor:
    return decode_and_resize(tf.io.read_file(path), img_size)


def build_augmenter(seed: int = settings.seed) -> keras.Sequential:
    """
    Аугментація ЛИШЕ для навчальної вибірки: невеликі повороти, масштаб,
    зсув і контраст, що імітують відмінності укладання пацієнта та апаратури.
    Віддзеркалення не застосовується (анатомічна асиметрія).

    Порожні ділянки після повороту/зсуву/масштабу заповнюються відбиттям (reflect), а НЕ
    чорним кольором: чорні кути з'являлися лише на навчальних зображеннях, а частка чорних
    країв на знімках різних класів різна. Модель з чорним заповненням на чистих знімках
    систематично завищувала Tuberculosis і Viral Pneumonia (класи з найменшою часткою
    чорних країв): accuracy на валідації 0.71 без аугментації проти 0.80 з нею.
    """
    return keras.Sequential([
        keras.layers.RandomRotation(settings.aug_rotation, fill_mode="reflect", seed=seed),
        keras.layers.RandomZoom(settings.aug_zoom, fill_mode="reflect", seed=seed + 1),
        keras.layers.RandomTranslation(settings.aug_translation, settings.aug_translation,
                                       fill_mode="reflect", seed=seed + 2),
        keras.layers.RandomContrast(settings.aug_contrast, seed=seed + 3),
    ], name="augmentation")


def read_split(split_csv: Path, split: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def make_dataset(df: pd.DataFrame, data_root: Path, training: bool,
                 batch_size: int = settings.batch_size, img_size: int = settings.img_size,
                 seed: int = settings.seed, repeat: bool = False) -> tf.data.Dataset:
    """
    Створює tf.data.Dataset з пар (зображення, індекс класу).
    training=True: перемішування + аугментація; інакше порядок рядків df зберігається
    (потрібно, щоб зіставити прогнози з файлами).
    repeat=True: нескінченний потік (з новим перемішуванням на кожному проході) - для
    model.fit зі steps_per_epoch, коротшим або не кратним повній епосі.
    """
    paths = [str(Path(data_root) / p) for p in df["path"]]
    labels = df["label_idx"].astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.map(lambda p, y: (load_image(p, img_size), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if training:
        augmenter = build_augmenter(seed)
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 255.0), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def compute_class_weights(train_df: pd.DataFrame, num_classes: int) -> dict[int, float]:
    """w_c = N / (K * N_c) - менш представлені класи отримують більшу вагу у функції втрат."""
    counts = np.bincount(train_df["label_idx"].to_numpy(), minlength=num_classes)
    n, k = counts.sum(), num_classes
    return {i: float(n / (k * c)) if c > 0 else 0.0 for i, c in enumerate(counts)}
