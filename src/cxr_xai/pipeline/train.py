"""
Крок 4. Навчання моделі: перенесення знань (етап 1) + fine-tuning (етап 2).

Етап 1: основа Xception заморожена, навчається лише «голова» (Dense + Dropout),
        Adam з lr = settings.stage1_lr.
Етап 2: розморожуються блоки Xception з номера settings.fine_tune_from_block,
        Adam з меншою lr = settings.stage2_lr; навчання стартує з найкращих ваг етапу 1.

Колбеки: EarlyStopping (val_loss), ReduceLROnPlateau (val_loss),
ModelCheckpoint (зберігається модель з найменшою val_loss).
Для вибору моделі використовується лише валідаційна вибірка; тестова - не торкається.

Результати:
  outputs/models/stage1_best.keras, outputs/models/final.keras,
  outputs/models/class_names.json, outputs/models/training_meta.json,
  outputs/reports/history_stage{1,2}.csv, outputs/figures/training_curves.png

Запуск:
    cxr-xai train
    cxr-xai train --smoke 16 --epochs1 1 --epochs2 1 --weights none   # швидка перевірка коду
"""
from __future__ import annotations

import argparse
import sys
import time

import matplotlib

matplotlib.use("Agg")
import keras
import matplotlib.pyplot as plt
import pandas as pd

from cxr_xai import constants, json_io, runtime
from cxr_xai.data import loading
from cxr_xai.modeling import builder
from cxr_xai.settings import settings


class EpochTimer(keras.callbacks.Callback):
    """Фіксує тривалість кожної епохи (для таблиці часу навчання в курсовій)."""

    def __init__(self):
        super().__init__()
        self.times: list[float] = []
        self._t0 = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._t0)


def make_callbacks(checkpoint_path) -> tuple[list, EpochTimer]:
    timer = EpochTimer()
    callbacks = [
        timer,
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=settings.early_stop_patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=settings.reduce_lr_factor,
                                          patience=settings.reduce_lr_patience,
                                          min_lr=settings.min_lr, verbose=1),
    ]
    return callbacks, timer


def history_to_df(history, timer: EpochTimer, stage: int) -> pd.DataFrame:
    h = pd.DataFrame(history.history)
    h.insert(0, "epoch", range(1, len(h) + 1))
    h.insert(0, "stage", stage)
    h["epoch_time_s"] = timer.times[:len(h)]
    return h


def plot_curves(h1: pd.DataFrame, h2: pd.DataFrame, path) -> None:
    h = pd.concat([h1, h2], ignore_index=True)
    h["global_epoch"] = range(1, len(h) + 1)
    boundary = len(h1) + 0.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric, title in ((axes[0], "loss", "Функція втрат"), (axes[1], "accuracy", "Accuracy")):
        ax.plot(h["global_epoch"], h[metric], marker="o", ms=3, label="train")
        ax.plot(h["global_epoch"], h[f"val_{metric}"], marker="o", ms=3, label="val")
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=1)
        ax.text(boundary, ax.get_ylim()[1], " fine-tuning →", va="top", fontsize=8, color="gray")
        ax.set_xlabel("Епоха")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Навчання моделі Xception (TL + fine-tuning)")
    parser.add_argument("--smoke", type=int, default=0,
                        help="взяти лише N зображень кожного класу в кожній вибірці (перевірка коду)")
    args = parser.parse_args(argv)
    runtime.configure_tensorflow()

    runtime.ensure_dirs()
    runtime.set_global_seed(settings.seed)
    if not settings.split_csv.exists():
        sys.exit("[ПОМИЛКА] Немає split.csv - спочатку виконайте cxr-xai split")

    data_root = settings.data_root
    df = loading.read_split()
    # У режимі --smoke результати пишуться в окрему папку outputs/smoke і не зачіпають основні.
    base = settings.output_dir / "smoke" if args.smoke else None
    out = {
        "stage1": base / "stage1_best.keras" if base else settings.stage1_model,
        "final": base / "final.keras" if base else settings.final_model,
        "class_names": base / "class_names.json" if base else settings.class_names_json,
        "meta": base / "training_meta.json" if base else settings.train_meta_json,
        "dir_reports": base or settings.reports_dir,
        "dir_figures": base or settings.figures_dir,
    }
    if base:
        base.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        df = df.groupby(["split", "label"], group_keys=False).head(args.smoke).reset_index(drop=True)
        print(f"[SMOKE] Використовується лише {len(df)} зображень - результати не для курсової!")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    num_classes = len(constants.CLASS_NAMES)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    ds_train = loading.make_dataset(train_df, data_root, training=True, batch_size=settings.batch_size)
    ds_val = loading.make_dataset(val_df, data_root, training=False, batch_size=settings.batch_size)
    class_weight = loading.compute_class_weights(train_df, num_classes) if settings.use_class_weights else None
    print(f"Ваги класів: {class_weight}")

    # ---------------- Етап 1: перенесення знань (основа заморожена) ----------------
    model, _ = builder.build_model(num_classes, settings.img_size, settings.backbone_weights, settings.dropout)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=settings.stage1_lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    params1 = builder.count_params(model)
    print(f"Етап 1: параметри {params1}")
    callbacks, timer1 = make_callbacks(out['stage1'])
    t0 = time.perf_counter()
    hist1 = model.fit(ds_train, validation_data=ds_val, epochs=settings.stage1_epochs,
                      class_weight=class_weight, callbacks=callbacks, verbose=2)
    time1 = time.perf_counter() - t0
    h1 = history_to_df(hist1, timer1, stage=1)
    h1.to_csv(out['dir_reports'] / "history_stage1.csv", index=False)

    # ---------------- Етап 2: fine-tuning ----------------
    # Стартуємо з найкращої (за val_loss) моделі етапу 1, збереженої ModelCheckpoint.
    model = keras.models.load_model(out['stage1'])
    unfreeze_info = builder.unfreeze_from_block(model, settings.fine_tune_from_block)
    # Після зміни trainable модель обов'язково перекомпільовується.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=settings.stage2_lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    params2 = builder.count_params(model)
    print(f"Етап 2: розморожено {len(unfreeze_info['unfrozen_layers_with_weights'])} шарів з вагами; "
          f"параметри {params2}")
    callbacks, timer2 = make_callbacks(out['final'])
    t0 = time.perf_counter()
    hist2 = model.fit(ds_train, validation_data=ds_val, epochs=settings.stage2_epochs,
                      class_weight=class_weight, callbacks=callbacks, verbose=2)
    time2 = time.perf_counter() - t0
    h2 = history_to_df(hist2, timer2, stage=2)
    h2.to_csv(out['dir_reports'] / "history_stage2.csv", index=False)

    plot_curves(h1, h2, out['dir_figures'] / "training_curves.png")
    json_io.save_json(constants.CLASS_NAMES, out['class_names'])

    meta = {
        "created": runtime.now_str(),
        "smoke_mode": bool(args.smoke),
        "class_names": constants.CLASS_NAMES,
        "n_train": len(train_df), "n_val": len(val_df),
        "img_size": settings.img_size, "batch_size": settings.batch_size,
        "backbone": "Xception", "backbone_weights": settings.backbone_weights,
        "dropout": settings.dropout, "class_weight": class_weight,
        "augmentation": {"rotation": settings.aug_rotation, "zoom": settings.aug_zoom,
                         "translation": settings.aug_translation, "contrast": settings.aug_contrast},
        "stage1": {"lr": settings.stage1_lr, "max_epochs": settings.stage1_epochs, "epochs_run": len(h1),
                   "best_epoch": int(h1["val_loss"].idxmin()) + 1,
                   "best_val_loss": float(h1["val_loss"].min()),
                   "val_accuracy_at_best": float(h1.loc[h1["val_loss"].idxmin(), "val_accuracy"]),
                   "time_s": round(time1, 1), "params": params1},
        "stage2": {"lr": settings.stage2_lr, "max_epochs": settings.stage2_epochs, "epochs_run": len(h2),
                   "fine_tune_from_block": settings.fine_tune_from_block,
                   "unfrozen_layers": unfreeze_info["unfrozen_layers_with_weights"],
                   "best_epoch": int(h2["val_loss"].idxmin()) + 1,
                   "best_val_loss": float(h2["val_loss"].min()),
                   "val_accuracy_at_best": float(h2.loc[h2["val_loss"].idxmin(), "val_accuracy"]),
                   "time_s": round(time2, 1), "params": params2},
        "callbacks": {"early_stopping_patience": settings.early_stop_patience,
                      "reduce_lr_patience": settings.reduce_lr_patience,
                      "reduce_lr_factor": settings.reduce_lr_factor, "min_lr": settings.min_lr,
                      "monitor": "val_loss"},
        "seed": settings.seed,
        "environment": runtime.environment_info(),
    }
    json_io.save_json(meta, out['meta'])
    if meta["stage2"]["best_val_loss"] > meta["stage1"]["best_val_loss"]:
        print("[УВАГА] Fine-tuning не зменшив val_loss порівняно з етапом 1. "
              "Це теж результат - зафіксуйте його і порівняйте моделі в cxr-xai evaluate.")
    print(f"\nНавчання завершено. Етап 1: {time1:.0f} с, етап 2: {time2:.0f} с.")
    print(f"Фінальна модель: {out['final']}")
    print(f"Метадані: {out['meta']}")


if __name__ == "__main__":
    main()
