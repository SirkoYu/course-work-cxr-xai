"""
Оцінювання моделі: прогнози, метрики, матриці помилок.

Метрики: accuracy, balanced accuracy, macro-/weighted-F1, macro precision/recall,
ROC-AUC (one-vs-rest, macro), повний classification report за класами.
Макро-усереднення - основне, бо класи можуть бути незбалансованими.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cxr_xai import constants
from cxr_xai.settings import settings


def predict_probs(model, df: pd.DataFrame, data_root: Path,
                  batch_size: int = settings.batch_size) -> np.ndarray:
    """Повертає матрицю softmax-оцінок (N, K) у порядку рядків df.
    Розмір входу береться з самої моделі, тож він завжди збігається з розміром під час навчання."""
    from cxr_xai.data import loading
    ds = loading.make_dataset(df, data_root, training=False, batch_size=batch_size,
                         img_size=int(model.input_shape[1]))
    return model.predict(ds.map(lambda x, y: x), verbose=1)


def predictions_frame(df: pd.DataFrame, probs: np.ndarray,
                      class_names=constants.CLASS_NAMES) -> pd.DataFrame:
    order = np.argsort(-probs, axis=1)
    p_sorted = np.take_along_axis(probs, order, axis=1)
    out = df[["path", "label", "label_idx"]].copy().reset_index(drop=True)
    out["pred_idx"] = order[:, 0]
    out["pred_label"] = [class_names[i] for i in order[:, 0]]
    out["second_idx"] = order[:, 1]
    out["p_max"] = p_sorted[:, 0]
    out["p_second"] = p_sorted[:, 1]
    out["margin"] = p_sorted[:, 0] - p_sorted[:, 1]
    out["correct"] = out["pred_idx"] == out["label_idx"]
    for i, c in enumerate(class_names):
        out[f"p_{c}"] = probs[:, i]
    return out


def probs_from_frame(pred_df: pd.DataFrame, class_names=constants.CLASS_NAMES) -> np.ndarray:
    return pred_df[[f"p_{c}" for c in class_names]].to_numpy(dtype=float)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, class_names=constants.CLASS_NAMES) -> dict:
    labels = list(range(len(class_names)))
    y_pred = probs.argmax(axis=1)
    m = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }
    try:
        m["roc_auc_ovr_macro"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                                               labels=labels)
    except ValueError:            # напр., у вибірці представлені не всі класи
        m["roc_auc_ovr_macro"] = float("nan")
    m = {k: (float(v) if not isinstance(v, int) else v) for k, v in m.items()}
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    m["report"] = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                        output_dict=True, zero_division=0)
    m["report_text"] = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                             digits=4, zero_division=0)
    return m


def report_frame(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics["report"]).T.rename_axis("class").round(4)


def summary_row(metrics: dict, **extra) -> dict:
    keys = ["n", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
            "macro_f1", "weighted_f1", "roc_auc_ovr_macro"]
    return {**extra, **{k: metrics[k] for k in keys}}


def plot_confusion(cm, class_names, path: Path, normalize: bool = False, title: str = "") -> None:
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Прогноз моделі")
    ax.set_ylabel("Справжній клас (розмітка датасету)")
    if title:
        ax.set_title(title)
    thresh = shown.max() / 2 if shown.max() > 0 else 0.5
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            text = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if shown[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_all(metrics: dict, name: str, split: str, out_dir: Path, class_names=constants.CLASS_NAMES) -> None:
    """Зберігає JSON метрик, CSV/TXT звіту та рисунки матриці помилок."""
    from cxr_xai import json_io
    tag = f"{name}_{split}"
    json_io.save_json({k: v for k, v in metrics.items() if k != "report_text"},
                     out_dir / f"metrics_{tag}.json")
    report_frame(metrics).to_csv(out_dir / f"classification_report_{tag}.csv")
    (out_dir / f"classification_report_{tag}.txt").write_text(metrics["report_text"],
                                                                         encoding="utf-8")
    pd.DataFrame(metrics["confusion_matrix"], index=class_names, columns=class_names) \
        .to_csv(out_dir / f"confusion_matrix_{tag}.csv")
    plot_confusion(metrics["confusion_matrix"], class_names,
                   out_dir / f"confusion_{tag}.png", normalize=False,
                   title=f"Матриця помилок ({name}, {split})")
    plot_confusion(metrics["confusion_matrix"], class_names,
                   out_dir / f"confusion_{tag}_norm.png", normalize=True,
                   title=f"Нормована матриця помилок ({name}, {split})")
