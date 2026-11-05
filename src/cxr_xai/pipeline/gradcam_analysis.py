"""
Крок 8. Аналіз хибних класифікацій із теплими картами Grad-CAM.

Що робить:
  1) таблиця пар помилок «справжній -> прогноз» (найчастіші плутанини);
  2) рисунки Grad-CAM для найупевненіших помилок і для прикладів правильних
     класифікацій кожного класу;
  3) евристика зміщення уваги: частка активації Grad-CAM у периферійній смузі
     знімка (написи, маркери, рамки) - окремо для помилок і правильних відповідей.

Запуск (після cxr-xai report; можна й для val):
    cxr-xai gradcam --split test --n-errors 12 --n-correct 2
"""
from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from cxr_xai import runtime
from cxr_xai.data import loading
from cxr_xai.modeling import gradcam
from cxr_xai.settings import settings


def load_display_image(path, size: int) -> np.ndarray:
    raw = tf.io.read_file(str(path))
    return loading.decode_and_resize(raw, size).numpy()


def grid_figure(items, path, title, ncols: int = 4) -> None:
    """items: список (оригінал, накладання, підпис)."""
    if not items:
        return
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(3 * ncols, 6.2 * nrows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (orig, ov, caption) in enumerate(items):
        r, c = (k // ncols) * 2, k % ncols
        axes[r, c].imshow(orig.astype(np.uint8))
        axes[r, c].set_title(caption, fontsize=7)
        axes[r + 1, c].imshow(ov)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Аналіз хибних класифікацій з Grad-CAM")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--n-errors", type=int, default=12)
    parser.add_argument("--n-correct", type=int, default=2, help="правильних прикладів на клас")
    args = parser.parse_args(argv)
    runtime.configure_tensorflow()

    runtime.ensure_dirs()
    pred_path = settings.data_dir / f"predictions_final_{args.split}.csv"
    if not pred_path.exists():
        script = "cxr-xai evaluate" if args.split == "val" else "cxr-xai report"
        sys.exit(f"[ПОМИЛКА] Немає {pred_path.name} - спочатку виконайте {script}")
    pred = pd.read_csv(pred_path)
    data_root = settings.data_root
    model = keras.models.load_model(settings.final_model, compile=False)
    size = int(model.input_shape[1])          # розмір входу, з яким навчено модель

    # --- 1. Пари помилок ---
    errors = pred[~pred["correct"]].copy()
    pairs = (errors.groupby(["label", "pred_label"]).size().rename("n").reset_index()
             .sort_values("n", ascending=False))
    pairs["share_of_errors"] = (pairs["n"] / max(len(errors), 1)).round(4)
    pairs.to_csv(settings.reports_dir / f"error_pairs_{args.split}.csv", index=False)
    print(f"Помилок: {len(errors)} з {len(pred)}")
    print(pairs.head(10).to_string(index=False))

    # --- 2. Вибір прикладів ---
    err_sel = errors.sort_values("p_max", ascending=False).head(args.n_errors)
    correct = pred[pred["correct"]]
    cor_sel = (correct.sort_values("p_max", ascending=False)
               .groupby("label", group_keys=False).head(args.n_correct))

    def run(row):
        img = load_display_image(data_root / row["path"], size)
        cam, _, _ = gradcam.compute_gradcam(model, img, int(row["pred_idx"]))
        ov = gradcam.overlay(img, cam, settings.gradcam_alpha)
        return img, cam, ov

    out_dir = settings.figures_dir / f"gradcam_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    err_items, cor_items, stats = [], [], []
    for kind, sel, items in (("error", err_sel, err_items), ("correct", cor_sel, cor_items)):
        for k, (_, row) in enumerate(sel.iterrows(), start=1):
            img, cam, ov = run(row)
            caption = f"справжній: {row['label']}\nпрогноз: {row['pred_label']} ({row['p_max']:.2f})"
            items.append((img, ov, caption))
            name = f"{kind}_{k:02d}_{row['label']}_as_{row['pred_label']}.png"
            plt.imsave(out_dir / name.replace(" ", "_"), ov)

    grid_figure(err_items, settings.figures_dir / f"gradcam_errors_{args.split}.png",
                "Grad-CAM: найупевненіші хибні класифікації (верх - знімок, низ - теплова карта)")
    grid_figure(cor_items, settings.figures_dir / f"gradcam_correct_{args.split}.png",
                "Grad-CAM: приклади правильних класифікацій")

    # --- 3. Евристика зміщення уваги до країв ---
    rng = np.random.default_rng(settings.seed)
    cor_sample = correct.iloc[rng.permutation(len(correct))[:settings.gradcam_stats_max_correct]]
    for kind, sel in (("error", errors), ("correct", cor_sample)):
        for _, row in sel.iterrows():
            _, cam, _ = run(row)
            stats.append({"kind": kind, "label": row["label"], "pred_label": row["pred_label"],
                          "p_max": row["p_max"],
                          "border_fraction": gradcam.border_activation_fraction(cam, settings.border_fraction)})
    st = pd.DataFrame(stats)
    st.to_csv(settings.data_dir / f"gradcam_border_{args.split}.csv", index=False)
    summary = st.groupby(["kind", "label"])["border_fraction"].agg(["count", "mean", "median"]).round(3)
    summary.to_csv(settings.reports_dir / f"gradcam_border_summary_{args.split}.csv")
    overall = st.groupby("kind")["border_fraction"].agg(["count", "mean", "median"]).round(3)

    expected = 1 - (1 - 2 * settings.border_fraction) ** 2
    report = "\n".join([
        f"АНАЛІЗ ХИБНИХ КЛАСИФІКАЦІЙ ({args.split})",
        f"Дата: {runtime.now_str()}",
        f"Помилок: {len(errors)} з {len(pred)}",
        "",
        "Найчастіші пари «справжній -> прогноз»:",
        pairs.head(10).to_string(index=False),
        "",
        f"Частка активації Grad-CAM у периферійній смузі {settings.border_fraction:.0%} "
        f"(частка площі смуги = {expected:.2f}):",
        overall.to_string(),
        "",
        "За класами:",
        summary.to_string(),
        "",
        "Інтерпретація: значення, суттєво більші за частку площі смуги, вказують на те, що модель",
        "спирається на ділянки біля країв (написи, маркери, рамки), а не на легеневі поля.",
        "Це евристика; висновки слід підкріпити візуальним переглядом рисунків.",
    ])
    (settings.reports_dir / f"gradcam_report_{args.split}.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
