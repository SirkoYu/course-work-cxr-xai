"""
Крок 5. Оцінювання моделей на ВАЛІДАЦІЙНІЙ вибірці (до і після fine-tuning).

Валідаційна вибірка використовується для всіх рішень, що стосуються налаштування
системи (вибір моделі, пороги T і δ, типові плутанини для бази знань).
Тестова вибірка тут НЕ використовується - вона оцінюється один раз у cxr-xai report.

Результати:
  outputs/data/predictions_{stage1,final}_val.csv
  outputs/reports/metrics_*_val.json, classification_report_*_val.{csv,txt},
  outputs/reports/metrics_val_stage1_vs_final.csv
  outputs/figures/confusion_*_val.png, confusion_*_val_norm.png

Запуск:
    cxr-xai evaluate
"""
from __future__ import annotations

import argparse
import sys

import keras
import pandas as pd

from cxr_xai import runtime
from cxr_xai.data import loading
from cxr_xai.modeling import evaluation
from cxr_xai.settings import settings


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="Метрики на валідаційній вибірці").parse_args(argv)
    runtime.configure_tensorflow()
    runtime.ensure_dirs()
    if not settings.final_model.exists():
        sys.exit("[ПОМИЛКА] Немає моделі - спочатку виконайте cxr-xai train")
    data_root = settings.data_root
    val_df = loading.read_split("val")
    print(f"Валідаційна вибірка: {len(val_df)} зображень")

    rows = []
    for name, path in (("stage1", settings.stage1_model), ("final", settings.final_model)):
        if not path.exists():
            print(f"[УВАГА] {path} не знайдено - пропуск")
            continue
        print(f"\n=== Модель: {name} ({path.name}) ===")
        model = keras.models.load_model(path, compile=False)
        probs = evaluation.predict_probs(model, val_df, data_root)
        pred_df = evaluation.predictions_frame(val_df, probs)
        pred_df.to_csv(settings.data_dir / f"predictions_{name}_val.csv", index=False)
        metrics = evaluation.compute_metrics(val_df["label_idx"].to_numpy(), probs)
        evaluation.save_all(metrics, name, "val")
        print(metrics["report_text"])
        rows.append(evaluation.summary_row(metrics, model=name, split="val"))

    table = pd.DataFrame(rows).round(4)
    table.to_csv(settings.reports_dir / "metrics_val_stage1_vs_final.csv", index=False)
    print("\nПорівняння (валідаційна вибірка):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
