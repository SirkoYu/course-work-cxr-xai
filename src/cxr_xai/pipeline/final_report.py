"""
Крок 7. ОДНОРАЗОВЕ фінальне оцінювання на ТЕСТОВІЙ вибірці.

Виконується лише після того, як усі рішення (модель, T, δ, база знань)
прийнято за валідаційною вибіркою. Повторне використання тестової вибірки
для налаштування системи є методологічною помилкою - скрипт попереджає про це.

Результати:
  outputs/reports/metrics_test_stage1_vs_final.csv   - метрики до/після fine-tuning
  outputs/reports/classification_report_*_test.{csv,txt}, confusion_matrix_*_test.csv
  outputs/figures/confusion_*_test.png, confusion_*_test_norm.png
  outputs/reports/rules_test_stats.csv               - статистика вирішувача з фіксованими T, δ
  outputs/data/predictions_*_test.csv, decisions_final_test.csv

Запуск:
    cxr-xai report
"""
from __future__ import annotations

import argparse
import sys

import keras
import pandas as pd

from cxr_xai import json_io, runtime
from cxr_xai.data import loading
from cxr_xai.knowledge import rule_tuning
from cxr_xai.knowledge.inference_engine import InferenceEngine
from cxr_xai.knowledge.knowledge_base import KnowledgeBase
from cxr_xai.modeling import evaluation
from cxr_xai.settings import settings

FLAG = settings.reports_dir / "TEST_EVALUATED.txt"


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="Фінальне оцінювання на тестовій вибірці").parse_args(argv)
    runtime.configure_tensorflow()
    runtime.ensure_dirs()
    kb = KnowledgeBase.load(settings.kb_path)
    if not kb.decision_params()["tuned"]:
        sys.exit("[ПОМИЛКА] T і δ ще не підібрано - спочатку виконайте cxr-xai tune")
    if FLAG.exists():
        print("[УВАГА] Тестову вибірку вже оцінювали:\n" + FLAG.read_text(encoding="utf-8"))
        print("Якщо після цього ви змінювали модель чи параметри, зазначте це в курсовій.\n")

    data_root = settings.data_root
    test_df = loading.read_split("test")
    print(f"Тестова вибірка: {len(test_df)} зображень")

    rows, final_probs = [], None
    for name, path in (("stage1", settings.stage1_model), ("final", settings.final_model)):
        if not path.exists():
            print(f"[УВАГА] {path} не знайдено - пропуск")
            continue
        print(f"\n=== Модель: {name} ===")
        model = keras.models.load_model(path, compile=False)
        probs = evaluation.predict_probs(model, test_df, data_root)
        predictions = evaluation.predictions_frame(test_df, probs)
        predictions.to_csv(settings.data_dir / f"predictions_{name}_test.csv", index=False)
        metrics = evaluation.compute_metrics(test_df["label_idx"].to_numpy(), probs)
        evaluation.save_all(metrics, name, "test")
        print(metrics["report_text"])
        rows.append(evaluation.summary_row(metrics, model=name, split="test"))
        if name == "final":
            final_probs = probs

    table = pd.DataFrame(rows).round(4)
    table.to_csv(settings.reports_dir / "metrics_test_stage1_vs_final.csv", index=False)
    print("\nМетрики на тестовій вибірці (до / після fine-tuning):")
    print(table.to_string(index=False))

    if final_probs is not None:
        class_names = json_io.load_json(settings.class_names_json)
        errors = kb.validate(class_names=class_names, num_outputs=final_probs.shape[1])
        if errors:
            sys.exit("[ПОМИЛКА] БЗ не узгоджена з моделлю:\n- " + "\n- ".join(errors))
        engine = InferenceEngine(kb)
        summary, per_sample = rule_tuning.engine_statistics(engine, final_probs, test_df["label"].tolist())
        per_sample.insert(0, "path", test_df["path"])
        per_sample.to_csv(settings.data_dir / "decisions_final_test.csv", index=False)
        pd.DataFrame([summary]).round(4).to_csv(settings.reports_dir / "rules_test_stats.csv", index=False)
        print("\nСтатистика вирішувача на тестовій вибірці (T, δ зафіксовано за val):")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    FLAG.write_text(f"Дата оцінювання: {runtime.now_str()}\nМодель: {settings.final_model.name}\n"
                    f"T = {kb.decision_params()['T']}, δ = {kb.decision_params()['delta']}\n",
                    encoding="utf-8")


if __name__ == "__main__":
    main()
