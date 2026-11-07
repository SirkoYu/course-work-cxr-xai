"""
Крок 6. Добір параметрів вирішувача T і δ та формування типових плутанин
для бази знань за ВАЛІДАЦІЙНОЮ вибіркою (модель final).

Що робить:
  1) бере прогнози validation/predictions_final_val.csv (створює cxr-xai evaluate);
  2) для кожного значення сітки рахує статистику правил R2 (p1 < T) і R3 (margin < δ),
     обирає T і δ за критерієм settings.threshold_criterion;
  3) за матрицею помилок на val формує слот typical_confusions кожного фрейму-класу;
  4) записує БЗ із результатами в каталог запуску (шаблон knowledge_base.json не змінюється);
  5) перевіряє узгодженість БЗ з моделлю і рахує статистику вирішувача на val.

Запуск:
    cxr-xai tune
"""
from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cxr_xai import constants, json_io, results, runtime
from cxr_xai.knowledge import rule_tuning
from cxr_xai.knowledge.inference_engine import InferenceEngine
from cxr_xai.knowledge.knowledge_base import DECISION_FRAME, KnowledgeBase
from cxr_xai.modeling.evaluation import probs_from_frame
from cxr_xai.settings import settings


def plot_tuning(t_tab, d_tab, t_best, d_best, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, tab, best, sym, feat in ((axes[0], t_tab, t_best, "T", "p1"),
                                     (axes[1], d_tab, d_best, "δ", "p1 − p2")):
        ax.plot(tab["threshold"], tab["errors_captured"], label="частка вловлених помилок (TPR)")
        ax.plot(tab["threshold"], tab["correct_flagged"], label="частка позначених правильних (FPR)")
        ax.plot(tab["threshold"], tab["youden_j"], label="J = TPR − FPR", linewidth=2)
        ax.plot(tab["threshold"], tab["flag_rate"], linestyle=":", label="частка позначених")
        ax.axvline(best, color="#C44E52", linestyle="--", label=f"обрано {sym} = {best:.2f}")
        ax.set_xlabel(f"Поріг {sym} (правило: {feat} < {sym})")
        ax.set_ylim(min(-0.05, float(tab["youden_j"].min()) - 0.05), 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_title("Добір T (правило R2)")
    axes[1].set_title("Добір δ (правило R3)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_risk_coverage(t_tab, t_best, path) -> None:
    """Точність серед «впевнених» результатів залежно від частки охоплених зображень."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(t_tab["coverage"], t_tab["accuracy_unflagged"], marker=".", ms=3)
    sel = t_tab[t_tab["threshold"] == t_best].iloc[0]
    ax.scatter([sel["coverage"]], [sel["accuracy_unflagged"]], color="#C44E52", zorder=3,
               label=f"T = {t_best:.2f}")
    ax.set_xlabel("Охоплення (частка зображень з p1 ≥ T)")
    ax.set_ylabel("Accuracy серед охоплених")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="Добір порогів T і δ, типові плутанини -> база знань").parse_args(argv)
    run = results.require_run()
    pred_path = run.validation_dir / "predictions_final_val.csv"
    if not pred_path.exists():
        sys.exit("[ПОМИЛКА] Немає predictions_final_val.csv - спочатку виконайте cxr-xai evaluate")
    pred = pd.read_csv(pred_path)
    probs = probs_from_frame(pred)
    correct = pred["correct"].to_numpy(dtype=bool)
    print(f"Валідаційна вибірка: {len(pred)} зображень, помилок моделі: {int((~correct).sum())}")

    # --- 1. Добір T і δ ---
    t_tab = rule_tuning.threshold_table(pred["p_max"].to_numpy(), correct, rule_tuning.grid(settings.t_grid))
    d_tab = rule_tuning.threshold_table(pred["margin"].to_numpy(), correct, rule_tuning.grid(settings.delta_grid))
    t_best, t_why = rule_tuning.choose_threshold(t_tab, settings.threshold_criterion, settings.max_flag_rate,
                                        settings.target_accuracy)
    d_best, d_why = rule_tuning.choose_threshold(d_tab, settings.threshold_criterion, settings.max_flag_rate,
                                        settings.target_accuracy)
    t_tab.round(4).to_csv(run.tuning_dir / "tuning_T_val.csv", index=False)
    d_tab.round(4).to_csv(run.tuning_dir / "tuning_delta_val.csv", index=False)
    plot_tuning(t_tab, d_tab, t_best, d_best, run.tuning_dir / "threshold_tuning_val.png")
    plot_risk_coverage(t_tab, t_best, run.tuning_dir / "risk_coverage_val.png")

    # --- 2. Типові плутанини за матрицею помилок на val ---
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(pred["label_idx"], pred["pred_idx"], labels=list(range(len(constants.CLASS_NAMES))))
    confusions = rule_tuning.typical_confusions(cm, constants.CLASS_NAMES, settings.confusion_min_rate)

    # --- 3. Запис у базу знань ---
    kb = KnowledgeBase.load(settings.kb_path)
    kb.set_slot(DECISION_FRAME, "T", t_best)
    kb.set_slot(DECISION_FRAME, "delta", d_best)
    kb.set_slot(DECISION_FRAME, "tuning", {
        "source": "validation", "n_samples": int(len(pred)), "n_errors": int((~correct).sum()),
        "criterion": settings.threshold_criterion, "max_flag_rate": settings.max_flag_rate,
        "T_reason": t_why, "delta_reason": d_why, "confusion_min_rate": settings.confusion_min_rate,
        "model_file": run.final_model.name, "date": runtime.now_str(),
    })
    for c, items in confusions.items():
        kb.set_slot(c, "typical_confusions", items)
    kb.data["kb_meta"]["updated"] = runtime.now_str()
    kb.data["kb_meta"]["updated_by"] = "cxr-xai tune"
    class_names = json_io.load_json(run.class_names_json) if run.class_names_json.exists() \
        else constants.CLASS_NAMES
    errors = kb.validate(class_names=class_names, num_outputs=probs.shape[1])
    if errors:
        sys.exit("[ПОМИЛКА] Після оновлення БЗ некоректна:\n- " + "\n- ".join(errors))
    kb.save(run.kb_path)

    # --- 4. Статистика вирішувача на val з обраними параметрами ---
    engine = InferenceEngine(kb)
    summary, per_sample = rule_tuning.engine_statistics(engine, probs, pred["label"].tolist())
    pd.DataFrame([summary]).round(4).to_csv(run.tuning_dir / "rules_val_stats.csv", index=False)
    per_sample.to_csv(run.tuning_dir / "decisions_final_val.csv", index=False)

    conf_rows = [{"class": c, **item} for c, items in confusions.items() for item in items]
    pd.DataFrame(conf_rows).to_csv(run.tuning_dir / "typical_confusions_val.csv", index=False)

    sel_t = t_tab[t_tab["threshold"] == t_best].iloc[0]
    sel_d = d_tab[d_tab["threshold"] == d_best].iloc[0]
    report = "\n".join([
        "ЗВІТ ПРО ДОБІР ПАРАМЕТРІВ ВИРІШУВАЧА (валідаційна вибірка)",
        f"Дата: {runtime.now_str()}",
        f"Критерій: {settings.threshold_criterion}; макс. частка позначених: {settings.max_flag_rate:.0%}",
        "",
        f"T = {t_best:.2f} ({t_why})",
        f"   позначено {sel_t['flag_rate']:.1%} зображень; вловлено помилок {sel_t['errors_captured']:.1%}; "
        f"accuracy серед непозначених {sel_t['accuracy_unflagged']:.4f}",
        f"δ = {d_best:.2f} ({d_why})",
        f"   позначено {sel_d['flag_rate']:.1%} зображень; вловлено помилок {sel_d['errors_captured']:.1%}",
        "",
        f"Типові плутанини (поріг частоти {settings.confusion_min_rate:.0%}):",
        *[f"  {c}: " + (", ".join(f"{i['with']} ({i['p_true_given_pred']:.1%} / {i['p_pred_given_true']:.1%})"
                                  for i in items) or "—") for c, items in confusions.items()],
        "  (перше число - P(справжній = j | прогноз = c), друге - P(прогноз = j | справжній = c))",
        "",
        "Статистика вирішувача на val:",
        *[f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in summary.items()],
        "",
        f"Базу знань із підібраними параметрами збережено: {run.kb_path} (шаблон {settings.kb_path.name} не змінено)",
    ])
    (run.tuning_dir / "tuning_report.txt").write_text(report, encoding="utf-8")
    print(report)
    if (~correct).sum() < 30:
        print("\n[УВАГА] На валідаційній вибірці менше 30 помилок - оцінки порогів нестабільні; "
              "зазначте це як обмеження в курсовій.")


if __name__ == "__main__":
    main()
