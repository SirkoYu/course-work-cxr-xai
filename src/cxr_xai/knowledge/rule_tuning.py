"""
Добір параметрів вирішувача (T, δ) і типових плутанин за ВАЛІДАЦІЙНОЮ вибіркою.

Ідея: правила R2 (p1 < T) і R3 (p1 - p2 < δ) мають позначати саме ті зображення,
на яких модель частіше помиляється. Тому добір розглядається як задача
виявлення помилок за оцінкою «score» (p1 або margin):

  позначено(x) = score(x) < поріг
  TPR = частка помилкових класифікацій, які позначено   (вловлені помилки)
  FPR = частка правильних класифікацій, які позначено   (зайві попередження)
  J   = TPR - FPR  (індекс Юдена)

Критерій "youden": поріг з максимальним J серед тих, що позначають не більше
MAX_FLAG_RATE валідаційних зображень. Критерій "target_accuracy": найменший
поріг, за якого точність серед непозначених зображень >= TARGET_ACCURACY.
Модуль не залежить від TensorFlow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def grid(spec: tuple[float, float, float]) -> np.ndarray:
    lo, hi, step = spec
    return np.round(np.arange(lo, hi + step / 2, step), 4)


def threshold_table(score: np.ndarray, correct: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    """Статистика правила «score < поріг» для кожного значення порогу."""
    score = np.asarray(score, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n, n_err = len(score), int((~correct).sum())
    n_ok = n - n_err
    rows = []
    for v in values:
        flagged = score < v
        kept = ~flagged
        tpr = (flagged & ~correct).sum() / n_err if n_err else np.nan
        fpr = (flagged & correct).sum() / n_ok if n_ok else np.nan
        rows.append({
            "threshold": float(v),
            "flag_rate": float(flagged.mean()),
            "coverage": float(kept.mean()),
            "accuracy_unflagged": float(correct[kept].mean()) if kept.any() else np.nan,
            "accuracy_flagged": float(correct[flagged].mean()) if flagged.any() else np.nan,
            "errors_captured": float(tpr),
            "correct_flagged": float(fpr),
            "youden_j": float(tpr - fpr) if n_err and n_ok else np.nan,
        })
    return pd.DataFrame(rows)


def choose_threshold(table: pd.DataFrame, criterion: str, max_flag_rate: float,
                     target_accuracy: float) -> tuple[float, str]:
    """Повертає (поріг, текстове обґрунтування вибору)."""
    allowed = table[table["flag_rate"] <= max_flag_rate]
    if allowed.empty:
        allowed = table.nsmallest(1, "flag_rate")
    if criterion == "youden":
        if allowed["youden_j"].isna().all():
            row = allowed.iloc[0]
            return float(row["threshold"]), ("на валідаційній вибірці немає помилок (або немає правильних "
                                              "відповідей), J не визначено; обрано найменший поріг")
        best = allowed["youden_j"].max()
        # серед порогів з максимальним J обираємо найменший (менше зайвих попереджень)
        row = allowed[allowed["youden_j"] == best].iloc[0]
        return float(row["threshold"]), (f"максимум індексу Юдена J = {best:.3f} за обмеження "
                                         f"частки позначених <= {max_flag_rate:.0%}")
    if criterion == "target_accuracy":
        ok = allowed[allowed["accuracy_unflagged"] >= target_accuracy]
        if not ok.empty:
            row = ok.iloc[0]
            return float(row["threshold"]), (f"найменший поріг, за якого точність серед непозначених "
                                             f">= {target_accuracy:.0%}")
        row = allowed.loc[allowed["accuracy_unflagged"].idxmax()]
        return float(row["threshold"]), (f"цільової точності {target_accuracy:.0%} не досягнуто; обрано поріг "
                                         f"з максимальною точністю серед непозначених")
    raise ValueError(f"Невідомий критерій: {criterion}")


def engine_statistics(engine, probs: np.ndarray, true_labels) -> tuple[dict, pd.DataFrame]:
    """
    Застосовує вирішувач до кожного вектора оцінок і рахує статистику правил:
    частоту спрацювання R1-R5, частку «впевнених» результатів (coverage),
    точність серед впевнених і невпевнених, точність top-2 для випадків R3.
    """
    rows = []
    for p, y in zip(probs, true_labels, strict=True):
        d = engine.infer(p)
        rows.append({
            "true": y, "top1": d.top1, "top2": d.top2, "p1": d.p1, "margin": d.margin,
            "status": d.status, "shown": "|".join(d.shown_classes),
            "n_shown": len(d.shown_classes), "fired": ",".join(d.fired_rules),
            "top1_correct": d.top1 == y, "true_in_shown": y in d.shown_classes,
        })
    df = pd.DataFrame(rows)
    conf = df["status"] == "confident"
    two = df["n_shown"] == 2
    summary = {
        "n": len(df),
        "T": engine.params["T"], "delta": engine.params["delta"],
        "accuracy_top1": float(df["top1_correct"].mean()),
        "coverage_confident": float(conf.mean()),
        "accuracy_confident": float(df.loc[conf, "top1_correct"].mean()) if conf.any() else float("nan"),
        "accuracy_low_confidence": float(df.loc[~conf, "top1_correct"].mean()) if (~conf).any() else float("nan"),
        "share_two_classes_shown": float(two.mean()),
        "true_in_shown_when_two": float(df.loc[two, "true_in_shown"].mean()) if two.any() else float("nan"),
        "true_in_shown_overall": float(df["true_in_shown"].mean()),
    }
    fired = df["fired"].str.split(",")
    for rid in ["R1", "R2", "R3", "R4", "R5"]:
        summary[f"fired_{rid}"] = float(fired.apply(lambda f, r=rid: r in f).mean())
    return summary, df


def typical_confusions(cm: np.ndarray, class_names: list[str], min_rate: float) -> dict[str, list[dict]]:
    """
    Для кожного класу c формує список класів j, з якими модель його плутає:
      p_true_given_pred = P(справжній = j | прогноз = c)   - «коли модель каже c, насправді j»;
      p_pred_given_true = P(прогноз = j | справжній = c)   - «знімки класу c модель відносить до j».
    Клас j включається, якщо хоча б одна з частот >= min_rate.
    """
    cm = np.asarray(cm, dtype=float)
    col = cm.sum(axis=0)
    row = cm.sum(axis=1)
    result = {}
    for c_idx, c in enumerate(class_names):
        items = []
        for j_idx, j in enumerate(class_names):
            if j_idx == c_idx:
                continue
            p_tp = cm[j_idx, c_idx] / col[c_idx] if col[c_idx] else 0.0
            p_pt = cm[c_idx, j_idx] / row[c_idx] if row[c_idx] else 0.0
            if max(p_tp, p_pt) >= min_rate:
                items.append({"with": j, "p_true_given_pred": round(float(p_tp), 4),
                              "p_pred_given_true": round(float(p_pt), 4),
                              "n_true_j_pred_c": int(cm[j_idx, c_idx]),
                              "n_true_c_pred_j": int(cm[c_idx, j_idx])})
        result[c] = sorted(items, key=lambda d: -max(d["p_true_given_pred"], d["p_pred_given_true"]))
    return result
