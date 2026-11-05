"""
Стратифікований поділ на train/val/test на рівні ГРУП дублікатів
та перевірка відсутності витоку даних між вибірками.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def group_stratified_split(df: pd.DataFrame, ratios: dict, seed: int) -> pd.DataFrame:
    """
    Ділить датафрейм на вибірки так, щоб:
      * кожна група дублікатів (group_id) цілком потрапила в одну вибірку;
      * пропорції класів у вибірках збігалися (стратифікація за міткою групи).
    Очікувані стовпці: path, label, group_id. Конфліктні групи (різні мітки)
    мають бути вилучені раніше (cxr-xai dedup).
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Сума часток має дорівнювати 1, отримано {ratios}")
    labels_per_group = df.groupby("group_id")["label"].nunique()
    if (labels_per_group > 1).any():
        raise ValueError("Є групи з різними мітками - спочатку вилучіть конфліктні групи.")

    groups = df.groupby("group_id")["label"].first().reset_index()
    counts = groups["label"].value_counts()
    if (counts < 3).any():
        raise ValueError(f"Для стратифікації потрібно щонайменше 3 групи на клас: {counts.to_dict()}")

    holdout = ratios["val"] + ratios["test"]
    g_train, g_hold = train_test_split(groups, test_size=holdout, stratify=groups["label"],
                                       random_state=seed)
    test_share = ratios["test"] / holdout
    g_val, g_test = train_test_split(g_hold, test_size=test_share, stratify=g_hold["label"],
                                     random_state=seed)
    split_of_group = {}
    for name, part in (("train", g_train), ("val", g_val), ("test", g_test)):
        split_of_group.update(dict.fromkeys(part["group_id"], name))

    out = df.copy()
    out["split"] = out["group_id"].map(split_of_group)
    return out


def check_no_leakage(df: pd.DataFrame) -> list[str]:
    """
    Повертає список виявлених проблем (порожній, якщо витоку немає):
      * група дублікатів присутня в кількох вибірках;
      * однаковий SHA-256 або pHash у різних вибірках;
      * один і той самий файл у кількох вибірках.
    """
    problems = []
    for col, what in (("group_id", "групи дублікатів"), ("sha256", "SHA-256"),
                      ("phash", "pHash"), ("path", "шляхи файлів")):
        if col not in df.columns:
            continue
        n_splits = df.groupby(col)["split"].nunique()
        bad = n_splits[n_splits > 1]
        if len(bad):
            problems.append(f"{what}: {len(bad)} значень зустрічаються в кількох вибірках")
    return problems
