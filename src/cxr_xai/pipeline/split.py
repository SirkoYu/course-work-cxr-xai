"""
Крок 3. Стратифікований поділ очищеного датасету на train/val/test.

* Частки задаються в settings.split_ratios (за замовчуванням 70/15/15).
* Поділ виконується на рівні груп дублікатів, тож копії одного знімка
  не можуть опинитися в різних вибірках.
* Seed фіксований (settings.seed) - поділ відтворюваний.
* Після поділу виконується автоматична перевірка відсутності витоку.

Запуск:
    cxr-xai split
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from cxr_xai import constants, runtime
from cxr_xai.data import class_labels, splitting
from cxr_xai.settings import settings


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="Стратифікований поділ на вибірки без витоку").parse_args(argv)
    runtime.ensure_dirs()
    if not settings.clean_csv.exists():
        sys.exit("[ПОМИЛКА] Немає clean_inventory.csv - спочатку виконайте cxr-xai dedup")
    df = pd.read_csv(settings.clean_csv)

    df = splitting.group_stratified_split(df, settings.split_ratios, settings.seed)
    df["label_idx"] = df["label"].map(class_labels.class_to_index()).astype(int)

    problems = splitting.check_no_leakage(df)
    if problems:
        sys.exit("[ПОМИЛКА] Виявлено витік даних між вибірками:\n- " + "\n- ".join(problems))

    cols = ["path", "label", "label_idx", "split", "group_id", "sha256", "phash", "width", "height"]
    df[cols].to_csv(settings.split_csv, index=False)

    table = (df.pivot_table(index="label", columns="split", values="path", aggfunc="count", fill_value=0)
             .reindex(index=constants.CLASS_NAMES, columns=["train", "val", "test"]))
    table["Разом"] = table.sum(axis=1)
    table.loc["Разом"] = table.sum()
    share = (table.loc[constants.CLASS_NAMES, "Разом"] / table.loc["Разом", "Разом"] * 100).round(1)
    table["Частка класу, %"] = list(share) + [100.0]
    table.to_csv(settings.reports_dir / "split_by_class.csv")

    report = "\n".join([
        "ЗВІТ ПРО ПОДІЛ НА ВИБІРКИ",
        f"Дата: {runtime.now_str()}",
        f"Частки: {settings.split_ratios}; seed = {settings.seed}",
        "Перевірка витоку (групи, SHA-256, pHash, шляхи): пройдена",
        "",
        table.to_string(),
    ])
    (settings.reports_dir / "split_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nПоділ збережено: {settings.split_csv}")


if __name__ == "__main__":
    main()
