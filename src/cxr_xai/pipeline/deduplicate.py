"""
Крок 2. Виявлення дублікатів ДО поділу на train/val/test.

Чому до поділу: якщо копії одного знімка потраплять і в навчальну, і в тестову
вибірку, оцінка якості буде завищеною (витік даних, data leakage).

Що робить:
  1) обчислює SHA-256 (точні копії), pHash (майже-копії) і pHash віддзеркаленого
     зображення (віддзеркалені копії) для кожного придатного зображення;
  2) об'єднує дублікати в групи; групи з різними мітками вилучає (конфлікт розмітки);
  3) з кожної групи залишає одне зображення (settings.dedup_policy = "keep_one");
  4) рахує, скільки груп дублікатів розірвано ОРИГІНАЛЬНИМ поділом датасету
     (тобто скільки витоків містив готовий поділ);
  5) зберігає outputs/data/clean_inventory.csv і звіти/рисунки для курсової.

Запуск:
    cxr-xai dedup [--threshold 4] [--workers 8]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from cxr_xai import constants, runtime
from cxr_xai.data import deduplication
from cxr_xai.settings import settings


def compute_hashes(paths, workers: int):
    def work(p):
        h, hm = deduplication.phash_pair(p)
        return deduplication.sha256_file(p), h, hm
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(tqdm(ex.map(work, paths), total=len(paths), desc="Хешування", unit="файл"))
    if not results:
        return [], [], []
    sha, ph, phm = zip(*results, strict=True)
    return list(sha), list(ph), list(phm)


def plot_nn_hist(nn_dist: np.ndarray, threshold: int, path) -> None:
    """Гістограма відстаней до найближчого сусіда за pHash - обґрунтування порогу."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, min(int(nn_dist.max()) + 2, 40))
    ax.hist(nn_dist, bins=bins, color="#4C72B0", edgecolor="white")
    ax.axvline(threshold + 0.5, color="#C44E52", linestyle="--", label=f"поріг = {threshold}")
    ax.set_xlabel("Відстань Геммінга до найближчого зображення (pHash, 64 біти)")
    ax.set_ylabel("Кількість зображень")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pair_examples(df: pd.DataFrame, pairs_df: pd.DataFrame, data_root, path, max_pairs: int = 8) -> None:
    """Приклади пар майже-дублікатів і віддзеркалених копій для візуальної перевірки порогу."""
    cand = pairs_df[pairs_df["kind"] != "exact"]
    if cand.empty:
        return
    # пари з найбільшою відстанню - найризикованіші щодо хибних збігів
    cand = cand.sort_values("phash_distance", ascending=False).head(max_pairs)
    fig, axes = plt.subplots(len(cand), 2, figsize=(6, 3 * len(cand)))
    axes = np.atleast_2d(axes)
    for row, (_, p) in enumerate(cand.iterrows()):
        for col, key in enumerate(("i", "j")):
            r = df.iloc[int(p[key])]
            with Image.open(data_root / r["path"]) as im:
                axes[row, col].imshow(deduplication.to_grayscale_8bit(im), cmap="gray")
            axes[row, col].set_title(f"{r['label']} · {r['source_split'] or '-'}\n{r['width']}x{r['height']}",
                                     fontsize=8)
            axes[row, col].axis("off")
        tag = "дзеркально" if p["kind"] == "mirrored" else "майже-копія"
        axes[row, 0].text(1.05, 0.5, f"{tag}\nd={p['phash_distance']}", transform=axes[row, 0].transAxes,
                          ha="center", va="center", fontsize=9, color="#C44E52")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Виявлення дублікатів до поділу на вибірки")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-mirror", action="store_true", help="не шукати віддзеркалені копії")
    args = parser.parse_args(argv)
    check_mirrored = settings.check_mirrored and not args.no_mirror

    runtime.ensure_dirs()
    if not settings.inventory_csv.exists():
        sys.exit("[ПОМИЛКА] Немає inventory.csv - спочатку виконайте cxr-xai check")
    data_root = settings.data_root
    inv = pd.read_csv(settings.inventory_csv)
    inv["source_split"] = inv["source_split"].fillna("")
    df = inv[inv["ok"]].reset_index(drop=True)
    print(f"Придатних зображень: {len(df)} (з {len(inv)} в інвентарі)")

    sha, ph, phm = compute_hashes([data_root / p for p in df["path"]], args.workers)
    df["sha256"] = sha
    df["phash"] = [f"{h:016x}" for h in ph]
    df["phash_mirror"] = [f"{h:016x}" for h in phm]

    group_id, near, nn_dist, mirror = deduplication.group_duplicates(
        sha, ph, settings.phash_threshold, phm if check_mirrored else None)
    df["group_id"] = group_id
    df["nn_phash_dist"] = nn_dist
    df = deduplication.decide_actions(df, policy=settings.dedup_policy, drop_conflicts=settings.drop_conflicting_groups)

    df.to_csv(settings.dedup_csv, index=False)
    clean = df[df["action"] == "keep"].reset_index(drop=True)
    clean.to_csv(settings.clean_csv, index=False)

    # --- Таблиця всіх знайдених пар ---
    rows = []
    for kind_src, plist in (("near", near), ("mirrored", mirror)):
        for i, j, d in plist:
            kind = "exact" if sha[i] == sha[j] else kind_src
            if kind_src == "mirrored" and kind == "exact":
                continue                      # симетричні точні копії вже враховано
            rows.append({"i": i, "j": j, "kind": kind, "phash_distance": d})
    pairs_df = pd.DataFrame(rows, columns=["i", "j", "kind", "phash_distance"])
    if len(pairs_df):
        pairs_df["i_path"] = df["path"].values[pairs_df["i"]]
        pairs_df["j_path"] = df["path"].values[pairs_df["j"]]
        pairs_df["i_label"] = df["label"].values[pairs_df["i"]]
        pairs_df["j_label"] = df["label"].values[pairs_df["j"]]
        pairs_df["i_source_split"] = df["source_split"].values[pairs_df["i"]]
        pairs_df["j_source_split"] = df["source_split"].values[pairs_df["j"]]
    pairs_df.to_csv(settings.data_dir / "duplicate_pairs.csv", index=False)

    # --- Статистика для курсової ---
    sha_counts = df["sha256"].value_counts()
    n_exact_extra = int((sha_counts - 1).clip(lower=0).sum())       # зайві побайтові копії
    n_near_pairs = int((pairs_df["kind"] == "near").sum())
    n_mirror_pairs = int((pairs_df["kind"] == "mirrored").sum())
    groups = df.groupby("group_id")
    sizes = groups.size()
    n_dup_groups = int((sizes > 1).sum())
    n_in_dup_groups = int(sizes[sizes > 1].sum())
    n_conflict_groups = int((groups["label"].nunique() > 1).sum())
    multi = df[df["group_id"].isin(sizes[sizes > 1].index)]
    n_cross_split = int((multi.groupby("group_id")["source_split"].nunique() > 1).sum())

    before = df.groupby("label").size().rename("before")
    after = clean.groupby("label").size().rename("after")
    dropped_dup = df[df["action"] == "drop_duplicate"].groupby("label").size().rename("drop_duplicate")
    dropped_conf = df[df["action"] == "drop_conflict"].groupby("label").size().rename("drop_conflict")
    by_class = (pd.concat([before, dropped_dup, dropped_conf, after], axis=1)
                .reindex(constants.CLASS_NAMES).fillna(0).astype(int))
    by_class.loc["Разом"] = by_class.sum()
    by_class["dropped_%"] = (100 * (by_class["before"] - by_class["after"]) /
                             by_class["before"].replace(0, np.nan)).round(2)
    by_class.to_csv(settings.reports_dir / "dedup_by_class.csv")

    plot_nn_hist(nn_dist, settings.phash_threshold, settings.figures_dir / "phash_nn_hist.png")
    plot_pair_examples(df, pairs_df, data_root, settings.figures_dir / "near_duplicate_pairs.png")

    report = "\n".join([
        "ЗВІТ ПРО ВИЯВЛЕННЯ ДУБЛІКАТІВ",
        f"Дата: {runtime.now_str()}",
        f"Метод: SHA-256 (точні копії) + pHash 64 біти, поріг Геммінга = {settings.phash_threshold}"
        + ("; перевірка віддзеркалених копій: так" if check_mirrored else ""),
        f"Політика: {settings.dedup_policy}; вилучення конфліктних груп: {settings.drop_conflicting_groups}",
        f"Зображень на вході: {len(df)}",
        f"Зайвих побайтових копій (SHA-256): {n_exact_extra}",
        f"Пар майже-дублікатів (pHash, без побайтових копій): {n_near_pairs}",
        f"Пар віддзеркалених копій: {n_mirror_pairs}",
        f"Груп дублікатів (розмір > 1): {n_dup_groups}; зображень у них: {n_in_dup_groups}",
        f"Конфліктних груп (однаковий знімок з різними мітками): {n_conflict_groups}",
        f"Груп дублікатів, розірвених ОРИГІНАЛЬНИМ поділом train/val/test: {n_cross_split}",
        f"Зображень після очищення: {len(clean)}",
        "",
        by_class.to_string(),
        "",
        "Рисунки: figures/phash_nn_hist.png, figures/near_duplicate_pairs.png",
        "Список пар: data/duplicate_pairs.csv",
        "ПЕРЕВІРТЕ рисунок пар вручну: якщо там різні знімки - зменште поріг (--threshold).",
    ])
    (settings.reports_dir / "dedup_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
