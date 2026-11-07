"""
Крок 0. Розпакування датасету з NPZ-файлів у PNG-зображення.

Датасет «Chest X-Ray Dataset for Respiratory Disease Classification» (Harvard Dataverse)
поширюється як три файли Dataset5_raw_{train,val,test}.npz. Кожен містить ключі:
  image       - 4D-масив пікселів усіх зображень;
  image_name  - масив назв вихідних файлів;
  image_label - масив міток 0..4 (0 COVID-19, 1 Lung-Opacity, 2 Normal,
                3 Viral Pneumonia, 4 Tuberculosis).

Скрипт зберігає кожне зображення як PNG (без втрат) у структурі
    dataset/<оригінальна вибірка>/<клас>/<вибірка>_<номер>__<назва>.png
і створює dataset/npz_index.csv (відповідність файлів записам NPZ).
Після цього працюють усі наступні кроки (01-08) та інтерфейс.

Оригінальний поділ train/val/test зберігається лише для довідки: усі зображення
надалі об'єднуються, очищуються від дублікатів і діляться заново (02, 03).

Запуск:
    cxr-xai extract        # шляхи: --npz-dir, --data-root або .env
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from cxr_xai import constants, results, runtime
from cxr_xai.settings import settings

REQUIRED_KEYS = {"image", "image_name", "image_label"}
SPLIT_PATTERNS = (("train", r"train"), ("val", r"val"), ("test", r"test"))


def split_of(path: Path) -> str:
    stem = path.stem.lower()
    for split, pattern in SPLIT_PATTERNS:
        if re.search(pattern, stem):
            return split
    return "all"


def class_folder(class_name: str) -> str:
    return class_name.replace(" ", "_")


def safe_name(name: str, max_len: int = 80) -> str:
    stem = re.sub(r"\.(png|jpe?g|bmp|tiff?|dcm)$", "", name.strip(), flags=re.IGNORECASE)
    stem = re.sub(r"[^A-Za-z0-9._()\- ]+", "_", stem).strip(" .") or "image"
    return stem[:max_len]


def name_prefix(name: str) -> str:
    """Текстова частина назви до першої цифри: 'COVID-123' -> 'COVID', 'CHNCXR_0001_0' -> 'CHNCXR'.
    Дає змогу оцінити, з яких джерел походять зображення класу."""
    stem = re.sub(r"\.(png|jpe?g|bmp|tiff?|dcm)$", "", name.strip(), flags=re.IGNORECASE)
    m = re.match(r"^(.*?)[\s_\-(]*\d", stem)
    prefix = (m.group(1) if m else stem).strip(" _-(")
    return prefix[:40] if prefix else "<починається з цифри>"


def as_str(x) -> str:
    return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x)


def load_npz(path: Path, allow_pickle: bool):
    try:
        with np.load(path, allow_pickle=allow_pickle) as z:
            missing = REQUIRED_KEYS - set(z.files)
            if missing:
                sys.exit(f"[ПОМИЛКА] У {path.name} немає ключів {missing}; є лише {z.files}")
            print(f"  читання {path.name} ...")
            return z["image"], z["image_name"], z["image_label"]
    except ValueError as e:
        if "allow_pickle" in str(e):
            sys.exit(f"[ПОМИЛКА] {path.name} містить об'єкти Python (pickle). Завантаження pickle з "
                     f"невідомих джерел небезпечне; якщо ви довіряєте файлу, запустіть з --allow-pickle.")
        raise


def decide_scaling(images: np.ndarray) -> tuple[str, float]:
    """
    Визначає ОДНЕ правило приведення пікселів до uint8 для всього масиву
    (однакове для всіх зображень, щоб не спотворювати відносну яскравість):
      uint8                  -> без змін;
      float з максимумом <= 1 -> x * 255;
      інші типи з максимумом <= 255 -> округлення;
      інакше                  -> x * 255 / глобальний максимум.
    """
    if images.dtype == np.uint8:
        return "none", 1.0
    rng = np.random.default_rng(0)
    idx = rng.choice(len(images), size=min(500, len(images)), replace=False)
    sample_max = float(np.max(images[np.sort(idx)]))
    global_max = float(np.max(images)) if images.size < 5e8 else sample_max
    if np.issubdtype(images.dtype, np.floating) and global_max <= 1.0 + 1e-6:
        return "x255", 255.0
    if global_max <= 255.0:
        return "round", 1.0
    return "scale_by_max", 255.0 / global_max


def to_uint8(img: np.ndarray, rule: str, factor: float) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):   # channels-first
        a = np.moveaxis(a, 0, -1)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if rule != "none":
        a = np.clip(np.rint(a.astype(np.float64) * factor), 0, 255)
    a = a.astype(np.uint8)
    if a.ndim == 3 and a.shape[-1] == 3 and np.array_equal(a[..., 0], a[..., 1]) \
            and np.array_equal(a[..., 0], a[..., 2]):
        a = a[..., 0]                         # 3 однакові канали -> градації сірого
    return a


def save_image(i: int, *, images, names, labels, split: str, npz_name: str, rule: str, factor: float,
               out: Path) -> dict:
    name = as_str(names[i])
    cls = constants.NPZ_LABELS[int(labels[i])]
    rel = Path(split) / class_folder(cls) / f"{split}_{i:05d}__{safe_name(name)}.png"
    Image.fromarray(to_uint8(images[i], rule, factor)).save(out / rel, compress_level=3)
    return {"split": split, "npz_file": npz_name, "index": i, "original_name": name,
            "name_prefix": name_prefix(name), "label_int": int(labels[i]), "label": cls,
            "file": rel.as_posix()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Розпакування NPZ-файлів датасету в PNG")
    parser.add_argument("--allow-pickle", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="перезаписати раніше розпаковані цим скриптом файли")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    npz_dir, out = settings.npz_dir, settings.data_root
    files = sorted(npz_dir.glob("*.npz"))
    if not files:
        sys.exit(f"[ПОМИЛКА] У каталозі {npz_dir} не знайдено файлів .npz "
                 f"(шлях задають --npz-dir або CXR_NPZ_DIR у .env)")
    index_csv = out / "npz_index.csv"
    if out.exists() and any(out.iterdir()):
        if not index_csv.exists():
            sys.exit(f"[ПОМИЛКА] Каталог {out} не порожній і створений не цим скриптом. "
                     f"Вкажіть інший через --data-root або CXR_DATA_ROOT у .env.")
        if not args.overwrite:
            sys.exit(f"[ПОМИЛКА] Датасет уже розпаковано в {out}. Для повторного розпакування додайте --overwrite.")
        for split in pd.read_csv(index_csv)["split"].unique():
            shutil.rmtree(out / str(split), ignore_errors=True)
        index_csv.unlink()
    out.mkdir(parents=True, exist_ok=True)
    ds = results.dataset()
    ds.prepare()

    rows, info = [], []
    for path in files:
        split = split_of(path)
        images, names, labels = load_npz(path, args.allow_pickle)
        labels = np.asarray(labels).astype(int).reshape(-1)
        if not (len(images) == len(names) == len(labels)):
            sys.exit(f"[ПОМИЛКА] {path.name}: різна кількість зображень ({len(images)}), "
                     f"назв ({len(names)}) і міток ({len(labels)})")
        unknown = sorted(set(labels.tolist()) - set(constants.NPZ_LABELS))
        if unknown:
            sys.exit(f"[ПОМИЛКА] {path.name}: невідомі мітки {unknown}")
        rule, factor = decide_scaling(images)
        info.append({"file": path.name, "split": split, "n": len(images), "shape": tuple(images.shape),
                     "dtype": str(images.dtype), "min": float(np.min(images[:200])),
                     "max": float(np.max(images[:200])), "scaling": rule})
        print(f"  {path.name}: {images.shape}, {images.dtype}, правило перетворення: {rule}")

        for lbl in constants.NPZ_LABELS.values():
            (out / split / class_folder(lbl)).mkdir(parents=True, exist_ok=True)

        save = partial(save_image, images=images, names=names, labels=labels, split=split,
                       npz_name=path.name, rule=rule, factor=factor, out=out)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            rows += list(tqdm(ex.map(save, range(len(images))), total=len(images),
                              desc=f"Запис PNG ({split})", unit="зобр."))
        del images, save

    df = pd.DataFrame(rows)
    df.to_csv(index_csv, index=False)
    df.to_csv(ds.extract_dir / "npz_index.csv", index=False)

    # --- Звіт ---
    counts = df.pivot_table(index="label", columns="split", values="file", aggfunc="count", fill_value=0)
    counts = counts.reindex(constants.CLASS_NAMES).fillna(0).astype(int)
    counts["Разом"] = counts.sum(axis=1)
    counts["За README"] = [constants.README_CLASS_COUNTS.get(c, 0) for c in counts.index]
    counts.loc["Разом"] = counts.sum()
    counts.to_csv(ds.extract_dir / "npz_counts.csv")

    prefixes = pd.crosstab(df["name_prefix"], df["label"]).reindex(columns=constants.CLASS_NAMES, fill_value=0)
    prefixes = prefixes.loc[prefixes.sum(axis=1).sort_values(ascending=False).index]
    prefixes.to_csv(ds.extract_dir / "name_prefixes_by_class.csv")

    dup_names = df[df.duplicated("original_name", keep=False)]
    report = "\n".join([
        "ЗВІТ ПРО РОЗПАКУВАННЯ NPZ",
        f"Дата: {runtime.now_str()}",
        f"Джерело: {npz_dir}",
        f"Призначення: {out}",
        "",
        "Файли:",
        pd.DataFrame(info).to_string(index=False),
        "",
        "Кількість зображень (оригінальний поділ) і звірка з README:",
        counts.to_string(),
        "",
        f"Назв файлів, що повторюються: {dup_names['original_name'].nunique()} "
        f"(записів: {len(dup_names)}); повтори назв можуть указувати на дублікати.",
        "",
        "Префікси назв файлів за класами (ознака джерела зображень), топ-25:",
        prefixes.head(25).to_string(),
    ])
    (ds.extract_dir / "extract_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("\nДалі: cxr-xai check")


if __name__ == "__main__":
    main()
