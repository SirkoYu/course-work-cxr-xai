"""
Крок 1. Інвентаризація та перевірка файлів датасету.

Що робить:
  1) рекурсивно сканує каталог датасету;
  2) визначає клас кожного зображення за назвою папки (constants.CLASS_ALIASES);
  3) перевіряє, що файл відкривається (PIL), має підтримуваний формат і
     достатній розмір, а також (за замовчуванням) що його декодує TensorFlow;
  4) зберігає інвентар outputs/data/inventory.csv та звіти для курсової.

Запуск:
    cxr-xai check        # шлях до датасету: --data-root або CXR_DATA_ROOT у .env
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from cxr_xai import constants, json_io, runtime
from cxr_xai.data import class_labels
from cxr_xai.settings import settings


def inspect_with_pil(path: Path) -> dict:
    """Відкриває файл засобами PIL і повертає його характеристики."""
    info = {"width": None, "height": None, "mode": "", "format": "", "error": ""}
    try:
        with Image.open(path) as im:
            im.verify()                      # перевірка цілісності структури файлу
        with Image.open(path) as im:         # verify() «закриває» об'єкт, тому відкриваємо ще раз
            im.load()                        # повне декодування пікселів
            info.update(width=im.width, height=im.height, mode=im.mode, format=im.format or "")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        info["error"] = f"pil_error: {type(e).__name__}: {e}"[:300]
        return info
    if info["format"] not in constants.SUPPORTED_PIL_FORMATS:
        info["error"] = f"unsupported_format: {info['format']}"
    elif min(info["width"], info["height"]) < settings.min_image_side:
        info["error"] = f"too_small: {info['width']}x{info['height']}"
    return info


def make_tf_checker():
    """Повертає функцію перевірки декодування TensorFlow або None, якщо TF недоступний."""
    try:
        import tensorflow as tf
    except ImportError:
        return None

    def check(path: Path) -> str:
        try:
            raw = tf.io.read_file(str(path))
            img = tf.io.decode_image(raw, channels=3, expand_animations=False)
            if img.shape.rank != 3 or img.shape[-1] != 3:
                return f"tf_error: неочікувана форма {img.shape}"
        except Exception as e:  # noqa: BLE001 - фіксуємо будь-яку помилку декодування
            return f"tf_error: {type(e).__name__}: {str(e)[:200]}"
        return ""

    return check


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Інвентаризація та перевірка файлів датасету")
    parser.add_argument("--no-tf-check", action="store_true",
                        help="не перевіряти декодування засобами TensorFlow (швидше)")
    args = parser.parse_args(argv)

    runtime.ensure_dirs()
    data_root = settings.data_root
    if not data_root.is_dir():
        sys.exit(f"[ПОМИЛКА] Каталог датасету не знайдено: {data_root}\n"
                 f"Вкажіть його через --data-root або CXR_DATA_ROOT у .env.")

    print(f"Сканування: {data_root}")
    all_files = [p for p in data_root.rglob("*") if p.is_file()]
    image_files = sorted(p for p in all_files if p.suffix.lower() in constants.IMAGE_EXTENSIONS)
    other_ext = Counter(p.suffix.lower() or "<без розширення>" for p in all_files
                        if p.suffix.lower() not in constants.IMAGE_EXTENSIONS)
    print(f"Усього файлів: {len(all_files)}; зображень підтримуваних форматів: {len(image_files)}")

    tf_check = None if args.no_tf_check else make_tf_checker()
    if tf_check is None and not args.no_tf_check:
        print("[УВАГА] TensorFlow не встановлено - перевірку декодування TF пропущено.")

    rows = []
    for p in tqdm(image_files, desc="Перевірка файлів", unit="файл"):
        rel = p.relative_to(data_root)
        info = inspect_with_pil(p)
        if not info["error"] and tf_check is not None:
            info["error"] = tf_check(p)
        cls = class_labels.infer_class(rel)
        rows.append({
            "path": rel.as_posix(),
            "folder": rel.parent.as_posix(),
            "label": cls,
            "source_split": class_labels.infer_source_split(rel),
            "file_size": p.stat().st_size,
            "width": info["width"],
            "height": info["height"],
            "mode": info["mode"],
            "format": info["format"],
            "ok": info["error"] == "" and cls is not None,
            "error": info["error"] if info["error"] else ("unknown_class" if cls is None else ""),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("[ПОМИЛКА] Не знайдено жодного зображення. Перевірте шлях до датасету.")
    df.to_csv(settings.inventory_csv, index=False)
    json_io.save_json({
        "data_root": str(data_root),
        "created": runtime.now_str(),
        "n_files_total": len(all_files),
        "n_image_files": len(image_files),
        "other_extensions": dict(other_ext),
    }, settings.dataset_meta_json)

    # --- Звіти ---
    folder_summary = (df.groupby(["folder", "label"], dropna=False).size()
                      .reset_index(name="n_images").sort_values("folder"))
    folder_summary.to_csv(settings.reports_dir / "folder_summary.csv", index=False)

    bad = df[~df["ok"]]
    bad.to_csv(settings.reports_dir / "bad_files.csv", index=False)

    known = df[df["label"].notna()].copy()
    known["min_side"] = known[["width", "height"]].min(axis=1)
    by_class = known.groupby("label").agg(
        n_total=("path", "size"),
        n_ok=("ok", "sum"),
        median_width=("width", "median"),
        median_height=("height", "median"),
        min_side=("min_side", "min"),
    ).reindex(constants.CLASS_NAMES)
    by_class["n_rejected"] = by_class["n_total"] - by_class["n_ok"]
    modes = known.groupby("label")["mode"].agg(lambda s: ", ".join(f"{k}:{v}" for k, v in Counter(s).items()))
    by_class["pixel_modes"] = modes
    by_class.to_csv(settings.reports_dir / "inventory_by_class.csv")

    err_types = Counter(e.split(":")[0] for e in bad["error"])

    lines = [
        "ЗВІТ ПРО ПЕРЕВІРКУ ФАЙЛІВ ДАТАСЕТУ",
        f"Дата: {runtime.now_str()}",
        f"Каталог: {data_root}",
        f"Усього файлів: {len(all_files)}",
        f"Файлів зображень (розширення {sorted(constants.IMAGE_EXTENSIONS)}): {len(image_files)}",
        f"Інші файли за розширенням: {dict(other_ext)}",
        f"Придатних зображень (ok): {int(df['ok'].sum())}",
        f"Відхилено: {len(bad)}; типи причин: {dict(err_types)}",
        "",
        "Кількість за класами (до виявлення дублікатів):",
        by_class[["n_total", "n_ok", "n_rejected", "median_width", "median_height"]].to_string(),
        "",
        "Готовий поділ в оригінальному датасеті (для довідки; НЕ використовується):",
        known.groupby(["source_split", "label"]).size().unstack(fill_value=0).to_string(),
    ]
    unknown = df[df["label"].isna()]
    if len(unknown):
        lines += ["", f"[УВАГА] {len(unknown)} зображень без визначеного класу. Папки:",
                  unknown["folder"].value_counts().head(20).to_string(),
                  "Додайте назви цих папок у constants.CLASS_ALIASES і запустіть скрипт повторно."]
    report = "\n".join(lines)
    (settings.reports_dir / "check_files_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nІнвентар збережено: {settings.inventory_csv}")


if __name__ == "__main__":
    main()
