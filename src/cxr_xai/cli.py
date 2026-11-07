"""
Єдина точка входу: cxr-xai <команда> [параметри].

Будь-який параметр із таблиці OVERRIDES можна необов'язково перевизначити для одного запуску:
    cxr-xai train --batch-size 8 --img-size 224

Пріоритет: параметр командного рядка > змінна середовища > .env > значення за замовчуванням.
Параметр перетворюється на змінну CXR_* ДО першого імпорту cxr_xai.settings, тому проходить
ту саму валідацію, що й .env. З цієї причини тут не можна імпортувати settings на рівні модуля.
Модулі команд теж імпортуються лише під час запуску, щоб не завантажувати TensorFlow без потреби.
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

# Порядок відповідає порядку виконання кроків конвеєра.
COMMANDS = {
    "extract": ("cxr_xai.pipeline.extract_npz", "NPZ-файли датасету -> PNG-зображення"),
    "check": ("cxr_xai.pipeline.check_files", "інвентаризація та перевірка файлів"),
    "dedup": ("cxr_xai.pipeline.deduplicate", "пошук дублікатів (SHA-256 + pHash) до поділу"),
    "split": ("cxr_xai.pipeline.split", "стратифікований поділ train/val/test без витоку"),
    "train": ("cxr_xai.pipeline.train", "навчання: перенесення знань + fine-tuning"),
    "evaluate": ("cxr_xai.pipeline.evaluate", "метрики на валідаційній вибірці"),
    "tune": ("cxr_xai.pipeline.tune_rules", "добір T і δ, типові плутанини -> база знань"),
    "report": ("cxr_xai.pipeline.final_report", "одноразове фінальне оцінювання на test"),
    "gradcam": ("cxr_xai.pipeline.gradcam_analysis", "аналіз хибних класифікацій з Grad-CAM"),
    "kb": ("cxr_xai.knowledge.knowledge_base", "перевірка та перегляд бази знань"),
}
APP_COMMAND = "app"
APP_PATH = Path(__file__).with_name("app.py")

# Налаштування, які використовує кожна команда і які можна перевизначити параметром.
OVERRIDES = {
    "extract": ("npz_dir", "data_root", "output_dir"),
    "check": ("data_root", "output_dir", "min_image_side"),
    "dedup": ("data_root", "output_dir", "phash_threshold", "dedup_policy", "drop_conflicting_groups",
              "check_mirrored"),
    "split": ("output_dir", "seed", "split_ratios"),
    "train": ("run", "data_root", "output_dir", "seed", "img_size", "batch_size", "backbone_weights", "dropout",
              "stage1_epochs", "stage1_lr", "stage2_epochs", "stage2_lr", "fine_tune_from_block",
              "use_class_weights", "early_stop_patience", "reduce_lr_patience", "reduce_lr_factor", "min_lr",
              "mixed_precision", "aug_rotation", "aug_zoom", "aug_translation", "aug_contrast"),
    "evaluate": ("run", "data_root", "output_dir", "batch_size"),
    "tune": ("run", "output_dir", "kb_path", "threshold_criterion", "target_accuracy", "max_flag_rate", "t_grid",
             "delta_grid", "confusion_min_rate"),
    "report": ("run", "data_root", "output_dir", "batch_size"),
    "gradcam": ("run", "data_root", "output_dir", "seed", "batch_size", "gradcam_alpha", "border_fraction",
                "gradcam_stats_max_correct"),
    "kb": ("run", "output_dir", "kb_path"),
    APP_COMMAND: ("run", "output_dir", "kb_path", "min_image_side", "gradcam_alpha", "max_upload_mb",
                  "display_max_side"),
}
BOOLEAN_SETTINGS = {"drop_conflicting_groups", "check_mirrored", "use_class_weights", "mixed_precision"}
# Коротші назви окремих параметрів.
ALIASES = {"stage1_epochs": "--epochs1", "stage2_epochs": "--epochs2", "backbone_weights": "--weights",
           "phash_threshold": "--threshold", "kb_path": "--kb"}


def usage() -> str:
    lines = ["Використання: cxr-xai <команда> [параметри]", "", "Команди (у порядку виконання):"]
    lines += [f"  {name:<10}{description}" for name, (_, description) in COMMANDS.items()]
    lines += [f"  {APP_COMMAND:<10}інтерфейс Streamlit", "",
              "Параметри команди, зокрема перевизначення налаштувань: cxr-xai <команда> --help"]
    return "\n".join(lines)


def build_override_parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, usage=argparse.SUPPRESS,
                                     description="Перевизначення налаштувань із .env (необов'язково):")
    for name in OVERRIDES[command]:
        flags = [f"--{name.replace('_', '-')}"]
        if name in ALIASES:
            flags.append(ALIASES[name])
        env_name = f"CXR_{name.upper()}"
        if name in BOOLEAN_SETTINGS:
            parser.add_argument(*flags, dest=name, action=argparse.BooleanOptionalAction, help=env_name)
        else:
            parser.add_argument(*flags, dest=name, metavar="ЗНАЧЕННЯ", help=env_name)
    return parser


def apply_overrides(command: str, args: list[str]) -> list[str]:
    """Записує задані параметри у змінні CXR_* і повертає решту параметрів для самої команди."""
    known, rest = build_override_parser(command).parse_known_args(args)
    for name, value in vars(known).items():
        if value is None:
            continue
        os.environ[f"CXR_{name.upper()}"] = str(value).lower() if isinstance(value, bool) else str(value)
    return rest


def load_settings_or_exit() -> None:
    """Перше завантаження налаштувань: показує помилки конфігурації без трасування стека."""
    from pydantic import ValidationError
    try:
        importlib.import_module("cxr_xai.settings")
    except ValidationError as error:
        print(f"[ПОМИЛКА] Некоректні налаштування:\n{error}", file=sys.stderr)
        raise SystemExit(2) from None


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(usage())
        return

    command, rest = args[0], args[1:]
    if command != APP_COMMAND and command not in COMMANDS:
        print(f"Невідома команда: {command}\n\n{usage()}", file=sys.stderr)
        raise SystemExit(2)

    rest = apply_overrides(command, rest)
    load_settings_or_exit()

    if command == APP_COMMAND:
        result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH), *rest], check=False)
        raise SystemExit(result.returncode)

    sys.argv[0] = f"cxr-xai {command}"   # для коректного тексту --help у argparse
    module = importlib.import_module(COMMANDS[command][0])
    if "-h" in rest or "--help" in rest:
        try:
            module.main(["--help"])
        except SystemExit:
            pass
        print("\n" + build_override_parser(command).format_help())
        return
    module.main(rest)


if __name__ == "__main__":
    main()
