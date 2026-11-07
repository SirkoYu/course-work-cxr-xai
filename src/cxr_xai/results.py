"""
Структура каталогу результатів.

outputs/
├── dataset/                  # підготовка даних; спільна для всіх запусків навчання
│   ├── extract/  check/  dedup/  split/
└── runs/
    ├── latest_run.txt        # назва останнього завершеного запуску
    └── <запуск>/             # один запуск навчання з усім, що від нього залежить
        ├── settings.json     # налаштування, з якими виконано навчання
        ├── split.csv         # копія поділу, на якому навчено модель
        ├── knowledge_base.json   # БЗ з підібраними T, δ і плутанинами (створює cxr-xai tune)
        └── training/  validation/  tuning/  test/  gradcam/

Копія поділу гарантує, що модель оцінюється саме на тих зображеннях, яких не бачила під час
навчання, навіть якщо пізніше поділ виконано заново.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from cxr_xai import json_io
from cxr_xai.settings import settings

LATEST_RUN_FILE = "latest_run.txt"
TEST_EVALUATED_FILE = "TEST_EVALUATED.txt"
NO_RUN_MESSAGE = "[ПОМИЛКА] Немає запуску навчання - спочатку виконайте cxr-xai train"


def _prepare(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetResults:
    """Результати кроків підготовки даних."""
    root: Path

    @property
    def extract_dir(self) -> Path:
        return self.root / "extract"

    @property
    def check_dir(self) -> Path:
        return self.root / "check"

    @property
    def dedup_dir(self) -> Path:
        return self.root / "dedup"

    @property
    def split_dir(self) -> Path:
        return self.root / "split"

    @property
    def inventory_csv(self) -> Path:
        return self.check_dir / "inventory.csv"

    @property
    def dataset_meta_json(self) -> Path:
        return self.check_dir / "dataset_meta.json"

    @property
    def dedup_csv(self) -> Path:
        return self.dedup_dir / "dedup_groups.csv"

    @property
    def clean_csv(self) -> Path:
        return self.dedup_dir / "clean_inventory.csv"

    @property
    def split_csv(self) -> Path:
        return self.split_dir / "split.csv"

    def prepare(self) -> None:
        _prepare(self.extract_dir, self.check_dir, self.dedup_dir, self.split_dir)


@dataclass(frozen=True)
class RunResults:
    """Результати одного запуску навчання та всіх наступних кроків."""
    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def training_dir(self) -> Path:
        return self.root / "training"

    @property
    def validation_dir(self) -> Path:
        return self.root / "validation"

    @property
    def tuning_dir(self) -> Path:
        return self.root / "tuning"

    @property
    def test_dir(self) -> Path:
        return self.root / "test"

    @property
    def gradcam_dir(self) -> Path:
        return self.root / "gradcam"

    @property
    def settings_json(self) -> Path:
        return self.root / "settings.json"

    @property
    def split_csv(self) -> Path:
        return self.root / "split.csv"

    @property
    def kb_path(self) -> Path:
        return self.root / "knowledge_base.json"

    @property
    def stage1_model(self) -> Path:
        return self.training_dir / "stage1_best.keras"

    @property
    def final_model(self) -> Path:
        return self.training_dir / "final.keras"

    @property
    def class_names_json(self) -> Path:
        return self.training_dir / "class_names.json"

    @property
    def train_meta_json(self) -> Path:
        return self.training_dir / "training_meta.json"

    @property
    def test_evaluated_file(self) -> Path:
        return self.test_dir / TEST_EVALUATED_FILE

    def prepare(self) -> None:
        _prepare(self.training_dir, self.validation_dir, self.tuning_dir, self.test_dir, self.gradcam_dir)


def dataset() -> DatasetResults:
    return DatasetResults(settings.output_dir / "dataset")


def runs_dir() -> Path:
    return settings.output_dir / "runs"


def default_run_name(smoke: bool = False) -> str:
    suffix = "_smoke" if smoke else ""
    return f"{datetime.now():%Y-%m-%d_%H%M%S}_xception_{settings.img_size}{suffix}"


def create_run(smoke: bool = False) -> RunResults:
    """Створює каталог нового запуску. Результати наявного запуску ніколи не перезаписуються."""
    run = RunResults(runs_dir() / (settings.run or default_run_name(smoke)))
    if run.root.exists():
        sys.exit(f"[ПОМИЛКА] Запуск «{run.name}» уже існує. Вкажіть іншу назву через --run.")
    run.prepare()
    json_io.save_json(settings.model_dump(mode="json"), run.settings_json)
    return run


def mark_latest(run: RunResults) -> None:
    runs_dir().mkdir(parents=True, exist_ok=True)
    (runs_dir() / LATEST_RUN_FILE).write_text(run.name, encoding="utf-8")


def find_run() -> RunResults | None:
    """Запуск, заданий через CXR_RUN або --run; інакше останній завершений. None, якщо його немає."""
    name = settings.run
    if name is None:
        latest = runs_dir() / LATEST_RUN_FILE
        if not latest.exists():
            return None
        name = latest.read_text(encoding="utf-8").strip()
    run = RunResults(runs_dir() / name)
    return run if run.root.is_dir() else None


def require_run() -> RunResults:
    """Те саме, що find_run(), але для кроків конвеєра: без запуску завершує роботу з поясненням."""
    run = find_run()
    if run is None:
        if settings.run is not None:
            sys.exit(f"[ПОМИЛКА] Запуск «{settings.run}» не знайдено в {runs_dir()}")
        sys.exit(NO_RUN_MESSAGE)
    run.prepare()
    return run


def runs_with_evaluated_test() -> list[RunResults]:
    """Запуски, для яких уже оцінювали тестову вибірку: вибірка одна на всі запуски."""
    return [RunResults(flag.parents[1]) for flag in sorted(runs_dir().glob(f"*/test/{TEST_EVALUATED_FILE}"))]
