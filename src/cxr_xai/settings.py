"""
Налаштування застосунку (pydantic-settings).

Значення беруться в такому порядку (перше знайдене перемагає):
  1) змінні середовища з префіксом CXR_ (наприклад, CXR_IMG_SIZE=224);
  2) файл .env у корені проєкту (шаблон із поясненнями: .env.example);
  3) значення за замовчуванням із цього класу.

Некоректні значення та помилки в назвах змінних у .env зупиняють запуск одразу,
а не посеред навчання. Незмінні факти про датасет (порядок класів тощо)
зберігаються окремо, у constants.py.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Grid = tuple[float, float, float]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CXR_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="forbid",          # помилка в назві змінної CXR_* у .env не лишиться непоміченою
        frozen=True,
        validate_default=True,
    )

    # --- Шляхи (відносні шляхи рахуються від кореня проєкту) ---
    npz_dir: Path = Path("dataverse_files")     # оригінальні NPZ-файли датасету
    data_root: Path = Path("dataset")           # розпаковані PNG-зображення
    output_dir: Path = Path("outputs")          # усі результати
    kb_path: Path = Path("knowledge_base.json")

    # --- Перевірка файлів ---
    min_image_side: int = Field(64, gt=0)       # зображення з меншою стороною непридатні

    # --- Виявлення дублікатів (виконується ДО поділу на вибірки) ---
    # Майже-дублікати: відстань Геммінга між 64-бітними pHash <= порога. Рентгенограми схожі
    # між собою, тому поріг має бути малим; перевірте figures/near_duplicate_pairs.png.
    phash_threshold: int = Field(4, ge=0, le=64)
    # keep_one - з групи дублікатів лишається одне зображення;
    # keep_all - лишаються всі, але група цілком потрапляє в одну вибірку.
    dedup_policy: Literal["keep_one", "keep_all"] = "keep_one"
    drop_conflicting_groups: bool = True        # вилучати групи однакових знімків з РІЗНИМИ мітками
    check_mirrored: bool = True                 # шукати також віддзеркалені копії

    # --- Поділ на вибірки ---
    seed: int = 42
    split_ratios: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

    # --- Модель і навчання ---
    img_size: int = Field(299, ge=71)           # 299 - стандартний вхід Xception, 71 - мінімум
    batch_size: int = Field(32, gt=0)
    # imagenet - перенесення знань; none - випадкові ваги (ЛИШЕ для перевірки коду).
    backbone_weights: Literal["imagenet", "none"] = "imagenet"
    dropout: float = Field(0.3, ge=0, lt=1)

    # Етап 1: навчається лише «голова» класифікатора, основа Xception заморожена.
    stage1_epochs: int = Field(15, gt=0)
    stage1_lr: float = Field(1e-3, gt=0)
    # Етап 2 (fine-tuning): розморожуються блоки Xception, починаючи з fine_tune_from_block
    # (13-14 - exit flow), з меншою швидкістю навчання.
    stage2_epochs: int = Field(20, gt=0)
    stage2_lr: float = Field(1e-5, gt=0)
    fine_tune_from_block: int = Field(13, ge=1, le=14)

    use_class_weights: bool = True              # компенсація дисбалансу класів
    early_stop_patience: int = Field(5, gt=0)
    reduce_lr_patience: int = Field(2, gt=0)
    reduce_lr_factor: float = Field(0.2, gt=0, lt=1)
    min_lr: float = Field(1e-7, gt=0)
    mixed_precision: bool = False               # пришвидшує навчання на сучасних GPU

    # --- Аугментація (ЛИШЕ для train) ---
    # Горизонтальне віддзеркалення свідомо НЕ використовується: розташування серця та інших
    # органів асиметричне, тож віддзеркалений знімок анатомічно неправдоподібний.
    aug_rotation: float = Field(0.02, ge=0)     # частка повного оберту: 0.02 * 360° ≈ ±7°
    aug_zoom: float = Field(0.10, ge=0)
    aug_translation: float = Field(0.05, ge=0)
    aug_contrast: float = Field(0.10, ge=0)

    # --- Вирішувач: добір порогів T і δ за ВАЛІДАЦІЙНОЮ вибіркою ---
    # youden - поріг максимізує J = TPR - FPR для задачі «виявити помилкову класифікацію»;
    # target_accuracy - найменший поріг, за якого точність серед непозначених
    # зображень >= target_accuracy.
    threshold_criterion: Literal["youden", "target_accuracy"] = "youden"
    target_accuracy: float = Field(0.95, gt=0, le=1)
    max_flag_rate: float = Field(0.30, gt=0, le=1)  # максимальна частка позначених зображень
    t_grid: Grid = (0.30, 0.99, 0.01)           # (від, до, крок)
    delta_grid: Grid = (0.00, 0.80, 0.01)
    # Клас j - «типова плутанина» для c, якщо P(справжній = j | прогноз = c)
    # або P(прогноз = j | справжній = c) на валідації не менше цього значення.
    confusion_min_rate: float = Field(0.03, ge=0, le=1)

    # --- Grad-CAM ---
    gradcam_alpha: float = Field(0.40, ge=0, le=1)  # прозорість теплової карти
    border_fraction: float = Field(0.10, gt=0, lt=0.5)  # ширина смуги для евристики «увага на краях»
    gradcam_stats_max_correct: int = Field(200, ge=0)   # правильних прикладів для статистики

    # --- Інтерфейс ---
    max_upload_mb: int = Field(20, gt=0)
    display_max_side: int = Field(512, gt=0)

    # --- Валідація ---
    @field_validator("npz_dir", "data_root", "output_dir", "kb_path")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        path = value.expanduser()
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

    @field_validator("split_ratios")
    @classmethod
    def _check_split_ratios(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != {"train", "val", "test"}:
            raise ValueError("потрібні рівно ключі train, val, test")
        if any(ratio <= 0 for ratio in value.values()) or not math.isclose(sum(value.values()), 1.0):
            raise ValueError("частки мають бути додатними, а їх сума має дорівнювати 1")
        return value

    @field_validator("t_grid", "delta_grid")
    @classmethod
    def _check_grid(cls, value: Grid) -> Grid:
        start, stop, step = value
        if not (start < stop and step > 0):
            raise ValueError("сітка задається як [від, до, крок], де від < до і крок > 0")
        return value

    # --- Похідні шляхи: визначаються структурою каталогу результатів, а не конфігурацією ---
    @property
    def data_dir(self) -> Path:
        return self.output_dir / "data"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def inventory_csv(self) -> Path:
        return self.data_dir / "inventory.csv"

    @property
    def dedup_csv(self) -> Path:
        return self.data_dir / "dedup_groups.csv"

    @property
    def clean_csv(self) -> Path:
        return self.data_dir / "clean_inventory.csv"

    @property
    def split_csv(self) -> Path:
        return self.data_dir / "split.csv"

    @property
    def dataset_meta_json(self) -> Path:
        return self.data_dir / "dataset_meta.json"

    @property
    def stage1_model(self) -> Path:
        return self.models_dir / "stage1_best.keras"

    @property
    def final_model(self) -> Path:
        return self.models_dir / "final.keras"

    @property
    def class_names_json(self) -> Path:
        return self.models_dir / "class_names.json"

    @property
    def train_meta_json(self) -> Path:
        return self.models_dir / "training_meta.json"


settings = Settings()
