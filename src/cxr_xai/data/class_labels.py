"""Визначення класу та оригінальної вибірки за назвами папок у шляху до зображення."""
from __future__ import annotations

import re
from pathlib import Path

from cxr_xai import constants


def normalize_name(name: str) -> str:
    """'Lung_Opacity' -> 'lung opacity'; 'COVID-19' -> 'covid 19'."""
    return re.sub(r"[\s_\-]+", " ", name.strip().lower()).strip()


_ALIASES = {normalize_name(k): v for k, v in constants.CLASS_ALIASES.items()}
_ALIASES.update({normalize_name(c): c for c in constants.CLASS_NAMES})


def infer_class(rel_path: Path) -> str | None:
    """
    Визначає клас за назвами батьківських папок (від найближчої до кореня).
    Повертає канонічну назву класу або None, якщо жодна папка не відповідає
    словнику CLASS_ALIASES.
    """
    for part in reversed(Path(rel_path).parts[:-1]):
        key = normalize_name(part)
        if key in _ALIASES:
            return _ALIASES[key]
    return None


def infer_source_split(rel_path: Path) -> str:
    """Повертає назву папки готового поділу (train/test/val), якщо вона є в шляху."""
    for part in Path(rel_path).parts[:-1]:
        if normalize_name(part) in constants.SPLIT_FOLDER_NAMES:
            return normalize_name(part)
    return ""


def class_to_index() -> dict[str, int]:
    return {c: i for i, c in enumerate(constants.CLASS_NAMES)}
