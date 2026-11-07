"""Налаштування: значення за замовчуванням, пріоритети джерел, валідація, актуальність .env.example."""
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from cxr_xai.settings import PROJECT_ROOT, Settings

ENV_EXAMPLE = PROJECT_ROOT / ".env.example"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Тести не повинні залежати від змінних CXR_* на машині розробника."""
    for name in list(os.environ):
        if name.startswith("CXR_"):
            monkeypatch.delenv(name)


def make_settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_env_example_matches_defaults():
    assert Settings(_env_file=ENV_EXAMPLE) == make_settings()


def test_env_example_documents_every_setting():
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    documented = {line.split("=", 1)[0] for line in lines if line.startswith("CXR_")}
    assert documented == {f"CXR_{name.upper()}" for name in Settings.model_fields}


def test_relative_paths_resolve_from_project_root():
    s = make_settings(data_root=Path("some/dir"))
    assert s.data_root == (PROJECT_ROOT / "some/dir").resolve()


def test_env_file_is_read_and_environment_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("CXR_IMG_SIZE=224\nCXR_BATCH_SIZE=8\n", encoding="utf-8")
    monkeypatch.setenv("CXR_BATCH_SIZE", "16")
    s = Settings(_env_file=env_file)
    assert s.img_size == 224
    assert s.batch_size == 16


def test_complex_values_are_parsed_from_env(monkeypatch):
    monkeypatch.setenv("CXR_SPLIT_RATIOS", '{"train": 0.8, "val": 0.1, "test": 0.1}')
    monkeypatch.setenv("CXR_T_GRID", "[0.5, 0.9, 0.05]")
    s = make_settings()
    assert s.split_ratios["train"] == 0.8
    assert s.t_grid == (0.5, 0.9, 0.05)


def test_empty_value_in_env_file_means_not_set(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("CXR_RUN=\n", encoding="utf-8")
    assert Settings(_env_file=env_file).run is None


def test_unknown_variable_in_env_file_is_rejected(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("CXR_IMG_SIZ=224\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)


@pytest.mark.parametrize("kwargs", [
    {"split_ratios": {"train": 0.9, "val": 0.2, "test": 0.1}},
    {"split_ratios": {"train": 0.7, "test": 0.3}},
    {"t_grid": (0.9, 0.3, 0.01)},
    {"t_grid": (0.3, 0.9, 0.0)},
    {"backbone_weights": "resnet"},
    {"threshold_criterion": "unknown"},
    {"img_size": 32},
    {"dropout": 1.0},
    {"run": "../outside"},
    {"run": "a/b"},
])
def test_invalid_values_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        make_settings(**kwargs)
