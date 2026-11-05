"""Командний рядок: довідка, команди, необов'язкове перевизначення налаштувань."""
import importlib.util
import os
import re
import subprocess
import sys
import typing
from pathlib import Path

import pytest

from cxr_xai import cli
from cxr_xai.settings import Settings


def run_cli(*args, **env):
    """Новий процес: налаштування мають завантажитися вже з перевизначеннями."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith("CXR_")}
    return subprocess.run([sys.executable, "-m", "cxr_xai", *map(str, args)], capture_output=True,
                          encoding="utf-8", errors="replace", env={**clean, "PYTHONUTF8": "1", **env}, check=False)


def test_help_lists_every_command(capsys):
    cli.main(["--help"])
    output = capsys.readouterr().out
    for command in [*cli.COMMANDS, cli.APP_COMMAND]:
        assert command in output


def test_unknown_command_fails(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["no-such-command"])
    assert exc.value.code == 2
    assert "no-such-command" in capsys.readouterr().err


@pytest.mark.parametrize("command", cli.COMMANDS)
def test_command_points_to_existing_module(command):
    module, _ = cli.COMMANDS[command]
    assert importlib.util.find_spec(module) is not None


def test_app_file_exists():
    assert cli.APP_PATH.is_file()


# --------------------------------------------------------------------------- таблиця перевизначень
def test_overrides_cover_every_command():
    assert set(cli.OVERRIDES) == {*cli.COMMANDS, cli.APP_COMMAND}


@pytest.mark.parametrize("command", cli.OVERRIDES)
def test_overrides_name_real_settings(command):
    assert set(cli.OVERRIDES[command]) <= set(Settings.model_fields)


def test_boolean_settings_match_field_types():
    hints = typing.get_type_hints(Settings)
    assert cli.BOOLEAN_SETTINGS == {name for name in Settings.model_fields if hints[name] is bool}


@pytest.mark.parametrize("command", cli.COMMANDS)
def test_settings_read_by_command_can_be_overridden(command):
    """Якщо крок починає читати нове налаштування, таблицю OVERRIDES теж треба оновити."""
    spec = importlib.util.find_spec(cli.COMMANDS[command][0])
    source = Path(spec.origin).read_text(encoding="utf-8")
    used = set(re.findall(r"\bsettings\.([a-z_0-9]+)", source)) & set(Settings.model_fields)
    assert used <= set(cli.OVERRIDES[command])


# --------------------------------------------------------------------------- розбір параметрів
def test_apply_overrides_sets_only_given_values(monkeypatch):
    monkeypatch.delenv("CXR_BATCH_SIZE", raising=False)
    monkeypatch.delenv("CXR_DROPOUT", raising=False)
    monkeypatch.setitem(os.environ, "CXR_BATCH_SIZE", "1")   # monkeypatch відновить значення після тесту
    rest = cli.apply_overrides("train", ["--batch-size", "8", "--smoke", "4"])
    assert os.environ["CXR_BATCH_SIZE"] == "8"
    assert "CXR_DROPOUT" not in os.environ
    assert rest == ["--smoke", "4"]


@pytest.mark.parametrize(("flag", "variable", "value"), [
    (["--epochs1", "3"], "CXR_STAGE1_EPOCHS", "3"),
    (["--weights", "none"], "CXR_BACKBONE_WEIGHTS", "none"),
    (["--no-use-class-weights"], "CXR_USE_CLASS_WEIGHTS", "false"),
    (["--mixed-precision"], "CXR_MIXED_PRECISION", "true"),
])
def test_aliases_and_boolean_flags(monkeypatch, flag, variable, value):
    monkeypatch.setitem(os.environ, variable, "placeholder")
    cli.apply_overrides("train", flag)
    assert os.environ[variable] == value


def test_app_keeps_streamlit_options(monkeypatch):
    monkeypatch.setitem(os.environ, "CXR_KB_PATH", "placeholder")
    rest = cli.apply_overrides("app", ["--kb-path", "x.json", "--server.port", "8600"])
    assert rest == ["--server.port", "8600"]


# --------------------------------------------------------------------------- наскрізна поведінка
def test_flag_reaches_settings_and_beats_environment(tmp_path):
    from_env, from_flag = tmp_path / "from_env", tmp_path / "from_flag"
    result = run_cli("extract", "--npz-dir", from_flag, "--output-dir", tmp_path / "out",
                     "--data-root", tmp_path / "data", CXR_NPZ_DIR=str(from_env))
    assert result.returncode == 1
    assert "from_flag" in result.stderr


def test_environment_is_used_without_flag(tmp_path):
    from_env = tmp_path / "from_env"
    result = run_cli("extract", "--output-dir", tmp_path / "out", "--data-root", tmp_path / "data",
                     CXR_NPZ_DIR=str(from_env))
    assert result.returncode == 1
    assert "from_env" in result.stderr


def test_invalid_override_gives_clean_error():
    result = run_cli("split", "--seed", "not-a-number")
    assert result.returncode == 2
    assert "seed" in result.stderr
    assert "Traceback" not in result.stderr


def test_help_shows_command_and_override_options():
    result = run_cli("split", "--help")
    assert result.returncode == 0
    assert "cxr-xai split" in result.stdout
    assert "--split-ratios" in result.stdout
