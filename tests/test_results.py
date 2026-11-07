"""Структура результатів: каталоги запусків, вибір запуску, захист від перезапису."""
import json

import pytest

from cxr_xai import results
from cxr_xai.settings import Settings


@pytest.fixture
def use_settings(monkeypatch, tmp_path):
    """Підміняє налаштування модуля results: результати потрапляють у тимчасовий каталог."""
    def apply(**kwargs):
        monkeypatch.setattr(results, "settings", Settings(_env_file=None, output_dir=tmp_path, **kwargs))
    apply()
    return apply


def test_dataset_results_are_grouped_by_step(tmp_path, use_settings):
    ds = results.dataset()
    ds.prepare()
    assert ds.inventory_csv == tmp_path / "dataset" / "check" / "inventory.csv"
    assert ds.split_csv == tmp_path / "dataset" / "split" / "split.csv"
    assert ds.dedup_dir.is_dir()


def test_create_run_saves_settings_snapshot(use_settings):
    use_settings(batch_size=8)
    run = results.create_run()
    assert run.training_dir.is_dir() and run.gradcam_dir.is_dir()
    assert json.loads(run.settings_json.read_text(encoding="utf-8"))["batch_size"] == 8


def test_default_name_shows_kind_of_run(use_settings):
    assert results.default_run_name(smoke=True).endswith("_xception_299_smoke")
    assert results.default_run_name().endswith("_xception_299")


def test_existing_run_is_never_overwritten(use_settings):
    use_settings(run="experiment")
    results.create_run()
    with pytest.raises(SystemExit):
        results.create_run()


def test_no_run_gives_helpful_exit(use_settings):
    assert results.find_run() is None
    with pytest.raises(SystemExit, match="cxr-xai train"):
        results.require_run()


def test_unfinished_run_is_not_selected_by_default(use_settings):
    results.create_run()
    assert results.find_run() is None


def test_latest_run_is_selected_by_default(use_settings):
    use_settings(run="first")
    first = results.create_run()
    results.mark_latest(first)
    use_settings()
    assert results.find_run() == first


def test_explicit_run_wins_over_latest(use_settings):
    use_settings(run="first")
    first = results.create_run()
    use_settings(run="second")
    results.mark_latest(results.create_run())
    use_settings(run="first")
    assert results.require_run() == first


def test_unknown_explicit_run_fails(use_settings):
    use_settings(run="missing")
    assert results.find_run() is None
    with pytest.raises(SystemExit, match="missing"):
        results.require_run()


def test_test_evaluation_is_tracked_across_runs(use_settings):
    use_settings(run="first")
    first = results.create_run()
    use_settings(run="second")
    second = results.create_run()
    assert results.runs_with_evaluated_test() == []
    first.test_evaluated_file.write_text("done", encoding="utf-8")
    assert results.runs_with_evaluated_test() == [first]
    assert second not in results.runs_with_evaluated_test()
