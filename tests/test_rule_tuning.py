"""Добір порогів вирішувача та типові плутанини."""
import numpy as np

from cxr_xai.knowledge import rule_tuning


def test_threshold_selection_separates_errors():
    rng = np.random.default_rng(0)
    correct = np.r_[np.ones(900, bool), np.zeros(100, bool)]
    score = np.where(correct, rng.uniform(0.7, 1.0, 1000), rng.uniform(0.3, 0.6, 1000))
    tab = rule_tuning.threshold_table(score, correct, rule_tuning.grid((0.3, 0.99, 0.01)))
    t, why = rule_tuning.choose_threshold(tab, "youden", max_flag_rate=0.3, target_accuracy=0.95)
    assert 0.6 <= t <= 0.71
    assert "Юдена" in why


def test_threshold_respects_max_flag_rate():
    score = np.linspace(0, 1, 100)
    correct = score > 0.5
    tab = rule_tuning.threshold_table(score, correct, rule_tuning.grid((0.0, 1.0, 0.05)))
    t, _ = rule_tuning.choose_threshold(tab, "youden", max_flag_rate=0.2, target_accuracy=0.95)
    assert t <= 0.2 + 1e-9


def test_typical_confusions_directions():
    names = ["A", "B", "C"]
    cm = np.array([[90, 10, 0],
                   [20, 80, 0],
                   [0, 0, 100]])
    conf = rule_tuning.typical_confusions(cm, names, min_rate=0.05)
    a_b = conf["A"][0]
    assert a_b["with"] == "B"
    assert a_b["p_true_given_pred"] == round(20 / 110, 4)   # прогноз A, насправді B
    assert a_b["p_pred_given_true"] == 0.1                   # справжній A, прогноз B
    assert conf["C"] == []
