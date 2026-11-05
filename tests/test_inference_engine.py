"""Тести вирішувача: спрацювання правил R1-R5 на контрольних векторах оцінок."""
import numpy as np
import pytest

from cxr_xai.knowledge.inference_engine import InferenceEngine
from cxr_xai.knowledge.knowledge_base import DECISION_FRAME, KnowledgeBase
from cxr_xai.settings import settings

# Порядок виходів: 0 COVID-19, 1 Lung Opacity, 2 Normal, 3 Viral Pneumonia, 4 Tuberculosis


@pytest.fixture
def engine():
    kb = KnowledgeBase.load(settings.kb_path).copy()
    kb.set_slot(DECISION_FRAME, "T", 0.60)
    kb.set_slot(DECISION_FRAME, "delta", 0.15)
    kb.set_slot("COVID-19", "typical_confusions",
                [{"with": "Viral Pneumonia", "p_true_given_pred": 0.08, "p_pred_given_true": 0.05}])
    for c in ("Lung Opacity", "Normal", "Viral Pneumonia", "Tuberculosis"):
        kb.set_slot(c, "typical_confusions", [])
    return InferenceEngine(kb)


def fired(decision):
    return set(decision.fired_rules)


def test_confident_single_class(engine):
    d = engine.infer([0.02, 0.03, 0.02, 0.03, 0.90])
    assert d.top1 == "Tuberculosis"
    assert d.status == "confident"
    assert d.shown_classes == ["Tuberculosis"]
    assert fired(d) == {"R1", "R5"}
    assert d.warnings == []
    assert d.description["model_output_index"] == 4


def test_low_confidence_and_two_classes(engine):
    d = engine.infer([0.40, 0.10, 0.05, 0.35, 0.10])
    assert d.status == "low_confidence"
    assert d.shown_classes == ["COVID-19", "Viral Pneumonia"]
    assert fired(d) == {"R1", "R2", "R3", "R4", "R5"}
    assert any("Вірусна пневмонія" in w for w in d.warnings)        # R4 згадує клас плутанини
    assert "належить до цього списку" in d.trace[3].explanation  # top-2 серед плутанин


def test_margin_rule_without_low_confidence(engine):
    d = engine.infer([0.01, 0.62, 0.35, 0.01, 0.01])
    assert d.status == "confident"                # 0.62 >= T
    assert fired(d) == {"R1", "R5"}               # 0.27 >= δ
    d2 = engine.infer([0.01, 0.52, 0.45, 0.01, 0.01])
    assert fired(d2) == {"R1", "R2", "R3", "R5"}


def test_confusion_warning_for_confident_class(engine):
    d = engine.infer([0.95, 0.02, 0.01, 0.01, 0.01])
    assert fired(d) == {"R1", "R4", "R5"}
    assert d.status == "confident"


def test_trace_contains_all_rules_in_order(engine):
    d = engine.infer([0.2, 0.2, 0.2, 0.2, 0.2])
    assert [t.rule_id for t in d.trace] == ["R1", "R2", "R3", "R4", "R5"]
    assert all(t.explanation for t in d.trace)


def test_threshold_boundary(engine):
    d = engine.infer([0.60, 0.10, 0.10, 0.10, 0.10])   # p1 == T -> R2 не спрацьовує
    assert "R2" not in fired(d)


@pytest.mark.parametrize("bad", [
    [0.5, 0.5, 0.0, 0.0],                     # неправильна довжина
    [0.7, 0.4, -0.1, 0.0, 0.0],               # від'ємне значення
    [0.5, 0.5, 0.5, 0.0, 0.0],                # сума != 1
    [np.nan, 0.5, 0.5, 0.0, 0.0],             # NaN
])
def test_invalid_input(engine, bad):
    with pytest.raises(ValueError):
        engine.infer(bad)


def test_text_report(engine):
    d = engine.infer([0.40, 0.10, 0.05, 0.35, 0.10])
    text = d.to_text(engine.kb)
    assert "ЗАСТЕРЕЖЕННЯ" in text and "R3" in text and "низька впевненість" in text
