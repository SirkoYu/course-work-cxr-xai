"""Тести бази знань: цілісність, успадкування слотів, відповідність індексів виходам моделі."""
import pytest

from cxr_xai import constants
from cxr_xai.knowledge.knowledge_base import DECISION_FRAME, KnowledgeBase, KnowledgeBaseError
from cxr_xai.settings import settings


@pytest.fixture
def kb():
    return KnowledgeBase.load(settings.kb_path)


def test_kb_is_valid_and_matches_class_order(kb):
    assert kb.validate(class_names=constants.CLASS_NAMES, num_outputs=len(constants.CLASS_NAMES)) == []


def test_index_mapping(kb):
    for i, name in enumerate(constants.CLASS_NAMES):
        assert kb.class_by_index(i) == name
        assert kb.get_slot(name, "model_output_index") == i


def test_slot_inheritance_from_prototype(kb):
    value, source, facet = kb.resolve_slot("COVID-19", "disclaimer")
    assert source == "CXRClass" and facet == "value"
    assert value == kb.get_slot("CXRClass", "disclaimer")
    # typical_confusions не заповнено -> береться default прототипу
    fresh = kb.copy()
    fresh.frames["COVID-19"]["slots"].pop("typical_confusions", None)
    assert fresh.get_slot("COVID-19", "typical_confusions") == []


def test_describe_returns_all_slots(kb):
    d = kb.describe("Tuberculosis")
    assert d["model_output_index"] == 4
    assert d["description"] and d["disclaimer"]
    assert {r["id"] for r in d["general_rules"]} == {"G1", "G2", "G3"}


def test_detects_wrong_class_order(kb):
    wrong = list(reversed(constants.CLASS_NAMES))
    errors = kb.validate(class_names=wrong)
    assert any("Вихід 0" in e for e in errors)


def test_detects_wrong_number_of_outputs(kb):
    assert kb.validate(num_outputs=4)


def test_detects_duplicate_index(kb):
    bad = kb.copy()
    bad.set_slot("Normal", "model_output_index", 0)
    errors = bad.validate()
    assert any("Повторювані індекси" in e for e in errors)


def test_detects_bad_confusion_reference(kb):
    bad = kb.copy()
    bad.set_slot("Normal", "typical_confusions", [{"with": "Unknown class"}])
    assert any("typical_confusions" in e for e in bad.validate())


def test_detects_unknown_rule_action(kb):
    bad = kb.copy()
    bad.get_slot(DECISION_FRAME, "rules")[0]["then"] = "delete_everything"
    assert any("невідома дія" in e for e in bad.validate())


def test_default_params_until_tuned(kb):
    fresh = kb.copy()
    fresh.set_slot(DECISION_FRAME, "T", None)
    fresh.set_slot(DECISION_FRAME, "delta", None)
    p = fresh.decision_params()
    assert p["tuned"] is False and p["T"] == 0.5 and p["delta"] == 0.1


def test_missing_frame_raises(kb):
    with pytest.raises(KnowledgeBaseError):
        kb.get_slot("Unknown", "description")
