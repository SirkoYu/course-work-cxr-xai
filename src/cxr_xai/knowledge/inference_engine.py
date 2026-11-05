"""
Вирішувач (машина логічного висновку) над вектором softmax-оцінок.

Правила зберігаються в базі знань (DecisionFrame.rules) у декларативній формі
«ЯКЩО <ознака> <оператор> <параметр/значення> ТО <дія>» і виконуються
послідовно (пряме виведення, forward chaining) у порядку R1 -> R5:

  R1  завжди                    -> обрати клас з максимальною оцінкою;
  R2  p1 < T                    -> позначити «низька впевненість»;
  R3  p1 - p2 < δ               -> показати два класи;
  R4  у класу є типові плутанини -> додати попередження;
  R5  завжди                    -> вивести опис класу і застереження.

Для кожного правила фіксується, чи спрацювало воно, і текстове пояснення -
це трасування є основою підсистеми пояснень.
Модуль не залежить від TensorFlow.
"""
from __future__ import annotations

import operator
from dataclasses import asdict, dataclass, field

import numpy as np

from cxr_xai.knowledge.knowledge_base import ACTIONS, KnowledgeBase, KnowledgeBaseError

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt,
        ">=": operator.ge, "==": operator.eq, "!=": operator.ne}


class _SafeDict(dict):
    """Підстановка в шаблон пояснення: невідомі поля лишаються як є."""

    def __missing__(self, key):
        return "{" + key + "}"


@dataclass
class RuleTrace:
    rule_id: str
    name: str
    fired: bool
    explanation: str


@dataclass
class Decision:
    top1: str
    top2: str
    p1: float
    p2: float
    margin: float
    status: str                          # "confident" | "low_confidence"
    shown_classes: list[str]
    warnings: list[str]
    ranking: list[tuple[str, float]]     # усі класи за спаданням оцінки
    description: dict = field(default_factory=dict)
    trace: list[RuleTrace] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def fired_rules(self) -> list[str]:
        return [t.rule_id for t in self.trace if t.fired]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self, kb: KnowledgeBase | None = None) -> str:
        """Текстовий звіт для збереження користувачем."""
        name = (lambda c: f"{kb.display_name(c)} ({c})") if kb else (lambda c: c)
        lines = ["РЕЗУЛЬТАТ АНАЛІЗУ РЕНТГЕНОГРАМИ (навчально-дослідницький прототип)", ""]
        status = "низька впевненість" if self.status == "low_confidence" else "достатня впевненість"
        lines.append("Показані класи: " + "; ".join(name(c) for c in self.shown_classes))
        lines.append(f"Статус: {status}")
        lines.append(f"Параметри вирішувача: T = {self.params.get('T'):.2f}, "
                     f"δ = {self.params.get('delta'):.2f}"
                     + ("" if self.params.get("tuned") else " (значення за замовчуванням, не підібрані)"))
        lines.append("")
        lines.append("Оцінки softmax (не є каліброваними ймовірностями):")
        lines += [f"  {name(c)}: {p:.4f}" for c, p in self.ranking]
        if self.warnings:
            lines += ["", "Попередження:"] + [f"  - {w}" for w in self.warnings]
        if self.description:
            lines += ["", f"Опис класу «{name(self.top1)}»:", f"  {self.description.get('description', '')}"]
            rules = (self.description.get("interpretation_rules") or []) + \
                    (self.description.get("general_rules") or [])
            if rules:
                lines += ["", "Правила інтерпретації:"] + [f"  [{r['id']}] {r['text']}" for r in rules]
        lines += ["", "Трасування правил вирішувача:"]
        for t in self.trace:
            lines.append(f"  {t.rule_id} ({t.name}) - {'спрацювало' if t.fired else 'не спрацювало'}: "
                         f"{t.explanation}")
        if self.description.get("disclaimer"):
            lines += ["", "ЗАСТЕРЕЖЕННЯ: " + self.description["disclaimer"]]
        return "\n".join(lines)


class InferenceEngine:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.index_to_class = kb.index_to_class()
        self.num_classes = len(self.index_to_class)
        self.params = kb.decision_params()
        self.rules = kb.rules()
        for r in self.rules:
            if r.get("then") not in ACTIONS:
                raise KnowledgeBaseError(f"Правило {r.get('id')}: невідома дія «{r.get('then')}»")

    # ------------------------------------------------------------------ ознаки
    def _check_probs(self, probs) -> np.ndarray:
        p = np.asarray(probs, dtype=float).reshape(-1)
        if p.shape[0] != self.num_classes:
            raise ValueError(f"Очікується вектор довжини {self.num_classes}, отримано {p.shape[0]}")
        if not np.all(np.isfinite(p)) or np.any(p < -1e-6):
            raise ValueError("Вектор оцінок містить від'ємні або нескінченні значення")
        if abs(p.sum() - 1.0) > 1e-3:
            raise ValueError(f"Сума оцінок softmax має дорівнювати 1, отримано {p.sum():.4f}")
        return np.clip(p, 0.0, 1.0)

    def features(self, probs) -> dict:
        p = self._check_probs(probs)
        order = np.argsort(-p, kind="stable")
        top1, top2 = self.index_to_class[int(order[0])], self.index_to_class[int(order[1])]
        confusions = self.kb.get_slot(top1, "typical_confusions") or []
        conf_names = [c["with"] for c in confusions]
        return {
            "probs": p, "order": order,
            "top1": top1, "top2": top2,
            "p1": float(p[order[0]]), "p2": float(p[order[1]]),
            "margin": float(p[order[0]] - p[order[1]]),
            "confusions": confusions,
            "n_confusions": len(confusions),
            "top2_in_confusions": top2 in conf_names,
        }

    def _condition(self, cond: dict, feats: dict) -> bool:
        if cond.get("always"):
            return True
        left = feats[cond["feature"]]
        right = self.params[cond["param"]] if "param" in cond else cond["value"]
        return bool(_OPS[cond["op"]](left, right))

    def _template_vars(self, feats: dict) -> _SafeDict:
        dn = self.kb.display_name
        conf_text = ", ".join(
            f"«{dn(c['with'])}» ({c.get('p_true_given_pred', 0):.0%} / {c.get('p_pred_given_true', 0):.0%})"
            for c in feats["confusions"]) or "—"
        top2_note = (f" Друга за оцінкою альтернатива «{dn(feats['top2'])}» належить до цього списку."
                     if feats["top2_in_confusions"] else "")
        return _SafeDict(top1=dn(feats["top1"]), top2=dn(feats["top2"]), top1_frame=feats["top1"],
                         p1=feats["p1"], p2=feats["p2"], margin=feats["margin"],
                         T=self.params["T"], delta=self.params["delta"],
                         confusions=conf_text, top2_note=top2_note)

    # ------------------------------------------------------------------ виведення
    def infer(self, probs) -> Decision:
        feats = self.features(probs)
        tv = self._template_vars(feats)
        ranking = [(self.index_to_class[int(i)], float(feats["probs"][i])) for i in feats["order"]]
        decision = Decision(top1=feats["top1"], top2=feats["top2"], p1=feats["p1"], p2=feats["p2"],
                            margin=feats["margin"], status="confident", shown_classes=[],
                            warnings=[], ranking=ranking, params=dict(self.params))
        for rule in self.rules:
            fired = self._condition(rule.get("if", {}), feats)
            if fired:
                self._apply(rule, feats, decision, tv)
                text = rule.get("explain", "").format_map(tv)
            else:
                text = rule.get("explain_not_fired", "Умову не виконано.").format_map(tv)
            decision.trace.append(RuleTrace(rule["id"], rule.get("name", ""), fired, text))
        if not decision.shown_classes:            # захист на випадок відсутності R1 у БЗ
            decision.shown_classes = [feats["top1"]]
        return decision

    def _apply(self, rule: dict, feats: dict, d: Decision, tv: _SafeDict) -> None:
        action = rule["then"]
        message = rule.get("message", "").format_map(tv)
        if action == "select_top1":
            d.shown_classes = [feats["top1"]]
        elif action == "flag_low_confidence":
            d.status = "low_confidence"
            if message:
                d.warnings.append(message)
        elif action == "show_top2":
            d.shown_classes = [feats["top1"], feats["top2"]]
            if message:
                d.warnings.append(message)
        elif action == "warn_confusion":
            if message:
                d.warnings.append(message)
        elif action == "attach_description":
            d.description = self.kb.describe(feats["top1"])
