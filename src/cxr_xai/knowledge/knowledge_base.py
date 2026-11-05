"""
База знань (БЗ) на основі фреймової моделі.

Фрейм = іменована структура зі слотами. Кожен слот має фасети:
  * value   - значення слота у цьому фреймі;
  * default - значення за замовчуванням (використовується, якщо value = null);
  * type, required - опис типу і обов'язковості (задаються у фреймі-прототипі).
Фрейми пов'язані відношенням is_a («є різновидом»): якщо слот не заповнено
у фреймі-екземплярі, його значення успадковується від батьківського фрейму.

Структура БЗ (knowledge_base.json):
  CXRClass       - фрейм-прототип класу рентгенограм (склад слотів, значення за замовчуванням);
  COVID-19, ...  - фрейми-екземпляри класів (is_a = CXRClass);
  DecisionFrame  - керуючий фрейм вирішувача (пороги T, δ і правила R1-R5).

Перевірка БЗ з командного рядка (замінник «редактора БЗ»: БЗ редагується як
JSON-файл, а цей валідатор гарантує її коректність і узгодженість із моделлю):
    cxr-xai kb --validate
    cxr-xai kb --describe COVID-19
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from cxr_xai import constants
from cxr_xai.settings import settings

PROTOTYPE = "CXRClass"
DECISION_FRAME = "DecisionFrame"

# Словник вирішувача: дозволені ознаки, оператори та дії правил.
FEATURES = {"p1", "p2", "margin", "n_confusions", "top2_in_confusions"}
OPS = {"<", "<=", ">", ">=", "==", "!="}
ACTIONS = {"select_top1", "flag_low_confidence", "show_top2", "warn_confusion", "attach_description"}

_TYPES = {"str": str, "int": int, "float": (int, float), "list": list, "dict": dict, "bool": bool}


class KnowledgeBaseError(Exception):
    pass


class KnowledgeBase:
    def __init__(self, data: dict, path: Path | None = None):
        self.data = data
        self.path = Path(path) if path else None
        self.frames: dict = data.get("frames", {})

    # ------------------------------------------------------------------ I/O
    @classmethod
    def load(cls, path) -> KnowledgeBase:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f), path)

    def save(self, path=None) -> None:
        path = Path(path or self.path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def copy(self) -> KnowledgeBase:
        return KnowledgeBase(copy.deepcopy(self.data), self.path)

    # ------------------------------------------------------- доступ до слотів
    def get_slot(self, frame: str, slot: str) -> Any:
        """Значення слота з урахуванням фасет value/default та успадкування is_a."""
        return self.resolve_slot(frame, slot)[0]

    def resolve_slot(self, frame: str, slot: str, _visited=None) -> tuple[Any, str, str]:
        """Повертає (значення, фрейм-джерело, фасет) - використовується підсистемою пояснень."""
        _visited = _visited or set()
        if frame in _visited:
            raise KnowledgeBaseError(f"Циклічне успадкування is_a через фрейм «{frame}»")
        _visited.add(frame)
        if frame not in self.frames:
            raise KnowledgeBaseError(f"Фрейм «{frame}» відсутній у базі знань")
        f = self.frames[frame]
        s = f.get("slots", {}).get(slot)
        if s is not None:
            if s.get("value") is not None:
                return s["value"], frame, "value"
            if "default" in s:
                return s["default"], frame, "default"
        parent = f.get("is_a")
        if parent:
            return self.resolve_slot(parent, slot, _visited)
        raise KnowledgeBaseError(f"Слот «{slot}» не визначено у фреймі «{frame}» та його предках")

    def set_slot(self, frame: str, slot: str, value: Any) -> None:
        self.frames[frame].setdefault("slots", {}).setdefault(slot, {})["value"] = value

    # ------------------------------------------------------------- класи
    def class_frames(self) -> list[str]:
        return [name for name, f in self.frames.items() if f.get("kind") == "class"]

    def index_to_class(self) -> dict[int, str]:
        return {int(self.get_slot(c, "model_output_index")): c for c in self.class_frames()}

    def class_by_index(self, index: int) -> str:
        mapping = self.index_to_class()
        if index not in mapping:
            raise KnowledgeBaseError(f"Немає фрейму для виходу моделі з індексом {index}")
        return mapping[index]

    def ordered_class_names(self) -> list[str]:
        mapping = self.index_to_class()
        return [mapping[i] for i in sorted(mapping)]

    def describe(self, class_frame: str) -> dict:
        """Отримання опису класу з БЗ (використовується вирішувачем і інтерфейсом)."""
        return {
            "class_name": self.get_slot(class_frame, "class_name"),
            "display_name": self.get_slot(class_frame, "display_name"),
            "model_output_index": self.get_slot(class_frame, "model_output_index"),
            "description": self.get_slot(class_frame, "description"),
            "typical_confusions": self.get_slot(class_frame, "typical_confusions"),
            "interpretation_rules": self.get_slot(class_frame, "interpretation_rules"),
            "general_rules": self.get_slot(class_frame, "general_rules"),
            "disclaimer": self.get_slot(class_frame, "disclaimer"),
        }

    def display_name(self, class_frame: str) -> str:
        return self.get_slot(class_frame, "display_name")

    # ------------------------------------------------------------- вирішувач
    def decision_params(self) -> dict:
        t, _, t_facet = self.resolve_slot(DECISION_FRAME, "T")
        d, _, d_facet = self.resolve_slot(DECISION_FRAME, "delta")
        return {"T": float(t), "delta": float(d),
                "tuned": t_facet == "value" and d_facet == "value",
                "tuning": self.get_slot(DECISION_FRAME, "tuning")}

    def rules(self) -> list[dict]:
        return self.get_slot(DECISION_FRAME, "rules")

    # ------------------------------------------------------------- перевірка
    def validate(self, class_names: list[str] | None = None, num_outputs: int | None = None) -> list[str]:
        """
        Перевіряє цілісність БЗ і відповідність її моделі. Повертає список помилок
        (порожній - БЗ коректна):
          * існування батьківських фреймів, відсутність циклів is_a;
          * наявність обов'язкових слотів і їх типи;
          * індекси виходів 0..K-1 без пропусків і повторів;
          * відповідність class_names.json моделі (порядок виходів) фреймам БЗ;
          * кількість виходів моделі = кількості фреймів-класів;
          * коректність посилань у типових плутанинах;
          * допустимі значення T, δ і синтаксис правил вирішувача.
        """
        errors: list[str] = []
        for name, f in self.frames.items():
            parent = f.get("is_a")
            if parent and parent not in self.frames:
                errors.append(f"Фрейм «{name}»: батьківський фрейм «{parent}» відсутній")
        if PROTOTYPE not in self.frames:
            return errors + [f"Відсутній фрейм-прототип «{PROTOTYPE}»"]
        if DECISION_FRAME not in self.frames:
            errors.append(f"Відсутній керуючий фрейм «{DECISION_FRAME}»")

        proto_slots = self.frames[PROTOTYPE].get("slots", {})
        classes = self.class_frames()
        if not classes:
            return errors + ["У БЗ немає жодного фрейму-класу (kind = class)"]

        indices = {}
        for c in classes:
            for slot, facets in proto_slots.items():
                try:
                    value = self.get_slot(c, slot)
                except KnowledgeBaseError as e:
                    if facets.get("required"):
                        errors.append(f"Фрейм «{c}»: {e}")
                    continue
                expected = _TYPES.get(facets.get("type", ""), object)
                if value is not None and not isinstance(value, expected) or \
                        (facets.get("type") == "int" and isinstance(value, bool)):
                    errors.append(f"Фрейм «{c}», слот «{slot}»: очікується тип {facets.get('type')}")
            try:
                if self.get_slot(c, "class_name") != c:
                    errors.append(f"Фрейм «{c}»: слот class_name має збігатися з назвою фрейму")
                idx = self.get_slot(c, "model_output_index")
                if isinstance(idx, int):
                    indices.setdefault(idx, []).append(c)
            except KnowledgeBaseError:
                pass

        k = len(classes)
        dup = {i: cs for i, cs in indices.items() if len(cs) > 1}
        if dup:
            errors.append(f"Повторювані індекси виходів: {dup}")
        if sorted(indices) != list(range(k)):
            errors.append(f"Індекси виходів мають бути 0..{k - 1} без пропусків, отримано {sorted(indices)}")

        if num_outputs is not None and num_outputs != k:
            errors.append(f"Модель має {num_outputs} виходів, а БЗ містить {k} фреймів-класів")
        if class_names is not None:
            if len(class_names) != k:
                errors.append(f"class_names містить {len(class_names)} назв, а БЗ - {k} класів")
            else:
                for i, name in enumerate(class_names):
                    kb_name = indices.get(i, [None])[0]
                    if kb_name != name:
                        errors.append(f"Вихід {i}: у моделі «{name}», у БЗ «{kb_name}»")

        for c in classes:
            try:
                conf = self.get_slot(c, "typical_confusions") or []
            except KnowledgeBaseError:
                continue
            for item in conf:
                other = item.get("with") if isinstance(item, dict) else None
                if other not in classes or other == c:
                    errors.append(f"Фрейм «{c}»: некоректне посилання в typical_confusions: {item}")

        if DECISION_FRAME in self.frames:
            errors += self._validate_decision_frame()
        return errors

    def _validate_decision_frame(self) -> list[str]:
        errors = []
        try:
            p = self.decision_params()
            if not 0.0 < p["T"] <= 1.0:
                errors.append(f"T має бути в (0; 1], отримано {p['T']}")
            if not 0.0 <= p["delta"] < 1.0:
                errors.append(f"δ має бути в [0; 1), отримано {p['delta']}")
        except (KnowledgeBaseError, TypeError, ValueError) as e:
            errors.append(f"Параметри вирішувача: {e}")
        try:
            rules = self.rules()
        except KnowledgeBaseError as e:
            return errors + [str(e)]
        ids = [r.get("id") for r in rules]
        if len(ids) != len(set(ids)):
            errors.append(f"Повторювані ідентифікатори правил: {ids}")
        for r in rules:
            rid = r.get("id", "?")
            if r.get("then") not in ACTIONS:
                errors.append(f"Правило {rid}: невідома дія «{r.get('then')}»")
            cond = r.get("if", {})
            if cond.get("always") is True:
                continue
            if cond.get("feature") not in FEATURES:
                errors.append(f"Правило {rid}: невідома ознака «{cond.get('feature')}»")
            if cond.get("op") not in OPS:
                errors.append(f"Правило {rid}: невідомий оператор «{cond.get('op')}»")
            if ("param" in cond) == ("value" in cond):
                errors.append(f"Правило {rid}: умова має містити або param, або value")
            elif "param" in cond:
                try:
                    self.get_slot(DECISION_FRAME, cond["param"])
                except KnowledgeBaseError:
                    errors.append(f"Правило {rid}: параметр «{cond['param']}» відсутній у DecisionFrame")
        return errors


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Перевірка та перегляд бази знань")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--model", default=None, help="шлях до .keras для перевірки кількості виходів")
    parser.add_argument("--describe", default=None, help="назва класу, напр. COVID-19")
    args = parser.parse_args(argv)

    kb = KnowledgeBase.load(settings.kb_path)
    if args.describe:
        print(json.dumps(kb.describe(args.describe), ensure_ascii=False, indent=2))
    if args.validate or not args.describe:
        class_names = None
        if settings.class_names_json.exists():
            with open(settings.class_names_json, encoding="utf-8") as f:
                class_names = json.load(f)
            print(f"Порядок виходів моделі (class_names.json): {class_names}")
        else:
            class_names = constants.CLASS_NAMES
            print(f"class_names.json ще немає - перевірка за constants.CLASS_NAMES: {class_names}")
        num_outputs = None
        if args.model:
            import keras
            num_outputs = int(keras.models.load_model(args.model, compile=False).output_shape[-1])
            print(f"Кількість виходів моделі: {num_outputs}")
        errors = kb.validate(class_names=class_names, num_outputs=num_outputs)
        print("\nВідповідність «індекс виходу -> фрейм»:")
        for i, c in sorted(kb.index_to_class().items()):
            print(f"  {i}: {c} ({kb.display_name(c)})")
        p = kb.decision_params()
        print(f"\nT = {p['T']}, δ = {p['delta']}, підібрано за валідацією: {p['tuned']}")
        if errors:
            print("\nБЗ НЕКОРЕКТНА:")
            for e in errors:
                print("  -", e)
            raise SystemExit(1)
        print("\nБЗ коректна і узгоджена з порядком виходів моделі.")


if __name__ == "__main__":
    main()
