"""
Інтерфейс користувача СШІ (Streamlit, графічний діалог).

Послідовність роботи користувача:
  1) завантажити рентгенограму (PNG / JPG / BMP);
  2) натиснути «Аналізувати»;
  3) переглянути результат: клас(и) моделі, статус впевненості, попередження,
     оцінки softmax, теплову карту Grad-CAM, опис класу з бази знань,
     трасування правил вирішувача; за потреби завантажити текстовий звіт.

Запуск:
    cxr-xai app
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from cxr_xai import constants, results
from cxr_xai.knowledge.inference_engine import Decision, InferenceEngine
from cxr_xai.knowledge.knowledge_base import KnowledgeBase, KnowledgeBaseError
from cxr_xai.settings import settings

APP_TITLE = "Класифікація рентгенограм органів грудної клітки"
DISCLAIMER = (
    "**Навчально-дослідницький прототип.** Система не встановлює клінічний діагноз і не "
    "призначена для медичної практики. Класи відповідають розмітці навчального датасету, "
    "а оцінки softmax не є каліброваними ймовірностями."
)
COLORS = {"primary": "#1f5f8b", "second": "#7fb3d5", "other": "#c8ced3", "threshold": "#c0392b"}


class UserInputError(Exception):
    """Помилка вхідних даних, зрозуміла користувачеві."""


# --------------------------------------------------------------------------- ресурси
@st.cache_resource(show_spinner="Завантаження моделі…")
def load_model(path: str, mtime: float):
    import keras  # імпорт тут: інтерфейс показує повідомлення ще до завантаження TF
    return keras.models.load_model(path, compile=False)


@st.cache_resource
def load_kb(path: str, mtime: float) -> KnowledgeBase:
    return KnowledgeBase.load(path)


def load_resources():
    """Завантажує модель і БЗ та перевіряє їх узгодженість. Повертає (run, model, kb, engine)."""
    run = results.find_run()
    if run is None or not run.final_model.exists():
        st.error(f"Навчену модель не знайдено в `{results.runs_dir()}`.\n\n"
                 "Спочатку навчіть модель (`cxr-xai train`) або скопіюйте каталог запуску "
                 "разом із файлом `latest_run.txt` у каталог `outputs/runs/`.", icon="🚫")
        st.stop()
    kb_file = run.kb_path if run.kb_path.exists() else settings.kb_path   # шаблон, якщо T і δ ще не підібрано
    try:
        kb = load_kb(str(kb_file), kb_file.stat().st_mtime)
    except (OSError, json.JSONDecodeError) as e:
        st.error(f"Не вдалося прочитати базу знань `{kb_file.name}`: {e}", icon="🚫")
        st.stop()
    model = load_model(str(run.final_model), run.final_model.stat().st_mtime)

    class_names = None
    if run.class_names_json.exists():
        class_names = json.loads(run.class_names_json.read_text(encoding="utf-8"))
    errors = kb.validate(class_names=class_names, num_outputs=int(model.output_shape[-1]))
    if errors:
        st.error("База знань не узгоджена з моделлю, аналіз неможливий:\n\n- " + "\n- ".join(errors), icon="🚫")
        st.stop()
    try:
        engine = InferenceEngine(kb)
    except KnowledgeBaseError as e:
        st.error(f"Помилка в правилах вирішувача: {e}", icon="🚫")
        st.stop()
    return run, model, kb, engine


# --------------------------------------------------------------------------- обробка
def read_upload(uploaded) -> tuple[bytes, Image.Image]:
    raw = uploaded.getvalue()
    if len(raw) == 0:
        raise UserInputError("Файл порожній.")
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise UserInputError(f"Файл завеликий (понад {settings.max_upload_mb} МБ).")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise UserInputError("Не вдалося відкрити файл як зображення. Перевірте, що це PNG, JPG або BMP, "
                             "і файл не пошкоджений.") from e
    if pil.format not in constants.SUPPORTED_PIL_FORMATS:
        raise UserInputError(f"Формат {pil.format} не підтримується. Використайте PNG, JPG або BMP.")
    if min(pil.size) < settings.min_image_side:
        raise UserInputError(f"Зображення замале ({pil.width}×{pil.height}); "
                             f"мінімальна сторона - {settings.min_image_side} пікселів.")
    return raw, pil


def to_display_rgb(pil: Image.Image) -> np.ndarray:
    """Зображення для показу: RGB, 8 біт, довша сторона не більше DISPLAY_MAX_SIDE."""
    from cxr_xai.data.deduplication import to_grayscale_8bit
    im = pil if pil.mode in ("RGB", "L") else (to_grayscale_8bit(pil) if pil.mode.startswith("I")
                                                 else pil.convert("RGB"))
    im = im.convert("RGB")
    im.thumbnail((settings.display_max_side, settings.display_max_side))
    return np.asarray(im)


def looks_like_color_photo(rgb: np.ndarray) -> bool:
    """Рентгенограма - зображення у градаціях сірого; висока насиченість кольору - ознака стороннього фото."""
    arr = rgb.astype(np.float32)
    saturation = arr.max(axis=-1) - arr.min(axis=-1)
    return float(saturation.mean()) > 25.0


def analyze(model, engine: InferenceEngine, raw: bytes) -> dict:
    """Повний цикл: попередня обробка -> модель -> Grad-CAM -> вирішувач."""
    from cxr_xai.data import loading
    from cxr_xai.modeling import gradcam
    x = loading.decode_and_resize(raw, int(model.input_shape[1])).numpy()   # розмір входу моделі
    heatmap1, probs, idx1 = gradcam.compute_gradcam(model, x)
    probs = probs / probs.sum()     # захист від похибок округлення float32
    decision = engine.infer(probs)
    heatmaps = {decision.top1: heatmap1}
    if len(decision.shown_classes) > 1:
        idx2 = {v: k for k, v in engine.index_to_class.items()}[decision.top2]
        heatmaps[decision.top2], _, _ = gradcam.compute_gradcam(model, x, idx2)
    return {"decision": decision, "heatmaps": heatmaps}


# --------------------------------------------------------------------------- відображення
def scores_figure(decision: Decision, kb: KnowledgeBase):
    ranking = list(reversed(decision.ranking))
    names = [kb.display_name(c) for c, _ in ranking]
    values = [p for _, p in ranking]
    colors = [COLORS["primary"] if c == decision.top1 else
              COLORS["second"] if c in decision.shown_classes else COLORS["other"] for c, _ in ranking]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.barh(names, values, color=colors)
    ax.axvline(decision.params["T"], color=COLORS["threshold"], linestyle="--", linewidth=1.2)
    ax.text(decision.params["T"], len(names) - 0.45, f" T = {decision.params['T']:.2f}",
            color=COLORS["threshold"], fontsize=8, va="bottom")
    for i, v in enumerate(values):
        ax.text(min(v + 0.01, 0.9), i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Оцінка softmax (не калібрована ймовірність)", fontsize=8)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def render_sidebar(run: results.RunResults, kb: KnowledgeBase, model) -> float:
    st.sidebar.header("Про систему")
    st.sidebar.markdown(
        "- **Модель:** Xception (ImageNet) + fine-tuning\n"
        "- **База знань:** фреймова, JSON\n"
        "- **Вирішувач:** правила R1–R5 над оцінками softmax\n"
        "- **Пояснення:** трасування правил + Grad-CAM")
    mtime = datetime.fromtimestamp(run.final_model.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    st.sidebar.caption(f"Запуск: `{run.name}` ({mtime}); "
                       f"вхід {int(model.input_shape[1])}×{int(model.input_shape[2])}")
    p = kb.decision_params()
    st.sidebar.caption(f"Версія БЗ: {kb.data['kb_meta'].get('version')}; T = {p['T']:.2f}; δ = {p['delta']:.2f}")
    if not p["tuned"]:
        st.sidebar.warning("T і δ ще не підібрано за валідаційною вибіркою - використано значення "
                           "за замовчуванням (виконайте cxr-xai tune).", icon="⚙️")
    st.sidebar.header("Налаштування показу")
    alpha = st.sidebar.slider("Прозорість теплової карти", 0.1, 0.9, float(settings.gradcam_alpha), 0.05)
    with st.sidebar.expander("Як читати результат"):
        st.markdown(
            "- **Клас моделі** - клас розмітки датасету з найбільшою оцінкою, а не діагноз.\n"
            "- **Низька впевненість** - найбільша оцінка нижча за поріг T.\n"
            "- **Два класи** - різниця двох найбільших оцінок менша за δ.\n"
            "- **Теплова карта** - області, що найбільше вплинули на оцінку (червоне - сильніше); "
            "це не контур ураження.\n"
            "- **Трасування** - які правила спрацювали і чому.")
    return alpha


def render_result(result: dict, rgb: np.ndarray, kb: KnowledgeBase, alpha: float, file_name: str) -> None:
    from cxr_xai.modeling import gradcam
    d: Decision = result["decision"]
    dn = kb.display_name

    st.subheader("3. Результат")
    if len(d.shown_classes) == 1:
        headline = f"Клас моделі: **{dn(d.top1)}** · оцінка {d.p1:.2f}"
    else:
        headline = (f"Два близькі класи: **{dn(d.top1)}** ({d.p1:.2f}) або "
                    f"**{dn(d.top2)}** ({d.p2:.2f})")
    if d.status == "low_confidence":
        st.warning(headline + "  \nСтатус: **низька впевненість**", icon="⚠️")
    else:
        st.info(headline + "  \nСтатус: достатня впевненість (за порогом T)", icon="ℹ️")
    for w in d.warnings:
        st.warning(w, icon="⚠️")

    col_img, col_info = st.columns([1.1, 1])
    with col_img:
        tab_names = ["Оригінал"] + [f"Grad-CAM: {dn(c)}" for c in d.shown_classes]
        tabs = st.tabs(tab_names)
        with tabs[0]:
            st.image(rgb, caption=file_name)
        for tab, cls in zip(tabs[1:], d.shown_classes, strict=True):
            with tab:
                st.image(gradcam.overlay(rgb, result["heatmaps"][cls], alpha),
                         caption=f"Області, що найбільше вплинули на оцінку класу «{dn(cls)}»")
        st.caption("Кольорова шкала: синій - слабкий вплив, червоний - сильний. "
                   "Теплова карта має низьку роздільність і не є сегментацією ураження.")
    with col_info:
        st.markdown("**Оцінки моделі за класами**")
        fig = scores_figure(d, kb)
        st.pyplot(fig)
        plt.close(fig)
        desc = d.description
        st.markdown(f"**Опис класу «{dn(d.top1)}» (з бази знань)**")
        st.write(desc.get("description", ""))
        rules = (desc.get("interpretation_rules") or []) + (desc.get("general_rules") or [])
        if rules:
            st.markdown("**Правила інтерпретації**")
            st.markdown("\n".join(f"- `{r['id']}` {r['text']}" for r in rules))

    with st.expander("🧭 Чому система так вирішила? Трасування правил вирішувача", expanded=True):
        trace_df = pd.DataFrame([{"Правило": t.rule_id, "Назва": t.name,
                                  "Спрацювало": "✅ так" if t.fired else "— ні",
                                  "Пояснення": t.explanation} for t in d.trace])
        st.dataframe(trace_df, hide_index=True)
        if not d.params.get("tuned"):
            st.caption("Пороги T і δ - значення за замовчуванням (ще не підібрані за валідаційною вибіркою).")

    st.error(desc.get("disclaimer", "") + " Оцінки softmax не є каліброваними ймовірностями.", icon="🛑")
    report = (f"Файл: {file_name}\nДата аналізу: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
              f"Запуск: {results.find_run().name}\n\n" + d.to_text(kb))
    st.download_button("💾 Завантажити текстовий звіт", data=report.encode("utf-8"),
                       file_name=f"cxr_report_{datetime.now():%Y%m%d_%H%M%S}.txt", mime="text/plain")


def main() -> None:
    st.set_page_config(page_title="CXR-XAI", page_icon="🩻", layout="wide")
    st.title(f"🩻 {APP_TITLE}")
    st.caption("Згорткова нейронна мережа Xception · фреймова база знань · вирішувач на правилах · "
               "пояснення результатів (трасування правил і Grad-CAM)")
    st.warning(DISCLAIMER, icon="⚠️")

    run, model, kb, engine = load_resources()
    alpha = render_sidebar(run, kb, model)

    st.subheader("1. Завантажте рентгенограму")
    uploaded = st.file_uploader("Фронтальна (передньо-задня або задньо-передня) рентгенограма органів "
                                "грудної клітки: PNG, JPG або BMP",
                                type=["png", "jpg", "jpeg", "bmp"])
    if uploaded is None:
        st.info("Очікування файлу. Після завантаження натисніть «Аналізувати».", icon="📂")
        return
    try:
        raw, pil = read_upload(uploaded)
    except UserInputError as e:
        st.error(str(e), icon="🚫")
        return

    rgb = to_display_rgb(pil)
    if looks_like_color_photo(rgb):
        st.warning("Зображення кольорове і, ймовірно, не є рентгенограмою. Модель навчено лише на "
                   "рентгенограмах грудної клітки, тому результат для такого файлу не має сенсу.", icon="⚠️")

    st.subheader("2. Запустіть аналіз")
    col_prev, col_btn = st.columns([1, 2])
    with col_prev:
        st.image(rgb, caption=f"{uploaded.name} · {pil.width}×{pil.height} · {pil.mode}")
    digest = hashlib.sha1(raw).hexdigest()
    with col_btn:
        clicked = st.button("🔍 Аналізувати", type="primary")
        st.caption("Аналіз займає кілька секунд (на CPU довше).")
    if clicked:
        with st.spinner("Обчислення оцінок моделі, теплової карти та виведення за правилами…"):
            try:
                st.session_state["result"] = {"digest": digest, **analyze(model, engine, raw)}
            except Exception as e:  # noqa: BLE001 - повідомити користувача замість трасування стеку
                st.session_state.pop("result", None)
                st.error(f"Під час аналізу сталася помилка: {type(e).__name__}: {e}", icon="🚫")
                return
    result = st.session_state.get("result")
    if result and result["digest"] == digest:
        render_result(result, rgb, kb, alpha, uploaded.name)


if __name__ == "__main__":
    main()
