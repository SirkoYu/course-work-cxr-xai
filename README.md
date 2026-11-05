# CXR-XAI: класифікація рентгенограм органів грудної клітки з базою знань і поясненнями

Навчально-дослідницький прототип СШІ: Xception (ImageNet, Transfer Learning + fine-tuning),
фреймова база знань (JSON), вирішувач на правилах R1–R5, пояснення (трасування правил + Grad-CAM),
інтерфейс Streamlit. **Система не встановлює клінічний діагноз; оцінки softmax не є каліброваними ймовірностями.**

## Як це працює

Проєкт складається з двох частин: **конвеєр** готує дані, навчає й налаштовує модель, а **інтерфейс**
використовує результат для аналізу окремого знімка.

```
NPZ-файли ─ extract ─ check ─ dedup ─ split ─ train ─ evaluate ─ tune ─ report ─ gradcam
                                                 │                  │
                                                 ▼                  ▼
                                            final.keras     knowledge_base.json
                                                 │                  │
                                                 └───── cxr-xai app ┘
                                       знімок → модель → правила R1–R5 → результат і пояснення
```

Кожен крок конвеєра є командою `cxr-xai <команда>`, читає результати попереднього кроку з `outputs/`
і записує туди свої.

## Структура проєкту

```
cxr_xai/
├── .env.example             # шаблон налаштувань (скопіюйте в .env)
├── knowledge_base.json      # база знань: фрейми + правила вирішувача
├── pyproject.toml           # залежності, команда cxr-xai, ruff, pytest
├── src/cxr_xai/
│   ├── cli.py               # точка входу: cxr-xai <команда>
│   ├── app.py               # інтерфейс Streamlit
│   ├── settings.py          # налаштування з .env (pydantic-settings)
│   ├── constants.py         # незмінні факти про датасет (порядок класів тощо)
│   ├── runtime.py           # каталоги, seed, середовище, налаштування TensorFlow
│   ├── json_io.py           # читання та запис JSON
│   ├── pipeline/            # кроки конвеєра, по одному модулю на команду
│   │   ├── extract_npz.py        # extract:  NPZ -> PNG
│   │   ├── check_files.py        # check:    інвентаризація та перевірка файлів
│   │   ├── deduplicate.py        # dedup:    дублікати (SHA-256 + pHash) ДО поділу
│   │   ├── split.py              # split:    стратифікований поділ без витоку
│   │   ├── train.py              # train:    етап 1 - TL, етап 2 - fine-tuning
│   │   ├── evaluate.py           # evaluate: метрики на val (до/після fine-tuning)
│   │   ├── tune_rules.py         # tune:     добір T і δ, типові плутанини -> БЗ
│   │   ├── final_report.py       # report:   одноразове фінальне оцінювання на test
│   │   └── gradcam_analysis.py   # gradcam:  аналіз хибних класифікацій
│   ├── data/                # робота з даними
│   │   ├── class_labels.py       # клас і вибірка за назвами папок
│   │   ├── deduplication.py      # хешування (у т.ч. віддзеркалені копії), union-find
│   │   ├── splitting.py          # груповий стратифікований поділ, перевірка витоку
│   │   └── loading.py            # tf.data, попередня обробка, аугментація (лише train)
│   ├── modeling/            # нейромережа
│   │   ├── builder.py            # модель Xception, розморожування блоків
│   │   ├── evaluation.py         # метрики, матриці помилок
│   │   └── gradcam.py            # Grad-CAM, накладання, евристика країв
│   └── knowledge/           # база знань і міркування
│       ├── knowledge_base.py     # фреймова БЗ: успадкування слотів, валідація
│       ├── inference_engine.py   # вирішувач: правила R1–R5, трасування
│       └── rule_tuning.py        # добір T і δ, типові плутанини, статистика правил
└── tests/                   # модульні тести (pytest)
```

Каталоги `dataverse_files/`, `dataset/`, `outputs/` і файл `.env` створюються локально й не потрапляють у git.

## Датасет

Файли з Harvard Dataverse: `Dataset5_raw_train.npz`, `Dataset5_raw_val.npz`, `Dataset5_raw_test.npz`, `README`.
Кожен NPZ містить ключі `image` (4D-масив пікселів), `image_name`, `image_label`.
Мітки за README датасету, вони ж індекси виходів моделі:

| Мітка | Клас | Зображень за README |
|---|---|---|
| 0 | COVID-19 | 4 189 |
| 1 | Lung Opacity | 6 012 |
| 2 | Normal | 10 192 |
| 3 | Viral Pneumonia | 7 397 |
| 4 | Tuberculosis | 4 897 |
| | **Разом** | **32 687** |

Команда `cxr-xai extract` зберігає кожне зображення як PNG (без втрат), після чого працюють усі наступні кроки.
Оригінальний поділ train/val/test використовується лише для довідки: усі зображення об'єднуються,
очищуються від дублікатів і діляться заново (`cxr-xai dedup` рахує, скільки дублікатів розірвав оригінальний поділ).

## Встановлення

Потрібні Python 3.10–3.13 і [uv](https://docs.astral.sh/uv/).
Шлях до проєкту краще без кирилиці та пробілів, наприклад `C:\Course_CNN_Recognition\cxr_xai`.

```powershell
uv sync                       # створює .venv і встановлює проєкт із залежностями
Copy-Item .env.example .env   # налаштування
uv run pytest
```

Покладіть NPZ-файли датасету в каталог `dataverse_files/` усередині проєкту або вкажіть інший шлях у `CXR_NPZ_DIR`.

Без uv: `python -m venv .venv`, активуйте середовище, потім `pip install -e .` і `pip install pytest ruff`.
Встановлювати потрібно саме в режимі `-e` (editable): `.env` і відносні шляхи рахуються від кореня проєкту.

**GPU.** Нативна підтримка GPU у TensorFlow для Windows закінчилася на версії 2.10, тому в звичайному
Windows-середовищі навчання йде на CPU. Варіанти:
1. **CPU** — працює без додаткових налаштувань, але повільно. Швидкість видно після першої епохи
   (`cxr-xai train` друкує час кожної епохи). Для пришвидшення можна навчати на 224×224: `CXR_IMG_SIZE=224` у `.env`
   (інші кроки та інтерфейс беруть розмір входу із самої моделі).
2. **WSL2 + GPU NVIDIA** — у PowerShell від адміністратора `wsl --install`, потім в Ubuntu:
   ```bash
   cd /mnt/c/Course_CNN_Recognition/cxr_xai
   uv sync && uv pip install "tensorflow[and-cuda]"
   .venv/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   Далі запускайте команди через `.venv/bin/cxr-xai`, бо `uv run` повертає середовище до стану з `uv.lock`.
3. **Google Colab** — див. розділ нижче.

## Налаштування

Усі параметри задаються у файлі `.env`: шляхи, гіперпараметри навчання, аугментація, пороги, параметри інтерфейсу.
Повний список із поясненнями міститься в `.env.example`, а типи й обмеження описано в `src/cxr_xai/settings.py`.

- Некоректне значення або помилка в назві змінної `CXR_*` у `.env` зупиняє запуск із зрозумілим повідомленням.
- Складні значення задаються у форматі JSON: `CXR_T_GRID='[0.30, 0.99, 0.01]'`.
- Шлях до датасету зберігайте в `CXR_DATA_ROOT`. Після його зміни кроки треба повторити починаючи з `cxr-xai check`.

Будь-яке налаштування, яке використовує команда, можна необов'язково перевизначити для одного запуску:

```bash
uv run cxr-xai train --batch-size 8 --img-size 224 --epochs1 5
uv run cxr-xai dedup --threshold 2 --no-check-mirrored
```

Пріоритет: параметр командного рядка > змінна середовища > `.env` > значення за замовчуванням.
Параметр діє лише на один запуск і проходить ту саму валідацію, що й `.env`. Якщо вказали `--data-root` або
`--output-dir`, повторюйте їх у всіх наступних кроках; для постійних змін використовуйте `.env`.
Перелік параметрів команди: `uv run cxr-xai <команда> --help`.

## Порядок запуску

```bash
uv run cxr-xai extract
uv run cxr-xai check
uv run cxr-xai dedup             # перегляньте figures/near_duplicate_pairs.png!
uv run cxr-xai split
uv run cxr-xai train
uv run cxr-xai evaluate
uv run cxr-xai tune              # записує T, δ і плутанини в knowledge_base.json
uv run cxr-xai report            # ОДИН раз, коли все налаштовано
uv run cxr-xai gradcam --split test
uv run cxr-xai kb --validate --model outputs/models/final.keras

uv run cxr-xai app
```

Список команд: `uv run cxr-xai --help`. Параметри окремої команди: `uv run cxr-xai train --help`.

Для розпакування потрібно близько 2 ГБ вільної оперативної пам'яті (найбільший NPZ ~1,4 ГБ)
і кілька гігабайт на диску для PNG.

## Google Colab (навчання на GPU)

1. Скопіюйте папку `cxr_xai` і три NPZ-файли на Google Drive, увімкніть GPU
   (Середовище виконання → Змінити тип → GPU).
2. Створіть у папці `cxr_xai` файл `.env` із такими рядками:

```bash
CXR_NPZ_DIR=/content/drive/MyDrive/dataverse_files
# PNG розпаковуються на локальний диск Colab (читання тисяч файлів з Drive дуже повільне).
CXR_DATA_ROOT=/content/dataset
```

3. У ноутбуці:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/cxr_xai
!pip install -q -e .
# Повторюйте цей рядок на початку КОЖНОЇ сесії: диск Colab очищується.
!cxr-xai extract --overwrite
!cxr-xai check
!cxr-xai dedup
!cxr-xai split
!cxr-xai train
!cxr-xai evaluate
!cxr-xai tune
!cxr-xai report
!cxr-xai gradcam --split test
```

Результати зберігаються в `cxr_xai/outputs/` на Drive. Інтерфейс запускайте на своєму комп'ютері:
скопіюйте `outputs/models/` і `knowledge_base.json` у локальну папку проєкту та виконайте `uv run cxr-xai app`.

## Швидка перевірка коду без GPU

Режим `--smoke` бере лише кілька зображень кожного класу й перевіряє, що навчання запускається.
Результати пишуться в окрему папку `outputs/smoke/` і не зачіпають основні; для курсової вони непридатні.
Виконуйте після кроків `extract`, `check`, `dedup`, `split`:

```powershell
uv run cxr-xai train --smoke 8 --epochs1 1 --epochs2 1
```

## Перевірка якості коду

```powershell
uv run ruff check .
uv run pytest
```

## Що з результатів куди в курсовій

| Файл у `outputs/` | Де використати |
|---|---|
| `reports/extract_report.txt`, `npz_counts.csv`, `name_prefixes_by_class.csv` | 1.1 - опис датасету, склад джерел за класами |
| `reports/check_files_report.txt`, `inventory_by_class.csv` | 3.1 - перевірка файлів |
| `reports/dedup_report.txt`, `dedup_by_class.csv`, `figures/phash_nn_hist.png` | 3.1 - виявлення дублікатів, обґрунтування порогу |
| `reports/split_by_class.csv` | 3.1 - таблиця поділу на вибірки |
| `models/training_meta.json`, `reports/history_stage*.csv` | 3.1 - гіперпараметри, кількість епох, час навчання, середовище |
| `figures/training_curves.png` | 3.1 - криві навчання |
| `reports/metrics_val_stage1_vs_final.csv` | 3.1 - вибір моделі (val) |
| `reports/metrics_test_stage1_vs_final.csv` | 3.1 - метрики до/після fine-tuning (test) |
| `reports/classification_report_final_test.txt` | Додаток - повний звіт класифікації |
| `figures/confusion_final_test*.png` | 3.1 / Додаток - матриця помилок |
| `reports/tuning_report.txt`, `figures/threshold_tuning_val.png`, `risk_coverage_val.png` | 3.1 - добір T і δ |
| `reports/typical_confusions_val.csv`, `knowledge_base.json` | 3.2 - заповнення слота «типові плутанини» |
| `reports/rules_test_stats.csv` | 3.1 - статистика вирішувача на test |
| `reports/gradcam_report_test.txt`, `figures/gradcam_*_test.png` | 3.1 - аналіз хибних класифікацій |
| вивід `cxr-xai kb --validate` | 3.2 - перевірка відповідності індексів виходам мережі |
| вивід `pytest` | 3.1 - тестування |

## Обмеження (зазначити в роботі)

- Поділ виконується на рівні зображень (груп дублікатів), а не пацієнтів, якщо в датасеті немає ідентифікаторів пацієнтів.
- pHash виявляє лише візуально майже ідентичні та віддзеркалені копії; копії з поворотом чи сильною обрізкою можуть лишитися непоміченими. Поріг підбирається з ручною перевіркою прикладів.
- Клас Viral Pneumonia за описом датасету доповнено зображеннями з кількох джерел і він неоднорідний.
- Датасет зібрано з кількох джерел, тож модель може навчитися ознак джерела; Grad-CAM і евристика країв допомагають це виявити, але не виключають.
- Grad-CAM має низьку роздільність (10×10 для входу 299×299) і не є сегментацією ураження.
