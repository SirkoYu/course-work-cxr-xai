"""
Незмінні факти про датасет і формати файлів.

Це не налаштування: зміна порядку класів чи міток зробила б навчену модель і
базу знань неузгодженими. Те, що можна змінювати, міститься в settings.py.
"""

# Порядок у списку = порядок виходів нейромережі (індекси 0..4) і водночас
# числові мітки image_label у NPZ-файлах датасету (див. README датасету):
#   0 - COVID-19, 1 - Lung-Opacity, 2 - Normal, 3 - Viral Pneumonia, 4 - Tuberculosis.
# Цей самий порядок записано в базі знань (слот model_output_index);
# відповідність перевіряється автоматично (cxr_xai/knowledge/knowledge_base.py).
CLASS_NAMES = ["COVID-19", "Lung Opacity", "Normal", "Viral Pneumonia", "Tuberculosis"]
NPZ_LABELS = dict(enumerate(CLASS_NAMES))

# Кількість зображень за класами, заявлена в README датасету (для звірки).
README_CLASS_COUNTS = {"COVID-19": 4189, "Lung Opacity": 6012, "Normal": 10192,
                       "Viral Pneumonia": 7397, "Tuberculosis": 4897}

# Назви папок -> канонічна назва класу.
# Порівняння нечутливе до регістру; символи "_", "-" і пробіли вважаються однаковими.
CLASS_ALIASES = {
    "covid": "COVID-19",
    "covid 19": "COVID-19",
    "covid19": "COVID-19",
    "lung opacity": "Lung Opacity",
    "lungopacity": "Lung Opacity",
    "viral pneumonia": "Viral Pneumonia",
    "pneumonia": "Viral Pneumonia",
    "tuberculosis": "Tuberculosis",
    "tb": "Tuberculosis",
    "normal": "Normal",
}

# Формати, які вміє декодувати tf.io.decode_image (TIFF не підтримується).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
SUPPORTED_PIL_FORMATS = {"PNG", "JPEG", "MPO", "BMP"}

# Назви папок, що позначають готовий поділ в оригінальному датасеті.
# Готовий поділ НЕ використовується: усі зображення об'єднуються, очищуються
# від дублікатів і діляться заново (див. pipeline/split.py).
SPLIT_FOLDER_NAMES = {"train", "training", "test", "testing", "val", "valid", "validation"}
