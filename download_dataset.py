import os
from roboflow import Roboflow

# Путь к папке dataset
DATASET_DIR = "dataset"

# Проверяем, существует ли папка dataset и не пуста ли она
if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
    print("Скачивание датасета...")

    # Получаем API-ключ из переменной окружения
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("API-ключ Roboflow не найден в переменных окружения (ROBOFLOW_API_KEY)")

    # Скачиваем датасет
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("austral").project("com-aus-3")
    dataset = project.version(1).download("yolov8", location=DATASET_DIR)

    print(f"Датасет сохранён в папку: {os.path.abspath(DATASET_DIR)}")
else:
    print("Датасет уже существует. Пропускаем скачивание.")