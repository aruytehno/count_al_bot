import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# Настройки
MODEL_PATH = "yolov8n.pt"  # Или ваша кастомная модель
CONFIDENCE_THRESH = 0.5  # Порог уверенности
IOU_THRESH = 0.45  # Порог для подавления дубликатов


def load_model():
    """Загружает YOLO модель"""
    model = YOLO(MODEL_PATH)
    return model


def process_image(model, input_path, output_dir):
    """Обрабатывает изображение с помощью YOLO"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Загрузка изображения
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"⚠️ Ошибка: не удалось загрузить {input_path.name}")
        return 0

    # Детекция объектов
    results = model.predict(
        source=img,
        conf=CONFIDENCE_THRESH,
        iou=IOU_THRESH,
        show_labels=True,
        show_conf=True
    )

    # Визуализация результатов
    result_img = results[0].plot()

    # Сохранение результата
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"yolo_result_{timestamp}_{input_path.name}"
    cv2.imwrite(str(output_path), result_img)

    return len(results[0].boxes)


def main():
    # Инициализация модели
    print("🔄 Загрузка модели YOLO...")
    model = load_model()
    print("✅ Модель загружена")

    # Пути к данным
    input_dir = Path("dataset/input")
    output_dir = Path("dataset/output_yolo")

    # Обработка изображений
    total = 0
    for img_path in input_dir.glob("*.*"):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            count = process_image(model, img_path, output_dir)
            print(f"📊 {img_path.name}: {count} труб")
            total += count

    print(f"\n✅ Готово! Всего обнаружено труб: {total}")


if __name__ == "__main__":
    main()