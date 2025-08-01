import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# Настройки обработки
MIN_AREA = 100  # Минимальная площадь контура профиля
CIRCULARITY_THRESH = 0.7  # Порог круглости (1.0 - идеальный круг)
CANNY_THRESH = (50, 150)  # Пороги для детекции краёв
BLUR_SIZE = (5, 5)  # Размер размытия


def is_circular(contour):
    """Проверяет, насколько контур близок к кругу"""
    area = cv2.contourArea(contour)
    if area == 0:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False

    # Вычисляем круглость
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity > CIRCULARITY_THRESH


def process_image(input_path, output_dir):
    """Обрабатывает одно изображение и сохраняет результат"""
    # Создаём папку для результатов
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Загрузка изображения
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"⚠️ Ошибка: не удалось загрузить {input_path.name}")
        return 0

    # Предварительная обработка
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_SIZE, 0)

    # Детекция краёв
    edges = cv2.Canny(blurred, *CANNY_THRESH)

    # Поиск контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA and is_circular(cnt):
            valid_contours.append(cnt)

    # Визуализация
    result_img = img.copy()
    cv2.drawContours(result_img, valid_contours, -1, (0, 255, 0), 2)

    # Добавляем текст с количеством
    cv2.putText(result_img, f"Circles: {len(valid_contours)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Сохраняем результат
    timestamp = datetime.now().strftime("%H%M%S")
    output_path = Path(output_dir) / f"result_{timestamp}_{input_path.name}"
    cv2.imwrite(str(output_path), result_img)

    return len(valid_contours)


def main():
    # Пути к данным
    input_dir = Path("dataset/input")
    output_dir = Path("dataset/output")

    print("🔍 Начинаю обработку изображений...")

    # Обрабатываем все JPG/PNG в папке
    total = 0
    for img_path in input_dir.glob("*.*"):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            count = process_image(img_path, output_dir)
            print(f"📊 {img_path.name}: {count} кругов")
            total += count

    print(f"\n✅ Готово! Обработано изображений: {len(list(input_dir.glob('*.*')))}")
    print(f"📦 Всего кругов обнаружено: {total}")


if __name__ == "__main__":
    main()