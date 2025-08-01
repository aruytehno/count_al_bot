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
CONTOUR_COLOR = (0, 255, 0)  # Зелёный цвет для контуров
CONTOUR_THICKNESS = 3  # Толщина линий контуров
CENTER_COLOR = (0, 0, 255)  # Красный цвет для центров
CENTER_RADIUS = 5  # Размер точки в центре
TEXT_COLOR = (0, 0, 255)  # Красный цвет для текста
TEXT_SCALE = 1.5  # Размер шрифта


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

    # Сохраняем копию оригинального изображения
    original_img = img.copy()

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

    # Рисуем контуры
    cv2.drawContours(result_img, valid_contours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)

    # Рисуем центры и нумеруем круги
    for i, cnt in enumerate(valid_contours):
        # Вычисляем центр контура
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Рисуем центр круга
            cv2.circle(result_img, (cX, cY), CENTER_RADIUS, CENTER_COLOR, -1)
            # Добавляем номер круга
            cv2.putText(result_img, str(i + 1), (cX - 10, cY + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Добавляем текст с количеством
    cv2.putText(result_img, f"Detected Circles: {len(valid_contours)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, 3)

    # Добавляем информацию о параметрах
    params_text = f"MinArea: {MIN_AREA}, Circularity: {CIRCULARITY_THRESH}"
    cv2.putText(result_img, params_text, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

    # Создаем изображение с оригиналом и результатом
    comparison_img = np.hstack((original_img, result_img))

    # Сохраняем результат
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"result_{timestamp}_{input_path.name}"
    cv2.imwrite(str(output_path), comparison_img)

    return len(valid_contours)


def main():
    # Пути к данным
    input_dir = Path("dataset/input")
    output_dir = Path("dataset/output")

    print("🔍 Начинаю обработку изображений...")
    print(f"Используемые параметры:")
    print(f"  MIN_AREA: {MIN_AREA}")
    print(f"  CIRCULARITY_THRESH: {CIRCULARITY_THRESH}")
    print(f"  CANNY_THRESH: {CANNY_THRESH}")
    print(f"  BLUR_SIZE: {BLUR_SIZE}")

    # Обрабатываем все JPG/PNG в папке
    total_count = 0
    image_count = 0
    for img_path in input_dir.glob("*.*"):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            count = process_image(img_path, output_dir)
            print(f"📊 {img_path.name}: {count} кругов")
            total_count += count
            image_count += 1

    print(f"\n✅ Готово! Обработано изображений: {image_count}")
    print(f"📦 Всего кругов обнаружено: {total_count}")


if __name__ == "__main__":
    main()