import cv2
import numpy as np
import os
from datetime import datetime


def count_profiles(image_path, output_dir="output"):
    """Основная функция подсчёта профилей"""
    # Создаём папку для результатов
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return 0

    # Предварительная обработка
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Детекция краёв
    edges = cv2.Canny(blurred, 50, 150)

    # Поиск контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация мелких контуров
    min_area = 100
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Визуализация результатов
    result_img = img.copy()
    cv2.drawContours(result_img, valid_contours, -1, (0, 255, 0), 2)

    # Сохранение результата
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"result_{timestamp}.jpg")
    cv2.imwrite(output_path, result_img)

    return len(valid_contours)


if __name__ == "__main__":
    # Пример использования
    input_image = "test_photo.jpg"  # Укажите путь к вашему изображению
    count = count_profiles(input_image)
    print(f"Найдено профилей: {count}")
    print(f"Результат сохранён в папке 'output'")