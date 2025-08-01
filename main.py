import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

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


def validate_paths(input_dir: Path, output_dir: Path) -> bool:
    """Проверяет существование входной директории и создаёт выходную"""
    if not input_dir.exists():
        print(f"❌ Ошибка: Входная директория {input_dir} не существует!")
        return False

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Ошибка при создании выходной директории: {e}")
        return False

    return True


def is_circular(contour):
    """Проверяет, насколько контур близок к кругу"""
    area = cv2.contourArea(contour)
    if area == 0:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False

    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity > CIRCULARITY_THRESH


def process_image(input_path: Path, output_dir: Path) -> int:
    """Обрабатывает одно изображение и сохраняет результат"""
    try:
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"⚠️ Ошибка: не удалось загрузить {input_path.name}")
            return 0

        # Обработка изображения
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, BLUR_SIZE, 0)
        edges = cv2.Canny(blurred, *CANNY_THRESH)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA and is_circular(cnt)]

        # Визуализация
        result_img = img.copy()
        cv2.drawContours(result_img, valid_contours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)

        for i, cnt in enumerate(valid_contours):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(result_img, (cX, cY), CENTER_RADIUS, CENTER_COLOR, -1)
                cv2.putText(result_img, str(i + 1), (cX - 10, cY + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Сохранение результата
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"result_{timestamp}_{input_path.name}"
        cv2.imwrite(str(output_path), result_img)

        return len(valid_contours)

    except Exception as e:
        print(f"⚠️ Ошибка при обработке {input_path.name}: {e}")
        return 0


def main():
    input_dir = Path("dataset/input")
    output_dir = Path("dataset/output")

    if not validate_paths(input_dir, output_dir):
        sys.exit(1)

    print("🔍 Начинаю обработку изображений...")
    print(f"Параметры обработки:\n"
          f"  MIN_AREA: {MIN_AREA}\n"
          f"  CIRCULARITY_THRESH: {CIRCULARITY_THRESH}\n"
          f"  CANNY_THRESH: {CANNY_THRESH}\n"
          f"  BLUR_SIZE: {BLUR_SIZE}")

    total_count = 0
    processed_files = 0

    for img_path in input_dir.glob("*.*"):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            count = process_image(img_path, output_dir)
            print(f"📊 {img_path.name}: {count} объектов")
            total_count += count
            processed_files += 1

    print(f"\n✅ Готово! Обработано файлов: {processed_files}")
    print(f"📦 Всего объектов обнаружено: {total_count}")


from pathlib import Path


def setup_directories():
    # Определяем пути
    base_dir = Path("dataset")
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"

    try:
        # Создаём директории
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Директории созданы:")
        print(f"   - Входные данные: {input_dir}")
        print(f"   - Результаты: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ Ошибка при создании директорий: {e}")
        return False


if __name__ == "__main__":
    setup_directories()
    main()