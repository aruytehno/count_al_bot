import os
import telebot
from ultralytics import YOLO
import cv2
from dotenv import load_dotenv
import logging
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import threading
from roboflow import Roboflow

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class UnifiedModelLoader:
    """Класс для загрузки и управления единой моделью"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.model_path = "models/com-aus-3.pt"
        os.makedirs("models", exist_ok=True)

        if not os.path.exists(self.model_path):
            self._download_model()

        try:
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names
            logger.info(f"Модель успешно загружена. Доступные классы: {self.class_names}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise

    def _download_model(self):
        """Скачивание модели через Roboflow API"""
        try:
            logger.info("Начинаем скачивание модели с Roboflow...")
            rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
            project = rf.workspace("austral").project("com-aus-3")
            project.version(1).download("yolov8", location="models")

            # Переименовываем файл для удобства
            os.rename("models/com-aus-3/weights/best.pt", self.model_path)
            logger.info("Модель успешно скачана и сохранена")
        except Exception as e:
            logger.error(f"Ошибка при скачивании модели: {str(e)}")
            raise

    def get_model(self):
        return self.model

    def get_class_names(self):
        return self.class_names


# Инициализация бота и модели
bot = telebot.TeleBot(os.getenv('BOT_TOKEN'))

try:
    model_loader = UnifiedModelLoader()
    detection_model = model_loader.get_model()
    class_names = model_loader.get_class_names()
except Exception as e:
    logger.error(f"Не удалось инициализировать модель: {str(e)}")
    exit(1)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "👋 Добро пожаловать в PipeDetectorBot!\n\n"
        "📸 Просто отправьте фото труб для автоматического анализа.\n\n"
        "Бот определит и посчитает:\n"
        "- Круглые трубы\n"
        "- Квадратные/прямоугольные трубы\n"
        "- Другие металлические профили\n\n"
        "Поддерживаемые форматы: JPEG, PNG"
    )
    bot.send_message(message.chat.id, welcome_text)


@bot.message_handler(content_types=['photo', 'document'])
def handle_photo(message):
    chat_id = message.chat.id

    try:
        # Получаем файл изображения
        if message.photo:
            file_id = message.photo[-1].file_id
        elif message.document and message.document.mime_type.startswith('image/'):
            file_id = message.document.file_id
        else:
            raise ValueError("Пожалуйста, отправьте изображение в формате JPEG или PNG")

        # Скачиваем изображение
        file_info = bot.get_file(file_id)
        file_bytes = bot.download_file(file_info.file_path)

        # Обрабатываем изображение
        img = read_image(file_bytes)
        result_img, counts = detect_objects(img)

        # Формируем отчет
        report = "🔍 Результаты анализа:\n"
        for class_id, count in counts.items():
            class_name = class_names.get(class_id, f"Объект {class_id}")
            report += f"• {class_name}: {count}\n"
        report += f"\nВсего обнаружено: {sum(counts.values())}"

        # Отправляем результат
        bot.send_photo(
            chat_id,
            image_to_bytes(result_img),
            caption=report
        )

    except Exception as e:
        error_msg = f"⚠️ Ошибка: {str(e)}"
        logger.error(f"Chat {chat_id}: {error_msg}", exc_info=True)
        bot.send_message(chat_id, error_msg)
        bot.send_message(chat_id, "Попробуйте отправить другое изображение.")


def read_image(file_bytes):
    """Чтение изображения из байтов"""
    try:
        img = np.array(Image.open(BytesIO(file_bytes)))
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img
    except Exception as e:
        raise ValueError("Не удалось прочитать изображение")


def detect_objects(image):
    """Обнаружение объектов на изображении"""
    results = detection_model(
        image,
        imgsz=1024,
        conf=0.5,
        verbose=False
    )

    # Подсчет объектов по классам
    counts = {}
    for box in results[0].boxes:
        class_id = int(box.cls)
        counts[class_id] = counts.get(class_id, 0) + 1

    # Визуализация
    result_img = results[0].plot(
        line_width=1,
        font_size=0.6,
        labels=True,
        pil=True
    )

    return result_img, counts


def image_to_bytes(img):
    """Конвертация изображения в байты"""
    img_pil = Image.fromarray(img)
    byte_io = BytesIO()
    img_pil.save(byte_io, format='JPEG', quality=85)
    byte_io.seek(0)
    return byte_io


if __name__ == "__main__":
    logger.info("Бот запущен")
    bot.infinity_polling()