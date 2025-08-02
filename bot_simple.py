import os
import telebot
from ultralytics import YOLO
import cv2
from dotenv import load_dotenv
import logging
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from roboflow import Roboflow

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Конфигурация моделей
MODELS = {
    "rectangle": "models/rectangle.pt",
}
models_cache = {}

# Инициализация бота
bot = telebot.TeleBot(os.getenv('BOT_TOKEN'))


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "👋 Добро пожаловать в PipeDetectorBot!\n\n"
        "📸 Отправьте фото труб для анализа.\n"
        "Бот автоматически определит количество.\n\n"
        "Поддерживаемые форматы: JPEG, PNG (максимальный размер 10MB)"
    )
    bot.send_message(message.chat.id, welcome_text)


@bot.message_handler(content_types=['photo', 'document'])
def handle_photo(message):
    chat_id = message.chat.id
    file_id = None

    try:
        # Получение файла
        if message.photo:
            file_id = message.photo[-1].file_id
        elif message.document:
            if not message.document.mime_type.startswith('image/'):
                raise ValueError("Неподдерживаемый формат файла")
            file_id = message.document.file_id
        else:
            raise ValueError("Пожалуйста, отправьте изображение")

        # Скачивание с проверкой размера
        file_info = bot.get_file(file_id)
        if file_info.file_size > 10 * 1024 * 1024:
            raise ValueError("Файл слишком большой (максимальный размер 10MB)")

        file_bytes = bot.download_file(file_info.file_path)
        img = read_image(file_bytes)

        # Ленивая загрузка модели
        pipe_type = "rectangle"
        if pipe_type not in models_cache:
            models_cache[pipe_type] = YOLO(MODELS[pipe_type])
            logger.info(f"Модель {pipe_type} успешно загружена")

        # Детекция труб
        result_img, count = detect_pipes(img, pipe_type)

        # Отправка результата с очисткой буфера
        byte_io = image_to_bytes(result_img)
        try:
            bot.send_photo(
                chat_id,
                byte_io,
                caption=f"✅ Результат: найдено труб - {count}"
            )
        finally:
            byte_io.close()

    except Exception as e:
        error_msg = f"⚠️ Ошибка обработки: {str(e)}"
        logger.error(f"Chat {chat_id}: {error_msg}")
        bot.send_message(chat_id, error_msg)


def read_image(file_bytes):
    """Чтение изображения из байтов"""
    try:
        img = np.array(Image.open(BytesIO(file_bytes)))
        if len(img.shape) == 2:  # Grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img
    except UnidentifiedImageError:
        raise ValueError("Неподдерживаемый формат изображения")


def detect_pipes(image, pipe_type):
    """Обработка изображения с помощью YOLO"""
    results = models_cache[pipe_type](
        image,
        imgsz=1024,
        conf=0.6,
        verbose=False
    )

    # Визуализация результатов
    result_img = results[0].plot(
        line_width=2,
        font_size=0.8,
        labels=True,
        pil=True
    )

    # Подсчёт объектов
    count = len(results[0].boxes)

    return result_img, count


def image_to_bytes(img):
    """Конвертация изображения в байты"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    byte_io = BytesIO()
    img.save(byte_io, format='JPEG', quality=90)
    byte_io.seek(0)
    return byte_io

def download_dataset():
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("austral").project("com-aus-3")
    dataset = project.version(1).download("yolov8")
    print('dataset загружен')


if __name__ == "__main__":
    download_dataset()
    logger.info("Бот запущен")
    bot.infinity_polling()