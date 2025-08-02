import os
import telebot
from ultralytics import YOLO
import cv2
from dotenv import load_dotenv
import uuid
import logging
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Конфигурация моделей
MODELS = {
    "round": "models/rou.pt",
    "rectangle": "models/rectangle.pt"
}

# Инициализация бота
bot = telebot.TeleBot(os.getenv('BOT_TOKEN'))
models_cache = {}


# Загрузка моделей при старте
def load_models():
    for pipe_type, path in MODELS.items():
        try:
            models_cache[pipe_type] = YOLO(path)
            logger.info(f"Модель {pipe_type} успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {pipe_type}: {str(e)}")


load_models()


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    markup = telebot.types.ReplyKeyboardMarkup(
        row_width=2,
        resize_keyboard=True
    )
    btn1 = telebot.types.KeyboardButton("Круглые")
    btn2 = telebot.types.KeyboardButton("Квадратные")
    markup.add(btn1, btn2)

    welcome_text = (
        "👋 Добро пожаловать в PipeDetectorBot!\n\n"
        "📸 Отправьте фото труб для анализа:\n"
        "1. Выберите тип труб\n"
        "2. Отправьте фотографию\n"
        "3. Получите результат с подсчетом\n\n"
        "Поддерживаемые форматы: JPEG, PNG"
    )

    bot.send_message(message.chat.id, welcome_text, reply_markup=markup)


@bot.message_handler(func=lambda m: m.text in ["Круглые", "Квадратные"])
def handle_pipe_type(message):
    chat_id = message.chat.id
    pipe_type = "round" if message.text == "Круглые" else "rectangle"

    if pipe_type not in models_cache:
        bot.send_message(chat_id, "⚠️ Модель для обработки недоступна, попробуйте позже")
        return

    bot.send_message(
        chat_id,
        f"🔄 Ожидаю фото для {message.text} труб...\n\n"
        "Отправьте четкое фото труб сверху. "
        "Для лучшего результата избегайте бликов и сильных теней."
    )
    bot.register_next_step_handler(message, process_photo, pipe_type)


def process_photo(message, pipe_type):
    chat_id = message.chat.id
    file_id = None

    try:
        # Обработка фото из разных источников
        if message.photo:
            file_id = message.photo[-1].file_id
        elif message.document:
            if message.document.mime_type.split('/')[0] != 'image':
                raise ValueError("Неподдерживаемый формат файла")
            file_id = message.document.file_id
        else:
            raise ValueError("Пожалуйста, отправьте изображение")

        # Скачивание файла
        file_info = bot.get_file(file_id)
        file_bytes = bot.download_file(file_info.file_path)

        # Обработка в памяти
        img = read_image(file_bytes)
        result_img, count = detect_pipes(img, pipe_type)

        # Подготовка результата
        result_bytes = image_to_bytes(result_img)

        # Отправка результата
        bot.send_photo(
            chat_id,
            result_bytes,
            caption=f"✅ Результат анализа: найдено труб - {count}"
        )

        # Возврат меню
        send_welcome(message)

    except Exception as e:
        error_msg = f"⚠️ Ошибка обработки: {str(e)}"
        logger.error(f"Chat {chat_id}: {error_msg}")
        bot.send_message(chat_id, error_msg)
        send_welcome(message)


def read_image(file_bytes):
    """Чтение изображения из байтов с проверкой формата"""
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
    # Детекция
    results = models_cache[pipe_type](
        image,
        imgsz=1024,
        conf=0.6,
        verbose=False  # Отключаем лишние логи
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
    """Конвертация изображения в байты для отправки"""
    img_pil = Image.fromarray(img)
    byte_io = BytesIO()
    img_pil.save(byte_io, format='JPEG', quality=90)
    byte_io.seek(0)
    return byte_io


if __name__ == "__main__":
    logger.info("Бот запущен")
    bot.infinity_polling()