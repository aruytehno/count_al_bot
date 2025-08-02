import os
import telebot
from ultralytics import YOLO
import cv2
from dotenv import load_dotenv
import logging
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from functools import lru_cache
import threading

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class ModelLoader:
    """Класс для управления загрузкой и кэшированием моделей"""
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
        self.models = {
            "round": "models/rou.pt",
            "rectangle": "models/rectangle.pt"
        }
        self.cache = {}
        self._load_all_models()

    def _load_all_models(self):
        """Параллельная загрузка моделей при старте"""
        threads = []
        for name, path in self.models.items():
            thread = threading.Thread(
                target=self._load_single_model,
                args=(name, path),
                daemon=True
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def _load_single_model(self, name, path):
        """Загрузка одной модели с обработкой ошибок"""
        try:
            model = YOLO(path)
            self.cache[name] = model
            logger.info(f"Модель {name} успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {name}: {str(e)}")
            self.cache[name] = None

    def get_model(self, name):
        """Получение модели из кэша"""
        return self.cache.get(name)


# Инициализация бота и моделей
bot = telebot.TeleBot(os.getenv('BOT_TOKEN'))
model_loader = ModelLoader()


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
    model = model_loader.get_model(pipe_type)

    if model is None:
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
        logger.error(f"Chat {chat_id}: {error_msg}", exc_info=True)
        bot.send_message(chat_id, error_msg)
        send_welcome(message)


@lru_cache(maxsize=32)
def read_image(file_bytes):
    """Чтение изображения из байтов с кэшированием"""
    try:
        img = np.array(Image.open(BytesIO(file_bytes)))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img
    except UnidentifiedImageError:
        raise ValueError("Неподдерживаемый формат изображения")


def detect_pipes(image, pipe_type):
    """Обработка изображения с помощью YOLO"""
    model = model_loader.get_model(pipe_type)
    if model is None:
        raise RuntimeError(f"Модель {pipe_type} не загружена")

    results = model(
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
    """Конвертация изображения в байты для отправки"""
    img_pil = Image.fromarray(img)
    byte_io = BytesIO()
    img_pil.save(byte_io, format='JPEG', quality=90)
    byte_io.seek(0)
    return byte_io


if __name__ == "__main__":
    logger.info("Бот запущен")
    bot.infinity_polling()