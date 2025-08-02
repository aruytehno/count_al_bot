import os
import telebot
from ultralytics import YOLO
import cv2
from dotenv import load_dotenv

load_dotenv()
# Конфигурация

MODELS = {
    "round": "models/rou.pt",
    "rectangle": "models/rectangle.pt"
}

# Инициализация

bot = telebot.TeleBot(os.getenv('BOT_TOKEN'))
models_cache = {}


# Загрузка моделей при старте
def load_models():
    for pipe_type, path in MODELS.items():
        models_cache[pipe_type] = YOLO(path)
        print(f"Модель {pipe_type} загружена")


load_models()


@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = telebot.types.ReplyKeyboardMarkup(row_width=2)
    btn1 = telebot.types.KeyboardButton("Круглые")
    btn2 = telebot.types.KeyboardButton("Квадратные")
    markup.add(btn1, btn2)
    bot.send_message(message.chat.id, "Выберите тип труб:", reply_markup=markup)


@bot.message_handler(func=lambda m: m.text in ["Круглые", "Квадратные"])
def handle_pipe_type(message):
    chat_id = message.chat.id
    pipe_type = "round" if message.text == "Круглые" else "rectangle"
    bot.send_message(chat_id, f"Ожидаю фото для {message.text} труб...")
    bot.register_next_step_handler(message, process_photo, pipe_type)


def process_photo(message, pipe_type):
    if not message.photo:
        bot.reply_to(message, "Пожалуйста, отправьте фото")
        return

    try:
        # Скачивание фото
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded = bot.download_file(file_info.file_path)

        # Сохранение временного файла
        temp_path = f"temp_{message.chat.id}.jpg"
        with open(temp_path, 'wb') as f:
            f.write(downloaded)

        # Обработка
        result_img, count = detect_pipes(temp_path, pipe_type)

        # Отправка результата
        result_path = f"result_{message.chat.id}.jpg"
        cv2.imwrite(result_path, result_img)
        with open(result_path, 'rb') as photo:
            bot.send_photo(message.chat.id, photo)
        bot.send_message(message.chat.id, f"Найдено труб: {count}")

        # Очистка
        os.remove(temp_path)
        os.remove(result_path)

    except Exception as e:
        bot.reply_to(message, f"Ошибка: {str(e)}")


def detect_pipes(image_path, pipe_type):
    # Загрузка изображения
    img = cv2.imread(image_path)

    # Детекция
    results = models_cache[pipe_type](img, imgsz=1024, conf=0.6)

    # Визуализация
    result_img = results[0].plot()

    # Подсчёт
    count = len(results[0].boxes)

    return result_img, count


if __name__ == "__main__":
    print("Бот запущен")
    bot.infinity_polling()