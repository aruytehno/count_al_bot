import os
import cv2
import numpy as np
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from dotenv import load_dotenv

# Загрузка токена бота
load_dotenv()
TOKEN = os.getenv('BOT_TOKEN')


# Клавиатура с режимами
def get_keyboard():
    return ReplyKeyboardMarkup([['🔍 Проверка заказа', '📦 Сверка остатков']], resize_keyboard=True)


# Обработка изображения (упрощенная версия)
def process_image(image_path):
    """Минимальная обработка изображения для демонстрации"""
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)
    except Exception as e:
        print(f"Ошибка обработки: {e}")
        return 0


async def start(update: Update, context: CallbackContext):
    """Обработка команды /start"""
    await update.message.reply_text(
        "Привет! Отправь мне фото алюминиевых профилей для подсчёта.",
        reply_markup=get_keyboard()
    )


async def handle_message(update: Update, context: CallbackContext):
    """Обработка текстовых сообщений"""
    text = update.message.text
    if text in ['🔍 Проверка заказа', '📦 Сверка остатков']:
        mode = 'order' if text == '🔍 Проверка заказа' else 'stock'
        context.user_data['mode'] = mode
        await update.message.reply_text(f"Режим: {text}. Отправьте фото профилей.")
    else:
        await update.message.reply_text("Используйте кнопки для выбора режима.")


async def handle_photo(update: Update, context: CallbackContext):
    """Обработка фотографий"""
    try:
        # Сохраняем фото
        photo = update.message.photo[-1]
        file = await photo.get_file()
        filename = f"temp_photo_{update.update_id}.jpg"
        await file.download_to_drive(filename)

        # Простая обработка
        count = process_image(filename)

        # Удаляем временный файл
        os.remove(filename)

        await update.message.reply_text(f"🔢 Предварительный подсчёт: {count} профилей\n"
                                        "⚠️ Это тестовая версия! Точность пока низкая.")

    except Exception as e:
        await update.message.reply_text(f"🚫 Ошибка: {str(e)}")


def main():
    """Запуск бота"""
    app = Application.builder().token(TOKEN).build()

    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()