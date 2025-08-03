import os
import logging
import base64
from io import BytesIO
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Загрузка переменных окружения
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
MODEL_ID = "com-aus-3/1"

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация клиента Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://infer.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# Конфигурация для получения изображения с аннотациями
CONFIG = InferenceConfiguration(
    format="image",  # Получаем только изображение с разметкой
    confidence_threshold=0.5,  # Порог уверенности
    visualize_labels=True,  # Показывать метки классов
    visualize_predictions=True  # Визуализировать предсказания
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    await update.message.reply_text(
        "Привет! Отправь мне изображение, и я проанализирую его с помощью модели компьютерного зрения."
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик входящих изображений"""
    user = update.message.from_user
    logger.info("Изображение от %s: %s", user.first_name, user.id)

    # Скачиваем фото
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = await photo_file.download_as_bytearray()

    # Отправляем запрос в Roboflow API
    try:
        with CLIENT.use_configuration(CONFIG):
            result = await CLIENT.infer_async(image_bytes, model_id=MODEL_ID)
    except Exception as e:
        logger.error(f"Ошибка Roboflow API: {e}")
        await update.message.reply_text("⚠️ Произошла ошибка при обработке изображения")
        return

    # Декодируем и отправляем результат
    try:
        if isinstance(result, bytes):
            # Для формата 'image'
            await update.message.reply_photo(photo=BytesIO(result))
        else:
            # Для формата 'image_and_json'
            base64_data = result["visualization"].split(",")[1]
            image_data = base64.b64decode(base64_data)
            await update.message.reply_photo(photo=BytesIO(image_data))

        await update.message.reply_text("✅ Анализ завершен!")
    except Exception as e:
        logger.error(f"Ошибка обработки результата: {e}")
        await update.message.reply_text("⚠️ Ошибка при формировании результата")


def main() -> None:
    """Запуск бота"""
    # Создаем приложение
    app = Application.builder().token(BOT_TOKEN).build()

    # Регистрируем обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Запускаем бота
    logger.info("Бот запущен...")
    app.run_polling()


if __name__ == '__main__':
    main()