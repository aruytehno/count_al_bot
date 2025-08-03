import os
import logging
import base64
import cv2
import numpy as np
from io import BytesIO
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from inference import get_model
import supervision as sv

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

# Инициализация модели Roboflow
model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    await update.message.reply_text(
        "Привет! Отправь мне изображение, и я проанализирую его с помощью модели компьютерного зрения."
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик входящих изображений"""
    user = update.message.from_user
    logger.info("Изображение от %s: %s", user.first_name, user.id)

    try:
        # Скачиваем фото
        photo_file = await update.message.photo[-1].get_file()
        file_path = await photo_file.download_to_drive()

        # Читаем изображение с помощью OpenCV
        image = cv2.imread(str(file_path))

        # Выполняем инференс
        results = model.infer(image)[0]

        # Обрабатываем результаты
        detections = sv.Detections.from_inference(results)

        # Визуализируем результаты
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)

        # Конвертируем в формат для Telegram
        _, img_encoded = cv2.imencode('.jpg', annotated_image)
        img_bytes = img_encoded.tobytes()

        # Отправляем результат
        await update.message.reply_photo(photo=BytesIO(img_bytes))

        # Формируем текстовый отчет
        predictions = results.get("predictions", [])
        if predictions:
            classes = [p["class"] for p in predictions]
            counts = {cls: classes.count(cls) for cls in set(classes)}
            report = "\n".join([f"{cls}: {count}" for cls, count in counts.items()])
            await update.message.reply_text(f"Обнаружены объекты:\n{report}")
        else:
            await update.message.reply_text("⚠️ На изображении не обнаружено объектов")

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}", exc_info=True)
        await update.message.reply_text(f"⚠️ Произошла ошибка: {str(e)}")

    finally:
        # Удаляем временный файл
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()


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