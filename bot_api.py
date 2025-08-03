import os
import logging
import cv2
import numpy as np
from io import BytesIO
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from inference import get_model
import supervision as sv
from pathlib import Path
import time

# Отключаем все ненужные модели для подавления предупреждений
for model_env in [
    'PALIGEMMA_ENABLED', 'FLORENCE2_ENABLED', 'QWEN_2_5_ENABLED',
    'CORE_MODEL_SAM_ENABLED', 'CORE_MODEL_SAM2_ENABLED', 'CORE_MODEL_CLIP_ENABLED',
    'CORE_MODEL_GAZE_ENABLED', 'SMOLVLM2_ENABLED', 'DEPTH_ESTIMATION_ENABLED',
    'MOONDREAM2_ENABLED', 'CORE_MODEL_TROCR_ENABLED', 'CORE_MODEL_GROUNDINGDINO_ENABLED',
    'CORE_MODEL_YOLO_WORLD_ENABLED', 'CORE_MODEL_PE_ENABLED'
]:
    os.environ[model_env] = 'False'

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
    try:
        await update.message.reply_text(
            "Привет! Отправь мне изображение, и я проанализирую его с помощью модели компьютерного зрения."
        )
    except Exception as e:
        logger.error(f"Ошибка в команде /start: {str(e)}")


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик входящих изображений"""
    user = update.message.from_user
    logger.info("Изображение от %s: %s", user.first_name, user.id)

    file_path = None
    try:
        # Скачиваем фото
        photo_file = await update.message.photo[-1].get_file()
        file_path = await photo_file.download_to_drive()

        # Читаем изображение с помощью OpenCV
        image = cv2.imread(str(file_path))

        # Ресайз изображения для обработки
        max_size = 1024
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        # Выполняем инференс
        start_time = time.time()
        result = model.infer(image)[0]
        logger.info(f"Инференс выполнен за {time.time() - start_time:.2f} сек")

        # Обрабатываем результаты через supervision
        detections = sv.Detections.from_inference(result)

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
        if len(detections) > 0:
            class_names = [
                result.predictions[i].class_name
                for i in detections.class_id
            ]
            counts = {cls: class_names.count(cls) for cls in set(class_names)}
            report = "\n".join([f"{cls}: {count}" for cls, count in counts.items()])
            await update.message.reply_text(f"Обнаружены объекты:\n{report}")
        else:
            await update.message.reply_text("⚠️ На изображении не обнаружено объектов")

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}", exc_info=True)
        try:
            await update.message.reply_text(f"⚠️ Произошла ошибка при обработке изображения: {str(e)}")
        except Exception as inner_e:
            logger.error(f"Ошибка при отправке сообщения об ошибке: {str(inner_e)}")

    finally:
        # Удаляем временный файл
        try:
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
        except Exception as e:
            logger.error(f"Ошибка при удалении временного файла: {str(e)}")


def main() -> None:
    """Запуск бота"""
    # Создаем приложение с настройками для работы в нестабильных сетях
    app = Application.builder().token(BOT_TOKEN).build()

    # Регистрируем обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Запускаем бота с обработкой сетевых ошибок
    logger.info("Бот запущен...")

    # Повторные попытки подключения при сетевых ошибках
    while True:
        try:
            app.run_polling(
                poll_interval=1.0,
                timeout=30,
                drop_pending_updates=True
            )
        except Exception as e:
            logger.error(f"Критическая ошибка в приложении: {str(e)}")
            logger.info("Перезапуск бота через 10 секунд...")
            time.sleep(10)


if __name__ == '__main__':
    main()