from ultralytics import YOLO
import os
import argparse
import torch
import time
from datetime import timedelta


def print_gpu_info():
    """Выводит информацию о GPU, если доступно"""
    if torch.cuda.is_available():
        print(f"🖥️ Доступные GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("🖥️ GPU не обнаружен, будет использоваться CPU")


def train_model(data_path, epochs=100, imgsz=640, batch=8, model_name="yolov8n.pt", patience=50, resume=False):
    """Обучение модели YOLOv8 на кастомном датасете"""
    # Вывод информации о системе
    print_gpu_info()
    start_time = time.time()

    # Загрузка предобученной модели
    model = YOLO(model_name)

    # Определение устройства
    device = "0" if torch.cuda.is_available() and os.getenv("USE_GPU", "False") == "True" else "cpu"
    print(f"⚙️ Используется устройство: {'GPU' if device == '0' else 'CPU'}")

    # Обучение модели с дополнительными параметрами
    try:
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project="pipe_detection",
            name="pipe_train",
            save=True,
            val=True,
            device=device,
            resume=resume,
            patience=patience,  # Ранняя остановка при отсутствии улучшений
            augment=True,  # Автоматическая аугментация данных
            optimizer="auto",  # Автовыбор оптимального оптимизатора
            lr0=0.01,  # Начальная скорость обучения
            lrf=0.01,  # Финальная скорость обучения
            momentum=0.937,  # Моментум для SGD
            weight_decay=0.0005,  # Вес декаи
            warmup_epochs=3,  # Прогрев
            box=7.5,  # Вес потери боксов
            cls=0.5,  # Вес потери классификации
            dfl=1.5,  # Вес потери DFL
            close_mosaic=10,  # Отключение mosaic за 10 эпох до конца
            nbs=64,  # Нормализация размера батча
            pretrained=True,  # Использовать предобученные веса
        )

        # Экспорт лучшей модели в форматы ONNX и TorchScript
        print("🔄 Экспорт модели в ONNX и TorchScript...")
        model.export(format="onnx")  # Для использования в production
        model.export(format="torchscript")  # Для мобильных устройств

        # Вывод времени обучения
        training_time = timedelta(seconds=int(time.time() - start_time))
        print(f"⏱️ Общее время обучения: {training_time}")

        return results

    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано пользователем! Последняя модель сохранена.")
        print(f"🔄 Для возобновления обучения используйте: --resume")
        return None


if __name__ == "__main__":
    """
    Как использовать:
    Обычное обучение: python train.py --data com-aus-3-1/data.yaml --epochs 150 --imgsz 1024
    
Продолжение обучения: python train.py --resume --data com-aus-3-1/data.yaml --model pipe_detection/pipe_train/weights/last.pt --imgsz 1024 --epochs 150
               
               С GPU: USE_GPU=True python train.py
"""
    parser = argparse.ArgumentParser(description="Train YOLOv8 model on custom dataset")
    parser.add_argument("--data", default="com-aus-3-1/data.yaml", help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model to use")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")

    args = parser.parse_args()

    print(f"🚀 Начало обучения модели на {args.epochs} эпох...")
    results = train_model(
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model,
        patience=args.patience,
        resume=args.resume
    )

    if results is not None:
        print("✅ Обучение завершено! Лучшая модель сохранена в:")
        print(f"   - pipe_detection/pipe_train/weights/best.pt")
        print(f"   - pipe_detection/pipe_train/weights/best.onnx")