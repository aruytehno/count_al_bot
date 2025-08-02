# train.py
from ultralytics import YOLO
import os
import argparse


def train_model(data_path, epochs=100, imgsz=640, batch=8, model_name="yolov8n.pt"):
    """Обучение модели YOLOv8 на кастомном датасете"""
    # Загрузка предобученной модели
    model = YOLO(model_name)

    # Обучение модели
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="pipe_detection",
        name="pipe_train",
        save=True,
        val=True,
        device="0" if os.getenv("USE_GPU", "False") == "True" else "cpu"
    )

    # Экспорт лучшей модели в формат ONNX
    model.export(format="onnx")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model on custom dataset")
    parser.add_argument("--data", default="com-aus-3-1/data.yaml", help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model to use")

    args = parser.parse_args()

    print(f"🚀 Начало обучения модели на {args.epochs} эпох...")
    results = train_model(
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model
    )
    print("✅ Обучение завершено! Лучшая модель сохранена в pipe_detection/pipe_train/weights/best.pt")