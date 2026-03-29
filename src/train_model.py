
from ultralytics import YOLO
import torch
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

project_path = os.path.join(BASE_DIR, "model", "runs")

def main():
    # Check GPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model (auto-downloads if not present)
    model_path = os.path.abspath("../yolov8m.pt")
    model = YOLO(model_path)

    print("Loaded model:", model.ckpt_path)
    

    # Start training
    results = model.train(
        data="../data/dir__patht.yaml",  # path to your yaml
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        patience=10,

        # Optimization
        optimizer="AdamW",
        lr0=0.001,

        # Augmentations
        fliplr=0.5,
        scale=0.5,

        # Save settings
        project=project_path,
        name="animal_model",
        exist_ok=True
    )

    print("\n✅ Training Completed!")

    # Validate model
    metrics = model.val()
    print("\n📊 Validation Results:")
    print(metrics)

    # Save/export model
    model.export(format="onnx")
    print("📦 Model exported!")

if __name__ == "__main__":
    main()