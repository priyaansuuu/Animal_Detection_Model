

from ultralytics import YOLO
import cv2
import os


MODEL_PATH = "../model/best.pt"  # or best.onnx
SOURCE = "../test_images"  # image / folder / video / 0 for webcam
SAVE_DIR = "../predictions"

def load_model():
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully")
    return model


# IMAGE / VIDEO PREDICTION

def run_prediction(model):
    results = model.predict(
        source=SOURCE,
        imgsz=640,
        conf=0.4,          # confidence threshold
        save=True,         # saves output images
        project=SAVE_DIR,
        name="results",
        exist_ok=True
    )

    print("Prediction completed!")
    return results

# WEBCAM PREDICTION

def webcam_prediction(model):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot access webcam")
        return

    print("Press 'q' to exit webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = load_model()

    #  Choose ONE
    run_prediction(model)        # for images/videos
    # webcam_prediction(model)   # for live camera