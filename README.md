# 🐾 Animal Detection using YOLOv8

This project is a custom-built **animal detection system** using YOLOv8. The goal was to create a model that can reliably detect multiple animal species from images (and eventually real-time video), and build a pipeline that is clean, reproducible, and deployable.

Instead of just training a model and stopping there, this project focuses on building a **complete workflow** — from dataset handling to training, prediction, and deployment readiness.

---

## 🚀 What this project can do

* Detect multiple animal species in images
* Use a custom-trained YOLOv8 model (not just pretrained)
* Automatically download dataset from Kaggle
* Run predictions on new images easily
* Export model to ONNX for deployment (e.g., Raspberry Pi)
* Designed to be reproducible for anyone cloning the repo

---

## 🧠 Model Overview

* Model: YOLOv8 (Ultralytics)
* Approach: Transfer Learning (fine-tuned on custom dataset)
* Image Size: 640 × 640
* Training Epochs: ~40–50

The model starts with pretrained weights and then learns to specialize on the custom animal dataset.

---

## 🐾 What animals can it detect?

The model is trained on the following classes:

* Bagula
* Bear
* Boar
* Buffalo
* Cow
* Crow
* Deer
* Hornbill
* Lion
* Monkey
* Pigeon
* Sarus
* Sparrow
* Tiger

---

## 📊 Dataset Details

This project uses a custom dataset hosted on Kaggle.

* Training Images: ~120,000
* Validation Images: ~30,000
* Test Set: Included

Each image is annotated using **YOLO format**, meaning:

* Bounding boxes are normalized
* Each image has a corresponding `.txt` label file

Dataset link:
https://www.kaggle.com/datasets/priyaansuu/animal

---

## 📂 Project Structure

```plaintext
Animal-detection-model/
├── data/                  # Dataset (downloaded via Kaggle)
├── src/                   # Training and prediction scripts
│   ├── train_model.py
│   └── predict.py
├── notebook/              # Jupyter notebooks (experiments)
├── model/                 # Saved models (weights)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/priyaansuuu/Animal_Detection_Model.git
cd Animal_Detection_Model
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Setup Kaggle API (one-time setup)

1. Go to Kaggle → Account
2. Click **Create New API Token**
3. Move the downloaded `kaggle.json` file to:

```plaintext
C:\Users\<your-username>\.kaggle\
```

---

## 📦 Download Dataset

Run this inside the project directory:

```bash
kaggle datasets download -d priyaansuu/animal
```

Then extract it (automatically handled in script if implemented).

---

## 🏋️ Training the Model

```bash
python src/train_model.py
```

This will:

* Load the dataset
* Train the YOLOv8 model
* Save weights in `model/runs/`

---

## 🔍 Running Predictions

```bash
python src/predict.py
```

You can:

* Run detection on images
* Extend it to video/webcam

Results will be saved with bounding boxes.

---

## 📤 Model Export (for deployment)

The model is also exported in ONNX format which is light, making it suitable for wider applictions such as:
* Raspberry Pi
* Edge devices
* Lightweight inference

---

## 🧪 How to use this project (simple flow)

```plaintext
Clone repo → Install dependencies → Download dataset → Train → Predict
```

Everything is designed so that a new user can run the project without confusion.

---

## 🔥 Future Improvements

* Add danger scoring system based on detected animals
* Real-time wildlife monitoring system
* Mobile or web interface
* Edge deployment optimization

---

## 💡 Why this project is interesting

This isn’t just a “train and forget” model. It focuses on:

* Building a **complete ML pipeline**
* Making the project **reproducible**
* Keeping code clean and modular
* Preparing for **real-world deployment**

---

## 🙌 Final Note

If you’re exploring computer vision or YOLO models, this project is a solid starting point to understand:

* Custom dataset training
* Model pipelines
* Deployment-ready workflows

Feel free to fork, experiment, and improve it 🚀
