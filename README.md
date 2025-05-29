# Car Retrieval System 🚗📹

This project is a Car Retrieval System that integrates object detection and object classification into a single, user-friendly application. Using **YOLOv5** for car detection and **ResNet50** for vehicle classification, the system provides accurate results through a GUI built with **CustomTkinter**. The best-performing model configuration is YOLOv5m with image augmentation (Y2), and the highest classification accuracy (73.86%) is achieved using ResNet50 with full fine-tuning and heavy augmentation (C2).

## 🔧 Features
- Real-time car detection using YOLOv5 (Ultralytics)
- Vehicle classification using ResNet50-based models
- GUI built with CustomTkinter
- Flowchart-based architecture for easy understanding
- Supports video input and displays detection/classification results

## 📽 Example Use Case
1. Select a video using the GUI.
2. Run detection and classification.
3. View the output with bounding boxes and predicted labels.

## 🧠 Best Model Results
- **Object Detection:** YOLOv5m + Augmentation (Y2)  
- **Classification:** ResNet50 + Fine-tune All + Heavy Augmentation (C2) – *73.86% Accuracy*

## 🛠 Built With

### 🐍 Programming Language
- Python 3.10

### 📦 Main Libraries
- `ultralytics` (YOLOv5)
- `torch`, `torchvision` (Deep Learning)
- `tensorflow`, `keras` (Classification Model)
- `opencv-python` (Video processing)
- `customtkinter` (Graphical User Interface)
- `pandas`, `numpy`, `matplotlib`, `seaborn` (Data analysis & visualization)
- `scikit-learn` (Model evaluation)

See full list of dependencies in `requirements.txt`.

## 🚀 Installation

```bash
git clone https://github.com/yourusername/car-retrieval-system.git
cd car-retrieval-system
pip install -r requirements.txt
