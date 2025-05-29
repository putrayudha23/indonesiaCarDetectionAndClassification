# Car Retrieval System 🚗📹

This project is a **Car Retrieval System** that integrates **object detection** and **object classification** into a single, user-friendly application. Using **YOLOv5** for vehicle detection and **ResNet50** for vehicle type classification, the system delivers reliable results through a graphical interface built with **CustomTkinter**.

The best-performing detection configuration is **YOLOv5m with image augmentation (Y2)**, while the highest classification accuracy (**73.86%**) is achieved using **ResNet50 with full fine-tuning and heavy augmentation (C2)**.

---

## 🔧 Features
- 🎯 Real-time car detection using **YOLOv5**
- 🚙 Vehicle classification with **ResNet50-based models**
- 🖥 GUI built with **CustomTkinter**
- 📊 Flowchart-based architecture for clarity
- 🎥 Supports video input and displays output with predictions

---

## 📽 Example Workflow
1. Select a video file via the GUI.
2. Click to start detection and classification.
3. View bounding boxes and predicted labels rendered on video frames.

---

## 🧠 Best Model Results
- **Detection:** YOLOv5m + Augmentation (Y2)  
- **Classification:** ResNet50 + Fine-tuning All Layers + Heavy Augmentation (C2) – *73.86% Accuracy*

---

## 🛠 Built With

### 🐍 Programming Language
- Python 3.10

### 📦 Main Libraries
- [`ultralytics`](https://pypi.org/project/ultralytics/) – YOLOv5 framework
- `torch`, `torchvision` – Deep learning (ResNet50)
- `tensorflow`, `keras` – For classification
- `opencv-python` – Video input/output processing
- `customtkinter` – GUI framework
- `pandas`, `numpy`, `matplotlib`, `seaborn` – Data processing and visualization
- `scikit-learn` – Model evaluation tools

See full list in [`requirements.txt`](./requirements.txt).

---

### 📦 Dataset for object detection
https://universe.roboflow.com/telkom-university-xlloq/indonesia-vehicle

## 🚀 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/car-retrieval-system.git
cd car-retrieval-system

# Install all dependencies
pip install -r requirements.txt
