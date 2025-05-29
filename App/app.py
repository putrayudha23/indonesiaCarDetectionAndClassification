from tkinter import filedialog, ttk
import os
import cv2
import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from torchvision import models, transforms
import torch.nn as nn
import urllib.error
import customtkinter as ctk


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Model App")
        self.root.geometry("1000x600")
        self.root.configure(bg="white")

        # Video display area (left)
        self.video_frame = tk.Frame(self.root, bg="white", highlightbackground="black", highlightthickness=2)
        self.video_frame.place(x=20, y=20, width=700, height=540)

        self.canvas = tk.Canvas(self.video_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Controls panel (right)
        self.right_frame = tk.Frame(self.root, bg="white")
        self.right_frame.place(x=740, y=20, width=230, height=540)

        # Load model files
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Full paths to model folders
        detection_path = os.path.join(base_dir, "Models", "Detection")
        classification_path = os.path.join(base_dir, "Models", "Classification")

        # Load model files
        self.detection_models = self.get_model_files(detection_path)
        self.classification_models = self.get_model_files(classification_path)

        # Detection model dropdown
        self.det_model_var = tk.StringVar()
        self.det_dropdown = ttk.Combobox(self.right_frame, textvariable=self.det_model_var, values=self.detection_models, state="readonly")
        if self.detection_models:
            self.det_dropdown.set(self.detection_models[0])
        self.det_dropdown.pack(pady=20)

        # Classification model dropdown
        self.cls_model_var = tk.StringVar()
        self.cls_dropdown = ttk.Combobox(self.right_frame, textvariable=self.cls_model_var, values=self.classification_models, state="readonly")
        if self.classification_models:
            self.cls_dropdown.set(self.classification_models[0])
        self.cls_dropdown.pack(pady=10)

        # Choose Video button
        self.choose_button = ctk.CTkButton(self.right_frame, text="Choose Video", command=self.choose_video, fg_color="#1e3f66", text_color="white", height=40)
        self.choose_button.pack(pady=20)

        # Run button
        self.run_button = ctk.CTkButton(self.right_frame, text="RUN", command=self.run_video, fg_color="#1e3f66", text_color="white", height=40)
        self.run_button.pack(pady=10)

        # Stop button
        self.stop_button = ctk.CTkButton(self.right_frame, text="STOP", command=self.stop_video, fg_color="#b22222", text_color="white", height=40)
        self.stop_button.pack(pady=10)

        self.cap = None  # VideoCapture object
        self.is_running = False  # Flag to control video playback

    def get_model_files(self, base_path):
        model_files = []
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(".pt") or file.endswith(".pth"):
                    # Relative to base_path, not full project root
                    rel_path = os.path.relpath(os.path.join(root, file), base_path)
                    model_files.append(rel_path)
        return model_files


    def choose_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if file_path:
            self.video_path = file_path

    def run_video(self):
        if hasattr(self, "video_path"):
            self.is_running = True  # Start the video playback
            base_dir = os.path.dirname(os.path.abspath(__file__))
            det_model_path = os.path.join(base_dir, "Models", "Detection", self.det_model_var.get())
            cls_model_path = os.path.join(base_dir, "Models", "Classification", self.cls_model_var.get())

            # Check detection model
            if not os.path.exists(det_model_path):
                tk.messagebox.showerror("Error", f"Detection model not found:\n{det_model_path}")
                return

            # Try to load YOLOv5 detection model from torch hub
            try:
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=det_model_path, force_reload=True)
                self.yolo_model.eval()
            except urllib.error.HTTPError as e:
                if e.code == 504:
                    tk.messagebox.showerror("Error", "Failed to download YOLOv5 model due to a Gateway Timeout (504).\nPlease check your internet connection and try again.")
                else:
                    tk.messagebox.showerror("Error", f"HTTP Error {e.code}: {e.reason}")
                return
            except Exception as e:
                tk.messagebox.showerror("Error", f"Unexpected error while loading detection model:\n{str(e)}")
                return

            # Check classification model
            if not os.path.exists(cls_model_path):
                tk.messagebox.showerror("Error", f"Classification model not found:\n{cls_model_path}")
                return

            # Load classification model dynamically based on filename
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if "C3" in os.path.basename(cls_model_path):
                self.classifier_model = models.resnet18(pretrained=False)
            else:
                self.classifier_model = models.resnet50(pretrained=False)

            self.classifier_model.fc = nn.Linear(self.classifier_model.fc.in_features, 8)
            self.classifier_model.load_state_dict(torch.load(cls_model_path, map_location=self.device))
            self.classifier_model.to(self.device)
            self.classifier_model.eval()

            self.class_names = ['CityCar', 'Crossover', 'DoubleCabin', 'Hatchback', 'MPV', 'Pickup', 'Sedan', 'SUV']

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])

            self.cap = cv2.VideoCapture(self.video_path)
            self.display_frame()

    def display_frame(self):
        if self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                results = self.yolo_model(frame)

                for *box, conf, cls in results.xyxy[0]:
                    x1, y1, x2, y2 = map(int, box)
                    cropped = frame[y1:y2, x1:x2]

                    try:
                        image_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            outputs = self.classifier_model(image_tensor)
                            _, predicted = torch.max(outputs, 1)
                            label = self.class_names[predicted.item()]
                    except Exception as e:
                        label = "Unknown"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (36, 255, 12), 2)

                # Display in GUI canvas
                frame = cv2.resize(frame, (700, 540))
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.image = imgtk

                self.root.after(10, self.display_frame)
            else:
                # Video ended, stop automatically
                self.stop_video()

    def stop_video(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
