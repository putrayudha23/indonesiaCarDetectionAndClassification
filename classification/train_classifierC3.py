import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
log_file = open("training_log_C3.txt", "w", encoding="utf-8")

# Light augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset
train_data = datasets.ImageFolder('output_dataset/train', transform=train_transform)
val_data = datasets.ImageFolder('output_dataset/val', transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Load ResNet18
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier (fine-tune FC only)
num_classes = len(train_data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Training setup
epochs = 50
patience = 10
best_val_loss = float('inf')
early_stop_counter = 0
train_losses = []
val_losses = []

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Logging
    log_file.write(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n")
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), 'car_classifier_resnet18_C3_best.pth')
        log_file.write("✅ Validation loss improved. Model saved.\n")
        print("✅ Validation loss improved. Model saved.")
    else:
        early_stop_counter += 1
        log_file.write(f"⚠️ Validation loss did not improve. Early stop count: {early_stop_counter}/{patience}\n")
        print(f"⚠️ Validation loss did not improve. Early stop count: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            log_file.write("⛔ Early stopping triggered.\n")
            print("⛔ Early stopping triggered.")
            break

log_file.close()

# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (C3)")
plt.legend()
plt.grid()
plt.savefig("loss_curve_C3.png")
plt.show()

# Evaluation
model.load_state_dict(torch.load('car_classifier_resnet18_C3_best.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification report
report = classification_report(all_labels, all_preds, target_names=val_data.classes, digits=4)
print("\n📋 Classification Report (C3):\n", report)

with open("classification_report_C3.txt", "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=val_data.classes, yticklabels=val_data.classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (C3)")
plt.savefig("confusion_matrix_C3.png")
plt.show()
