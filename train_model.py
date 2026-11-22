import os
import copy
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Dataset location
DATA_DIR = r"C:\Users\Nayana D\Downloads\archive (7)\data"

# Save the trained model in your project folder
MODEL_PATH = "roadguard_accident_model.pth"

BATCH_SIZE = 16
EPOCHS = 8
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)

# Image preprocessing
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Load datasets
train_data = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_data = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=eval_transform
)

test_data = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=eval_transform
)

print("Class names:", train_data.classes)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Testing images:", len(test_data))

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# Load pretrained MobileNetV2
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)

# Replace final classification layer
model.classifier[1] = nn.Linear(
    model.last_channel, len(train_data.classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001
)

best_val_accuracy = 0.0
best_state = copy.deepcopy(model.state_dict())

# Train model
for epoch in range(EPOCHS):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            val_total += labels.size(0)

    train_accuracy = 100 * train_correct / max(train_total, 1)
    val_accuracy = 100 * val_correct / max(val_total, 1)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss / max(len(train_loader), 1):.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_state = copy.deepcopy(model.state_dict())

# Save best validation model
model.load_state_dict(best_state)

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": train_data.classes
}, MODEL_PATH)

print(f"\nModel saved to: {os.path.abspath(MODEL_PATH)}")

# Evaluate on separate test dataset
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        test_total += labels.size(0)

test_accuracy = 100 * test_correct / max(test_total, 1)

print(f"Final Test Accuracy: {test_accuracy:.2f}%")

