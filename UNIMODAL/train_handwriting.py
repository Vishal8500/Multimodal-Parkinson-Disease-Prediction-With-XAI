"""
Train ResNet50 model for Parkinson prediction on spiral/wave images using GPU.
Data folder structure:

Dataset/
  healthy/
  parkinson/

The script automatically splits into train/val/test sets and trains ResNet50 with no command line arguments. 
At the end, it saves:
 - best model as .pkl and .pth
 - feature vectors and labels as separate .npy files
 - training & validation loss/accuracy plots in './plots'
Uses tqdm progress bars for training and evaluation loops.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------- Configuration -----------------
DATA_DIR = './Dataset'  # path to your dataset folder with healthy/parkinson
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
OUT_DIR = './output_resnet50'
PLOTS_DIR = './plots'
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
FREEZE_LAYERS = True  # Freeze early layers

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ----------------- Data Preparation -----------------
def build_transforms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=build_transforms(IMG_SIZE, train=True))
classes = full_dataset.classes
num_classes = len(classes)
print('Classes:', classes)

full_len = len(full_dataset)
val_len = int(full_len * VAL_SPLIT)
test_len = int(full_len * TEST_SPLIT)
train_len = full_len - val_len - test_len

train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len])
val_ds.dataset.transform = build_transforms(IMG_SIZE, train=False)
test_ds.dataset.transform = build_transforms(IMG_SIZE, train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print('Train samples:', len(train_ds), 'Val samples:', len(val_ds), 'Test samples:', len(test_ds))

# ----------------- Model -----------------
model = models.resnet50(pretrained=True)
if FREEZE_LAYERS:
    for param in model.parameters():
        param.requires_grad = False

model.fc = nn.Linear(2048, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ----------------- Training & Evaluation Functions -----------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc='Training', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device, desc='Evaluating'):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# ----------------- Feature Extractor -----------------
def extract_features(model, loader, device):
    model.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Extracting Features', leave=False):
            imgs = imgs.to(device)
            feats = model.forward(imgs)
            features_list.append(feats.detach().cpu().numpy())
            labels_list.append(labels.numpy())
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels

# ----------------- Main Training Loop -----------------
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(1, EPOCHS + 1):
    print(f'Epoch {epoch}/{EPOCHS}')
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc='Validation')
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUT_DIR, 'best_resnet50.pth'))
        pkl_path = os.path.join(OUT_DIR, 'best_resnet50.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({'model_state_dict': model.state_dict(), 'classes': classes}, f)
        print('Saved best model as .pth and .pkl')

# ----------------- Feature Extraction -----------------
model.load_state_dict(torch.load(os.path.join(OUT_DIR, 'best_resnet50.pth'), map_location=device))
features, labels_np = extract_features(model, DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False), device)
np.save(os.path.join(OUT_DIR, 'features.npy'), features)
np.save(os.path.join(OUT_DIR, 'labels.npy'), labels_np)
print('Saved feature vectors to features.npy and labels to labels.npy')

# ----------------- Test Evaluation -----------------
test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc='Testing')
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# ----------------- Plotting Training Curves -----------------
epochs_range = range(1, EPOCHS + 1)

# Loss Curve
plt.figure(figsize=(10,6))
plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
plt.plot(epochs_range, val_losses, label='Val Loss', marker='s')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'loss_curve.png'))
plt.close()

# Accuracy Curve
plt.figure(figsize=(10,6))
plt.plot(epochs_range, train_accs, label='Train Accuracy', marker='o')
plt.plot(epochs_range, val_accs, label='Val Accuracy', marker='s')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_curve.png'))
plt.close()

# Learning Rate vs Validation Accuracy (constant LR here, modify if scheduler used)
plt.figure(figsize=(10,6))
plt.plot([LR]*EPOCHS, val_accs, label='Validation Accuracy', marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.title("Learning Rate vs Validation Accuracy")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'lr_vs_accuracy.png'))
plt.close()

print("Saved all training curves in:", PLOTS_DIR)
