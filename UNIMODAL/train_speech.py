import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import models
import torchaudio
import numpy as np
import joblib
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, Shift
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========================
# 1. Device
# ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# 2. Augmentations
# ========================
read_text_augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    Gain(min_gain_db=-3.0, max_gain_db=3.0, p=0.5),
    Shift(min_shift=-0.05, max_shift=0.05, p=0.5)
])

spontaneous_augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.02, p=0.5),
    TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
    PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
    Gain(min_gain_db=-5.0, max_gain_db=5.0, p=0.5),
    Shift(min_shift=-0.1, max_shift=0.1, p=0.5)
])

# ========================
# 3. Dataset Class
# ========================
class AudioDataset(Dataset):
    def __init__(self, base_dir, type_='read_text', sample_rate=16000,
                 n_mels=128, fixed_length=5):
        self.base_dir = base_dir
        self.type_ = type_
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fixed_length_samples = int(sample_rate * fixed_length)
        self.files = []
        self.labels = []

        self.classes = ['HC', 'PD']

        for idx, cls in enumerate(self.classes):
            cls_folder = os.path.join(base_dir, type_, cls)
            for file in os.listdir(cls_folder):
                if file.lower().endswith('.wav'):
                    self.files.append(os.path.join(cls_folder, file))
                    self.labels.append(idx)

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=256
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(file_path)
        waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)  # mono

        aug = read_text_augment if self.type_ == 'read_text' else spontaneous_augment
        waveform_np = waveform.numpy()
        waveform_aug = aug(samples=waveform_np, sample_rate=self.sample_rate)
        waveform = torch.tensor(waveform_aug)

        # force fixed length
        if waveform.shape[0] < self.fixed_length_samples:
            pad = self.fixed_length_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:self.fixed_length_samples]
        waveform = waveform.unsqueeze(0)

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        mel_spec_db = mel_spec_db.repeat(3, 1, 1)  # 3-channel

        return mel_spec_db, label

# ========================
# 4. Load Data and Split
# ========================
read_text_dataset = AudioDataset('Speech', type_='read_text')
spontaneous_dataset = AudioDataset('Speech', type_='spontaneous')
full_dataset = ConcatDataset([read_text_dataset, spontaneous_dataset])

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# ========================
# 5. Model: EfficientNet-B0
# ========================
model = models.efficientnet_b0(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ========================
# 6. Training
# ========================
best_val_acc = 0.0
feature_vectors = []
all_labels = []

epochs = 30

# Create plots directory
PLOTS_DIR = './plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

train_losses, train_accs, val_accs = [], [], []

for epoch in range(epochs):
    # ---- Train ----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for mel_spec_db, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        mel_spec_db = mel_spec_db.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(mel_spec_db)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Save feature vectors
        with torch.no_grad():
            feats = model.features(mel_spec_db).mean(dim=[2, 3])  # global avg pool
            feature_vectors.append(feats.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # ---- Validation ----
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for mel_spec_db, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
            mel_spec_db = mel_spec_db.to(device)
            labels = labels.to(device)
            outputs = model(mel_spec_db)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = val_correct / val_total
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")

    # ---- Save best model ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_efficientnet_audio.pth')
        joblib.dump(model, 'best_efficientnet_audio.pkl')
        print(f"Best model saved at epoch {epoch+1} with Val Acc: {best_val_acc:.4f}")

# ========================
# 7. Save Features
# ========================
feature_vectors = np.concatenate(feature_vectors, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
np.save('feature_vectors.npy', feature_vectors)
np.save('labels.npy', all_labels)

# ========================
# 8. Plot Training Curves
# ========================
epochs_range = range(1, epochs+1)

# Loss Curve
plt.figure(figsize=(8,6))
plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, 'train_loss_curve.png'))
plt.close()

# Accuracy Curve
plt.figure(figsize=(8,6))
plt.plot(epochs_range, train_accs, label='Train Accuracy', marker='o')
plt.plot(epochs_range, val_accs, label='Val Accuracy', marker='s')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_curve.png'))
plt.close()

print(f"Training complete. Plots saved in {PLOTS_DIR}, best model, features, and labels saved.")
