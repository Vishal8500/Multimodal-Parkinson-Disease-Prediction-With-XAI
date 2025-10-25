import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchcam.methods import SmoothGradCAMpp

# ========================
# 1. Device
# ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# 2. Dataset Class
# ========================
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, type_='read_text', sample_rate=16000, n_mels=128, fixed_length=5):
        self.base_dir = base_dir
        self.type_ = type_
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fixed_length_samples = int(sample_rate * fixed_length)
        self.files, self.labels = [], []

        self.classes = ['HC', 'PD']
        for idx, cls in enumerate(self.classes):
            cls_folder = os.path.join(base_dir, type_, cls)
            for file in os.listdir(cls_folder):
                if file.lower().endswith('.wav'):
                    self.files.append(os.path.join(cls_folder, file))
                    self.labels.append(idx)

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=256
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

        # Force fixed length
        if waveform.shape[0] < self.fixed_length_samples:
            pad = self.fixed_length_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:self.fixed_length_samples]
        waveform = waveform.unsqueeze(0)

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        mel_spec_db = mel_spec_db.repeat(3, 1, 1)  # 3-channel for EfficientNet

        return mel_spec_db, label, os.path.basename(file_path)

# ========================
# 3. Load Dataset
# ========================
dataset_types = ['read_text', 'spontaneous']
all_datasets = [AudioDataset('Speech', type_=dtype) for dtype in dataset_types]
full_dataset = ConcatDataset(all_datasets)
data_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)

# ========================
# 4. Load Model
# ========================
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(r'D:\SSDM\ssdm da-2\output_EFFICIENTb0_Speech\best_efficientnet_audio.pth', map_location=device))
model.eval()
model.to(device)

# ========================
# 5. Grad-CAM Setup
# ========================
cam_extractor = SmoothGradCAMpp(model, target_layer='features.7')  # last conv block

# Output directory
os.makedirs('gradcam_outputs_speech', exist_ok=True)

# ========================
# 6. Generate Grad-CAM for all samples
# ========================
for mel_spec_db, label, fname in data_loader:
    mel_spec_db = mel_spec_db.to(device)
    output = model(mel_spec_db)
    pred_class = output.argmax(dim=1).item()

    activation_map = cam_extractor(pred_class, output)
    cam = activation_map[0].cpu().detach().numpy()

    # Reduce extra dimensions if present
    if cam.ndim > 2:
        cam = cam.mean(axis=0)

    # Resize CAM to match Mel spectrogram
    mel_np = mel_spec_db[0, 0].cpu().numpy()
    cam_resized = cv2.resize(cam, (mel_np.shape[1], mel_np.shape[0]))

    # Plot and save
    plt.figure(figsize=(10,4))
    plt.imshow(mel_np, aspect='auto', origin='lower', cmap='viridis')
    plt.imshow(cam_resized, cmap='jet', alpha=0.5, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(f'{fname[0]} - Predicted: {pred_class}')
    save_path = os.path.join('gradcam_outputs_speech', f'{fname[0].split(".")[0]}_gradcam.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
