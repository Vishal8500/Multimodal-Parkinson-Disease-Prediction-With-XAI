import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torchvision import models, transforms
from xgboost import XGBClassifier
import joblib
from PIL import Image
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings("ignore")

# ============================
# Hard-coded input paths
# ============================
audio_file = r"D:\SSDM\ssdm da-2\TRIMODAL\subject1\ID02_pd_2_0_0.wav"
gait_txt   = r"D:\SSDM\ssdm da-2\TRIMODAL\subject1\GaCo01_01.txt"
handwriting_img = r"D:\SSDM\ssdm da-2\TRIMODAL\subject1\Parkinson7.png"

# ============================
# Model and object paths
# ============================
SPEECH_MODEL_PATH = r"D:\SSDM\ssdm da-2\output_EFFICIENTb0_Speech\best_efficientnet_audio.pth"
HAND_MODEL_PATH   = r"D:\SSDM\ssdm da-2\output_resnet50\best_resnet50.pth"
GAIT_SCALER_PATH  = "D:\SSDM\ssdm da-2\outputs_from_saved_ae\scaler.pkl"
TRIMODAL_MODEL_PKL = "./TRIMODAL/xgb_trimodal_model.pkl"
SPEECH_PCA_PATH   = "./TRIMODAL/speech_pca.pkl"
HAND_PCA_PATH     = "./TRIMODAL/handwriting_pca.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# --- 1. Speech Model ---
# ============================
def load_speech_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(SPEECH_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def preprocess_speech(wav_path, sample_rate=16000, n_mels=128, fixed_length=5):
    waveform, sr = torchaudio.load(wav_path)
    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    waveform = waveform.mean(dim=0)
    fixed_len = int(sample_rate * fixed_length)
    if waveform.shape[0] < fixed_len:
        pad = fixed_len - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:fixed_len]
    waveform = waveform.unsqueeze(0)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    mel_db = mel_db.repeat(3, 1, 1)
    return mel_db.unsqueeze(0).to(DEVICE)

def extract_speech_feature(model, wav_path):
    x = preprocess_speech(wav_path)
    with torch.no_grad():
        feat = model.features(x).mean(dim=[2, 3]).cpu().numpy().flatten()
    return feat  # 1280D

# ============================
# --- 2. Handwriting Model ---
# ============================
def load_hand_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 2)
    model.load_state_dict(torch.load(HAND_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def preprocess_image(img_path, size=224):
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return tfm(img).unsqueeze(0).to(DEVICE)

def extract_hand_feature(model, img_path):
    x = preprocess_image(img_path)
    with torch.no_grad():
        feat = model.forward(x)
        feat_vec = feat.cpu().numpy().flatten()
    return feat_vec  # 2048D

# ============================
# --- 3. Gait Preprocessing ---
# ============================
def load_scaler():
    with open(GAIT_SCALER_PATH, "rb") as f:
        return pickle.load(f)

def read_gait_file(txt_path):
    arr = np.loadtxt(txt_path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = arr[:, 1:]  # drop time col if present
    return arr  # (N, 18)

def extract_gait_feature(txt_path):
    scaler = load_scaler()
    data = read_gait_file(txt_path)
    WINDOW_SAMPLES = 200
    if data.shape[0] < WINDOW_SAMPLES:
        pad = np.zeros((WINDOW_SAMPLES - data.shape[0], data.shape[1]), dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)
    else:
        data = data[:WINDOW_SAMPLES]
    flat = data.T.ravel()[None, :]  # (1, 3600)
    scaled = scaler.transform(flat)
    return scaled.flatten()[:18]  # match 18 features used in trimodal training

# ============================
# --- 4. Trimodal Fusion ---
# ============================
def load_trimodal_model():
    return joblib.load(TRIMODAL_MODEL_PKL)

def load_pca(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"PCA file not found: {path}")

# ============================
# --- 5. Predict Function ---
# ============================
def predict_parkinson():
    print("\nLoading models...")
    speech_model = load_speech_model()
    hand_model   = load_hand_model()
    trimodal_model = load_trimodal_model()
    speech_pca   = load_pca(SPEECH_PCA_PATH)
    hand_pca     = load_pca(HAND_PCA_PATH)

    print("Extracting features...")
    speech_feat = extract_speech_feature(speech_model, audio_file)
    hand_feat   = extract_hand_feature(hand_model, handwriting_img)
    gait_feat   = extract_gait_feature(gait_txt)

    print(f"Feature shapes -> Speech: {speech_feat.shape}, Hand: {hand_feat.shape}, Gait: {gait_feat.shape}")

    # Apply PCA reductions
    speech_feat_reduced = speech_pca.transform(speech_feat.reshape(1, -1))  # -> (1, 50)
    hand_feat_reduced   = hand_pca.transform(hand_feat.reshape(1, -1))      # -> (1, 2)
    gait_feat_final     = gait_feat.reshape(1, -1)                          # -> (1, 18)

    print(f"Reduced -> Speech: {speech_feat_reduced.shape}, Hand: {hand_feat_reduced.shape}, Gait: {gait_feat_final.shape}")

    # Fuse all
    fused = np.concatenate(
        [speech_feat_reduced.flatten(), gait_feat_final.flatten(), hand_feat_reduced.flatten()],
        axis=0
    ).reshape(1, -1)

    print(f"Final fused feature vector shape: {fused.shape}")

    print("Running trimodal prediction...")
    proba = trimodal_model.predict_proba(fused)[0, 1]
    pred = "Parkinson" if proba >= 0.5 else "Healthy"

    print("\n🧠 Trimodal Parkinson Prediction")
    print(f"Speech file: {audio_file}")
    print(f"Handwriting: {handwriting_img}")
    print(f"Gait file  : {gait_txt}")
    print(f"Predicted Class: {pred}")
    print(f"Probability (Parkinson): {proba:.4f}\n")

# ============================
# --- Main Entry ---
# ============================
if __name__ == "__main__":
    predict_parkinson()
