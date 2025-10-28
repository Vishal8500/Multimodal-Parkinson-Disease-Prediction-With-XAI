import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torchvision import models, transforms
from PIL import Image
from xgboost import XGBClassifier
import joblib
import pickle
import os
import tempfile

# ============================
# Paths (update base directory)
# ============================
BASE_DIR = r"D:\SSDM\ssdm da-2"
SPEECH_MODEL_PATH = os.path.join(BASE_DIR, "output_EFFICIENTb0_Speech", "best_efficientnet_audio.pth")
HAND_MODEL_PATH   = os.path.join(BASE_DIR, "output_resnet50", "best_resnet50.pth")
GAIT_SCALER_PATH  = os.path.join(BASE_DIR, "outputs_from_saved_ae", "scaler.pkl")
TRIMODAL_MODEL_PKL = os.path.join(BASE_DIR, "TRIMODAL", "xgb_trimodal_model.pkl")
SPEECH_PCA_PATH   = os.path.join(BASE_DIR, "TRIMODAL", "speech_pca.pkl")
HAND_PCA_PATH     = os.path.join(BASE_DIR, "TRIMODAL", "handwriting_pca.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Model Loaders
# ============================
@st.cache_resource
def load_speech_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(SPEECH_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_hand_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 2)
    model.load_state_dict(torch.load(HAND_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_trimodal_model():
    return joblib.load(TRIMODAL_MODEL_PKL)

@st.cache_resource
def load_scaler():
    with open(GAIT_SCALER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_pca(path):
    return joblib.load(path)

# ============================
# Feature Extraction
# ============================
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
    return feat

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
    return feat_vec

def read_gait_file(txt_path):
    arr = np.loadtxt(txt_path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = arr[:, 1:]  # drop time col if present
    return arr  # (N, 18)

def extract_gait_feature(txt_path, scaler):
    data = read_gait_file(txt_path)
    WINDOW_SAMPLES = 200
    if data.shape[0] < WINDOW_SAMPLES:
        pad = np.zeros((WINDOW_SAMPLES - data.shape[0], data.shape[1]), dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)
    else:
        data = data[:WINDOW_SAMPLES]
    flat = data.T.ravel()[None, :]
    scaled = scaler.transform(flat)
    return scaled.flatten()[:18]

# ============================
# Streamlit App
# ============================
st.set_page_config(page_title="Trimodal Parkinson Prediction", page_icon="🧠", layout="centered")

st.title("🧠 Trimodal Parkinson Prediction App")
st.markdown("Upload **speech (.wav)**, **handwriting image (.png/.jpg)**, and **gait (.txt)** to predict Parkinson's disease probability.")

with st.sidebar:
    st.info("📂 Model files are loaded from your local trained models.")
    st.write("This app uses:")
    st.write("- EfficientNet-B0 for Speech")
    st.write("- ResNet-50 for Handwriting")
    st.write("- XGBoost Fusion Model for Final Prediction")

speech_file = st.file_uploader("🎤 Upload Speech Audio (.wav)", type=["wav"])
hand_file = st.file_uploader("✍️ Upload Handwriting Image (.png, .jpg)", type=["png", "jpg", "jpeg"])
gait_file = st.file_uploader("🚶 Upload Gait Text File (.txt)", type=["txt"])

if st.button("🔍 Predict Parkinson Status"):
    if not (speech_file and hand_file and gait_file):
        st.error("Please upload all three inputs (speech, handwriting, and gait).")
    else:
        with st.spinner("Loading models and processing inputs..."):
            # Save temp files
            tmp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(tmp_dir, speech_file.name)
            img_path   = os.path.join(tmp_dir, hand_file.name)
            gait_path  = os.path.join(tmp_dir, gait_file.name)
            with open(audio_path, "wb") as f: f.write(speech_file.read())
            with open(img_path, "wb") as f: f.write(hand_file.read())
            with open(gait_path, "wb") as f: f.write(gait_file.read())

            # Load models
            speech_model = load_speech_model()
            hand_model   = load_hand_model()
            trimodal_model = load_trimodal_model()
            speech_pca   = load_pca(SPEECH_PCA_PATH)
            hand_pca     = load_pca(HAND_PCA_PATH)
            scaler       = load_scaler()

            # Extract features
            speech_feat = extract_speech_feature(speech_model, audio_path)
            hand_feat   = extract_hand_feature(hand_model, img_path)
            gait_feat   = extract_gait_feature(gait_path, scaler)

            # PCA reductions
            speech_feat_reduced = speech_pca.transform(speech_feat.reshape(1, -1))
            hand_feat_reduced   = hand_pca.transform(hand_feat.reshape(1, -1))
            gait_feat_final     = gait_feat.reshape(1, -1)

            fused = np.concatenate(
                [speech_feat_reduced.flatten(), gait_feat_final.flatten(), hand_feat_reduced.flatten()],
                axis=0
            ).reshape(1, -1)

            # Predict
            proba = trimodal_model.predict_proba(fused)[0, 1]
            pred_class = "Parkinson" if proba >= 0.5 else "Healthy"

        # Show result
        st.success(f"🧩 **Prediction:** {pred_class}")
        st.metric(label="Confidence (Parkinson Probability)", value=f"{proba*100:.2f}%")
        st.caption(f"Feature vector shape: {fused.shape} → Model expects 70 features.")
        st.info("✅ Prediction complete. All inputs processed successfully.")

# =======================================================
# ===============  XAI SECTION  =========================
# =======================================================
st.divider()
st.header("🔍 Explainability (XAI) Insights")

import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import torchaudio
from torchcam.methods import SmoothGradCAMpp

# ---------------- HANDWRITING GRAD-CAM ----------------
st.subheader("✍️ Handwriting Grad-CAM Visualization")

IMG_SIZE = 224

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay_heatmap(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    overlayed = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed


# Apply Grad-CAM to uploaded handwriting
pil_img = Image.open(img_path).convert("L")
input_tensor = preprocess_image(img_path)
gradcam = GradCAM(hand_model, hand_model.layer4[-1].conv3)
cam, _ = gradcam(input_tensor)
overlayed = overlay_heatmap(pil_img, cam)
st.image(overlayed, caption="Model attention on handwriting (Grad-CAM)", use_container_width=True)

# ---------------- SPEECH GRAD-CAM ----------------
st.subheader("🎤 Speech Mel-Spectrogram Grad-CAM++")

speech_model.eval()
cam_extractor = SmoothGradCAMpp(speech_model, target_layer='features.7')

mel_spec = preprocess_speech(audio_path)
output = speech_model(mel_spec)
pred_class = output.argmax(dim=1).item()
activation_map = cam_extractor(pred_class, output)[0].cpu().detach().numpy()

if activation_map.ndim > 2:
    activation_map = activation_map.mean(axis=0)

mel_np = mel_spec[0, 0].cpu().numpy()
cam_resized = cv2.resize(activation_map, (mel_np.shape[1], mel_np.shape[0]))

plt.figure(figsize=(8, 3))
plt.imshow(mel_np, aspect='auto', origin='lower', cmap='viridis')
plt.imshow(cam_resized, cmap='jet', alpha=0.5, aspect='auto', origin='lower')
plt.colorbar()
plt.title("Speech Mel-Spectrogram Grad-CAM++")
st.pyplot(plt.gcf())

# ---------------- GAIT SHAP ----------------
st.subheader("🚶 Gait Feature Importance (SHAP)")

import shap
explainer = shap.TreeExplainer(trimodal_model)
shap_vals = explainer.shap_values(fused)
gait_shap = shap_vals[0][-18:]
fig, ax = plt.subplots(figsize=(5,3))
ax.bar(range(18), np.abs(gait_shap))
ax.set_title("Gait Feature Importance (SHAP)")
ax.set_xlabel("Feature index")
st.pyplot(fig)

st.info("✅ XAI visualizations generated: Grad-CAM (Handwriting), Grad-CAM++ (Speech), SHAP (Gait).")
