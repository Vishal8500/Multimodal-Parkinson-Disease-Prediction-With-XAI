# use_saved_ae_pipeline.py
# Load existing AE .pt (state_dict), extract embeddings, cluster, train classifier, SHAP, IG.
# Now also: Confusion matrix + classification report

import os
import re
import time
import random
from glob import glob
from io import StringIO
import pickle
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

from captum.attr import IntegratedGradients
import shap

# ----------------- HARD-CODED CONFIG -----------------
DATA_DIR = r"D:\Parkinson Disease Multimodal\gait-in-parkinsons-disease-1.0.0"
OUT_DIR = "./outputs_from_saved_ae"
os.makedirs(OUT_DIR, exist_ok=True)

AE_WEIGHTS = "./outputs/tcn_autoencoder.pt"   # <- set to your saved AE .pt
SAVE_SCALER_PATH = os.path.join(OUT_DIR, "scaler.pkl")

SAMPLE_RATE = 100
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)  # 200
WINDOW_STEP = WINDOW_SAMPLES // 2               # 50% overlap

BATCH_SIZE = 64
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Classifier fine-tune
DO_CLASSIFIER_FINETUNE = True
EPOCHS_CLF = 40
PATIENCE_CLF = 6
LR_CLF = 1e-3
CLASSIFIER_VAL_SPLIT = 0.2

# Model (must match AE)
TCN_ENCODER_CHANNELS = [64, 64, 128]
DILATIONS = [1, 2, 4]
KERNEL_SIZE = 3
DROP_PROB = 0.2

# ---------------- Utilities -----------------
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ---------------- Model -----------------
class Classifier(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x): return self.net(x)

# ---------------- Training util -----------------
def train_classifier(clf, Xtr, ytr, Xva, yva, epochs=EPOCHS_CLF, lr=LR_CLF, patience=PATIENCE_CLF):
    clf.to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()
    best_acc = -1.0; no_imp = 0; best_state = None
    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long())
    va_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(yva).long())
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    for ep in range(1, epochs+1):
        clf.train(); total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = clf(xb); loss = crit(logits, yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step(); total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(tr_loader.dataset)

        # validation
        clf.eval(); preds=[]; trues=[]; val_loss_total=0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = clf(xb)
                val_loss_total += crit(logits, yb).item() * xb.size(0)
                preds.append(logits.argmax(dim=1).cpu().numpy()); trues.append(yb.cpu().numpy())
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        val_acc = accuracy_score(trues, preds)
        val_loss = val_loss_total / len(va_loader.dataset)

        print(f"[Epoch {ep}] Train Loss={avg_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_acc + 1e-8:
            best_acc = val_acc; best_state = clf.state_dict(); no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            print("[CLF] Early stopping"); break

    if best_state is not None:
        clf.load_state_dict(best_state)
    return clf, (Xva, yva)

# ---------------- Main -----------------
def main():
    seed_everything()

    # --- Load embeddings & pseudo-labels ---
    embeddings = np.load(os.path.join(OUT_DIR, "embeddings.npy"))
    cluster_ids = np.load(os.path.join(OUT_DIR, "clusters.npy"))
    print("Embeddings:", embeddings.shape, "Clusters:", np.unique(cluster_ids))

    # --- Split train/val ---
    idx = np.arange(len(embeddings)); np.random.shuffle(idx)
    split = int((1.0 - CLASSIFIER_VAL_SPLIT) * len(embeddings))
    tr_idx, va_idx = idx[:split], idx[split:]
    Xtr, Xva = embeddings[tr_idx], embeddings[va_idx]
    ytr, yva = cluster_ids[tr_idx], cluster_ids[va_idx]

    # --- Train classifier ---
    clf = Classifier(embeddings.shape[1], n_classes=len(np.unique(cluster_ids)))
    clf, val_data = train_classifier(clf, Xtr, ytr, Xva, yva)
    torch.save(clf.state_dict(), os.path.join(OUT_DIR, "tcn_classifier.pth"))

    # --- Evaluation: Confusion matrix + Classification report ---
    print("\n=== Evaluating Classifier on Validation Set ===")
    Xva, yva = val_data
    clf.eval()
    with torch.no_grad():
        y_pred = clf(torch.from_numpy(Xva).float().to(DEVICE)).argmax(dim=1).cpu().numpy()

    cm = confusion_matrix(yva, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Gait Classifier")
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_gait.png"), dpi=300, bbox_inches="tight")
    plt.close()

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm)
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title("Normalized Confusion Matrix (%) - Gait Classifier")
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_gait_normalized.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("\nClassification Report:")
    print(classification_report(yva, y_pred, target_names=[f"Cluster {i}" for i in np.unique(yva)]))

    print("\n✅ Confusion matrices and report saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
