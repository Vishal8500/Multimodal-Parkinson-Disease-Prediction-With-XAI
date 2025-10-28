# use_saved_ae_pipeline.py
# Load existing AE .pt (state_dict), extract embeddings, cluster, train classifier, SHAP, IG.
# No AE retraining. Hard-coded paths/params. Windows-safe.

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
from sklearn.metrics import silhouette_score, accuracy_score

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

PREFERRED_NUM_WORKERS = 4
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

# Clustering K search
K_SEARCH = True
K_MIN = 2
K_MAX = 6
DEFAULT_K = 2

# Explainability (reduce for speed if needed)
NUM_SHAP_BACKGROUND = 100
NUM_SHAP_EXPL = 50
NUM_IG_SAMPLES = 6

# Deterministic
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
# ----------------------------------------------------

# Windows: avoid DataLoader pickling issues
if os.name == "nt":
    WORKERS = 0
else:
    WORKERS = PREFERRED_NUM_WORKERS

# ---------------- Utilities -----------------
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def parse_subject(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r'^(GaCo\d+|GaPt\d+|JuCo\d+)', base, re.IGNORECASE)
    return m.group(1) if m else base.split('_')[0]

def _is_float(s):
    try:
        float(s); return True
    except:
        return False

def read_txt_file(path):
    """Robust loader that tolerates occasional non-numeric lines."""
    try:
        arr = np.loadtxt(path)
    except Exception:
        with open(path, 'r', errors='ignore') as f:
            lines = f.readlines()
        numeric_lines = []
        for ln in lines:
            parts = ln.strip().split()
            numeric_parts = [tok for tok in parts if _is_float(tok)]
            if len(numeric_parts) >= 2:
                numeric_lines.append(" ".join(numeric_parts))
        if not numeric_lines:
            raise ValueError(f"No numeric data in {path}")
        arr = np.loadtxt(StringIO("\n".join(numeric_lines)))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # drop time column
    return arr[:, 1:].astype(np.float32)

def sliding_windows(data, win_len=WINDOW_SAMPLES, step=WINDOW_STEP):
    L = data.shape[0]
    if L < win_len:
        pad = np.zeros((win_len - L, data.shape[1]), dtype=data.dtype)
        data = np.concatenate([data, pad], axis=0)
        L = win_len
    starts = list(range(0, L - win_len + 1, step))
    return [(s, s+win_len, data[s:s+win_len]) for s in starts]

# ---------------- Dataset -----------------
class WindowDataset(Dataset):
    def _init_(self, windows):
        self.samples = [w[2].T.copy() for w in windows]  # (channels, time)
    def _len_(self): return len(self.samples)
    def _getitem_(self, idx): return torch.from_numpy(self.samples[idx]).float()

class WindowMetaDataset(Dataset):
    def _init_(self, windows_with_meta):
        self.data = []; self.meta = []
        for (s,e,arr,fname,subject) in windows_with_meta:
            self.data.append(arr.T.copy()); self.meta.append((fname,subject,s,e))
    def _len_(self): return len(self.data)
    def _getitem_(self, idx):
        return torch.from_numpy(self.data[idx]).float(), self.meta[idx]

# top-level collate (picklable)
def emb_collate(batch):
    tensors = [item[0] for item in batch]
    metas = [item[1] for item in batch]
    return torch.stack(tensors, dim=0), metas

# ---------------- Model -----------------
class ConvBlock(nn.Module):
    def _init_(self, in_ch, out_ch, kernel_size, dilation, drop_prob):
        super()._init_()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch); self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(drop_prob)
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.out_act = nn.ReLU()
    def forward(self, x):
        out = self.conv1(x); out = self.bn1(out); out = self.act1(out)
        out = self.dropout(out)
        out = self.conv2(out); out = self.bn2(out)
        res = self.down(x) if self.down is not None else x
        return self.out_act(out + res)

class TCN_Autoencoder(nn.Module):
    def _init_(self, in_ch, channels, dilations, kernel_size=3, drop=0.2):
        super()._init_()
        enc_layers = []
        cur = in_ch
        for ch, d in zip(channels, dilations):
            enc_layers.append(ConvBlock(cur, ch, kernel_size, d, drop))
            cur = ch
        self.encoder = nn.Sequential(*enc_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.decoder_fc = nn.Sequential(
            nn.Linear(cur, cur*4), nn.ReLU(),
            nn.Linear(cur*4, cur*WINDOW_SAMPLES)
        )
        self.reconstruct_conv = nn.Conv1d(cur, in_ch, kernel_size=1)
    def forward(self, x):
        z = self.encoder(x)
        z_pool = self.pool(z).squeeze(-1)
        dec = self.decoder_fc(z_pool).view(-1, z.shape[1], WINDOW_SAMPLES)
        out = self.reconstruct_conv(dec)
        return out, z_pool

class Classifier(nn.Module):
    def _init_(self, in_ch, n_classes):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x): return self.net(x)

# ---------------- Training util (tqdm) -----------------
def train_classifier(clf, Xtr, ytr, Xva, yva, epochs=EPOCHS_CLF, lr=LR_CLF, patience=PATIENCE_CLF):
    clf.to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()
    best_acc = -1.0; no_imp = 0; best_state = None
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long())
    va_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(yva).long())
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    scaler = torch.amp.GradScaler() if DEVICE.startswith("cuda") else None

    pbar = tqdm(range(1, epochs+1), desc="Classifier Training", unit="epoch")
    for ep in pbar:
        clf.train(); total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast(device_type="cuda" if DEVICE.startswith("cuda") else "cpu", enabled=(scaler is not None)):
                logits = clf(xb); loss = crit(logits, yb)
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0); scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0); opt.step()
            total_loss += loss.item() * xb.size(0)
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
        lr_now = opt.param_groups[0]['lr']

        history["epoch"].append(ep); history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc); history["lr"].append(lr_now)
        pbar.set_postfix({"train_loss": f"{avg_loss:.6f}", "val_loss": f"{val_loss:.6f}", "val_acc": f"{val_acc:.4f}"})

        if val_acc > best_acc + 1e-8:
            best_acc = val_acc; best_state = clf.state_dict(); no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            pbar.write("[CLF] Early stopping"); break

    if best_state is not None:
        clf.load_state_dict(best_state)
    return clf, history

# ---------------- Main -----------------
def main():
    seed_everything()

    # 1) discover files
    all_txt = sorted(glob(os.path.join(DATA_DIR, "", "*.txt"), recursive=True))
    files = [p for p in all_txt if os.path.basename(p).startswith(("GaCo","GaPt","JuCo"))]
    print(f"Found {len(files)} candidate data files under {DATA_DIR}")

    all_windows_meta = []
    skipped = []
    for fp in tqdm(files, desc="Reading files"):
        try:
            arr = read_txt_file(fp)
        except Exception as e:
            print(f"Skipping {fp} (parse error: {e})"); skipped.append(fp); continue
        subj = parse_subject(fp)
        for (s,e,win) in sliding_windows(arr):
            all_windows_meta.append((s,e,win, os.path.basename(fp), subj))
    print(f"Total windows: {len(all_windows_meta)}; skipped files: {len(skipped)}")
    if len(all_windows_meta) == 0:
        raise RuntimeError("No windows extracted. Check DATA_DIR and file formats.")

    # 2) Fit or load scaler (we recompute here and save)
    X_flat = np.stack([w[2].T.ravel() for w in all_windows_meta], axis=0)
    scaler = StandardScaler().fit(X_flat)
    # save scaler
    with open(SAVE_SCALER_PATH, "wb") as fh:
        pickle.dump(scaler, fh)
    for i in range(len(all_windows_meta)):
        s,e,arr,fname,subj = all_windows_meta[i]
        arr_scaled = scaler.transform(arr.T.ravel()[None,:]).reshape(arr.T.shape).T
        all_windows_meta[i] = (s,e,arr_scaled,fname,subj)
    print("Saved scaler ->", SAVE_SCALER_PATH)

    # 3) load AE weights (no training)
    in_ch = all_windows_meta[0][2].shape[1]
    ae = TCN_Autoencoder(in_ch, TCN_ENCODER_CHANNELS, DILATIONS, KERNEL_SIZE, DROP_PROB)
    if not os.path.exists(AE_WEIGHTS):
        raise FileNotFoundError(f"AE weights not found at {AE_WEIGHTS}")
    state = torch.load(AE_WEIGHTS, map_location=DEVICE)
    loaded = False
    # prefer state_dict
    if isinstance(state, dict):
        try:
            ae.load_state_dict(state)
            loaded = True
        except Exception:
            # maybe state contains keys prefixed (e.g., 'module.')
            new_state = { (k.replace("module.", "") if k.startswith("module.") else k): v for k,v in state.items() }
            try:
                ae.load_state_dict(new_state)
                loaded = True
            except Exception as e2:
                print("Failed to load state_dict directly:", e2)
    if not loaded:
        # try loading full model object (less common)
        try:
            ae = state
            loaded = True
        except Exception as e:
            raise RuntimeError("Failed to load AE weights. Ensure AE_WEIGHTS is a state_dict saved from same architecture.") from e
    ae.to(DEVICE).eval()
    print("Loaded AE weights from", AE_WEIGHTS)

    # 4) Extract embeddings
    print("Extracting embeddings...")
    ds_meta = WindowMetaDataset([(s,e,arr,fname,subj) for (s,e,arr,fname,subj) in all_windows_meta])
    emb_loader = DataLoader(ds_meta, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True, collate_fn=emb_collate)
    embeddings = []; meta = []
    ae.to(DEVICE).eval()
    with torch.no_grad():
        for xb, batch_meta in emb_loader:
            xb = xb.to(DEVICE)
            _, z = ae(xb)
            embeddings.append(z.cpu().numpy()); meta.extend(batch_meta)
    embeddings = np.vstack(embeddings)
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(OUT_DIR, "feature_vectors.npy"), embeddings)
    print("Saved embeddings (shape {}) and feature_vectors.npy".format(embeddings.shape))

    # save meta CSV
    rows_meta = [(m[0], m[1], m[2], m[3]) for m in meta]
    df_meta = pd.DataFrame(rows_meta, columns=["filename","subject","win_start","win_end"])
    df_meta.to_csv(os.path.join(OUT_DIR, "windows_meta.csv"), index=False)

    # 5) PCA + clustering (K-search)
    emb_pca = PCA(n_components=min(50, embeddings.shape[1])).fit_transform(embeddings)
    best_k = None; best_sil = -999.0
    if K_SEARCH:
        print("Searching for best K by silhouette score...")
        for k in range(K_MIN, K_MAX+1):
            try:
                km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(emb_pca)
                labels_k = km.labels_
                sil = silhouette_score(emb_pca, labels_k) if len(set(labels_k))>1 else -1.0
                print(f"K={k} silhouette={sil:.4f}")
                if sil > best_sil:
                    best_sil = sil; best_k = k
            except Exception as e:
                print(f"K={k} failed: {e}")
        if best_k is None:
            best_k = DEFAULT_K
        print(f"Selected K={best_k} with silhouette={best_sil:.4f}")
    else:
        best_k = DEFAULT_K

    kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit(emb_pca)
    cluster_ids = kmeans.labels_
    np.save(os.path.join(OUT_DIR, "clusters.npy"), cluster_ids)
    np.save(os.path.join(OUT_DIR, "labels.npy"), cluster_ids)
    np.savez_compressed(os.path.join(OUT_DIR, "features_and_labels.npz"), X=embeddings, y=cluster_ids, meta=meta)
    print("Saved clustering outputs")

    # csv summary
    rows = [(fname, subj, s, e, int(cid)) for (s,e,arr,fname,subj), cid in zip(all_windows_meta, cluster_ids)]
    df = pd.DataFrame(rows, columns=["filename","subject","win_start","win_end","cluster"])
    df.to_csv(os.path.join(OUT_DIR, "summary_clusters.csv"), index=False)
    df.groupby(["subject","cluster"]).size().unstack(fill_value=0).to_csv(os.path.join(OUT_DIR, "subject_cluster_counts.csv"))
    print("Saved CSV summaries")

    # cluster counts plot
    plt.figure(figsize=(6,4))
    df['cluster'].value_counts().sort_index().plot(kind='bar')
    plt.xlabel("Cluster"); plt.ylabel("Count"); plt.title("Cluster counts")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "cluster_counts.png")); plt.close()

    # 6) Classifier fine-tune on pseudo-labels
    clf_model = None; clf_history = None
    if DO_CLASSIFIER_FINETUNE:
        print("Fine-tuning classifier on pseudo-labels...")
        X = embeddings; y = cluster_ids
        idx = np.arange(len(X)); np.random.RandomState(SEED).shuffle(idx)
        split = int((1.0 - CLASSIFIER_VAL_SPLIT) * len(X))
        tr_idx, va_idx = idx[:split], idx[split:]
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        clf = Classifier(X.shape[1], n_classes=len(np.unique(y)))
        clf, clf_history = train_classifier(clf, Xtr, ytr, Xva, yva, epochs=EPOCHS_CLF, lr=LR_CLF, patience=PATIENCE_CLF)
        torch.save(clf.state_dict(), os.path.join(OUT_DIR, "tcn_classifier.pth"))
        clf_model = clf
        print("Saved classifier ->", os.path.join(OUT_DIR, "tcn_classifier.pth"))

        # classifier plots: train vs val loss, val acc, lr vs acc
        if clf_history and len(clf_history["epoch"])>0:
            plt.figure(figsize=(6,4))
            plt.plot(clf_history["epoch"], clf_history["train_loss"], marker='o', label='train_loss')
            plt.plot(clf_history["epoch"], clf_history["val_loss"], marker='o', label='val_loss')
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Classifier Train vs Val Loss")
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "clf_loss_train_vs_val.png")); plt.close()

            plt.figure(figsize=(6,4))
            plt.plot(clf_history["epoch"], clf_history["val_acc"], marker='o')
            plt.xlabel("Epoch"); plt.ylabel("Val Accuracy"); plt.title("Classifier Val Accuracy")
            plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "clf_val_acc.png")); plt.close()

            fig, ax1 = plt.subplots(figsize=(6,4))
            ax1.plot(clf_history["epoch"], clf_history["val_acc"], marker='o', label='val_acc')
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Accuracy")
            ax2 = ax1.twinx()
            ax2.plot(clf_history["epoch"], clf_history["lr"], marker='x', color='orange', label='lr')
            ax2.set_ylabel("Learning Rate")
            ax1.set_title("Val Accuracy and LR vs Epochs")
            ax1.grid(True)
            fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "clf_lr_vs_acc.png")); plt.close()

            print("Saved classifier training plots")

    # save epoch counts
    clf_epochs_done = len(clf_history.get("epoch", [])) if (DO_CLASSIFIER_FINETUNE and clf_history is not None) else 0
    with open(os.path.join(OUT_DIR, "epoch_counts.json"), "w") as fh:
        json.dump({"ae_reused": True, "clf_epochs": clf_epochs_done}, fh)
    print("Saved epoch_counts.json")

    # ---------------- Define human-friendly channel names (based on your dataset) ----------------
    CHANNEL_NAMES = [
        "VGRF1_left","VGRF2_left","VGRF3_left","VGRF4_left",
        "VGRF5_left","VGRF6_left","VGRF7_left","VGRF8_left",
        "VGRF1_right","VGRF2_right","VGRF3_right","VGRF4_right",
        "VGRF5_right","VGRF6_right","VGRF7_right","VGRF8_right",
        "TotalForce_left","TotalForce_right"
    ]

    # 7) SHAP on RAW windows (map back to sensor names) — CPU-friendly & safe fallbacks
    if clf_model is not None:
        try:
            print("Preparing SHAP on RAW windows (CPU; small background/expl for memory safety).")
            # safe small defaults (adjust upward if you have lots of RAM)
            NUM_SHAP_RAW_BACKGROUND = min(20, len(all_windows_meta))
            NUM_SHAP_RAW_EXPL = min(10, len(all_windows_meta))

            # helper to flatten windows into (n, channels*time)
            def windows_to_flat(idxs):
                arrs = []
                for i in idxs:
                    s,e,arr,fname,subj = all_windows_meta[i]
                    arrs.append(arr.T.copy().ravel())
                return np.stack(arrs, axis=0)

            total_windows = len(all_windows_meta)
            rng = np.random.RandomState(SEED + 2025)
            bg_idx = rng.choice(total_windows, size=NUM_SHAP_RAW_BACKGROUND, replace=False)
            # choose expl indexes excluding background
            remaining = [i for i in range(total_windows) if i not in set(bg_idx)]
            expl_idx = rng.choice(remaining, size=min(NUM_SHAP_RAW_EXPL, len(remaining)), replace=False)

            bg = windows_to_flat(bg_idx)
            expl_raw = windows_to_flat(expl_idx)

            # Create CPU predictor wrapper (move models to CPU for SHAP to avoid GPU OOM)
            device_for_shap = "cpu"
            ae_cpu = TCN_Autoencoder(in_ch, TCN_ENCODER_CHANNELS, DILATIONS, KERNEL_SIZE, DROP_PROB)
            ae_cpu.load_state_dict({k: v.cpu() for k, v in ae.state_dict().items()})
            ae_cpu.to(device_for_shap).eval()
            clf_cpu = Classifier(embeddings.shape[1], n_classes=len(np.unique(cluster_ids)))
            clf_cpu.load_state_dict({k: v.cpu() for k, v in clf_model.state_dict().items()})
            clf_cpu.to(device_for_shap).eval()

            def predict_proba_np_cpu(x_np):
                # x_np: (n_samples, channels*time)
                xb = torch.from_numpy(x_np.astype(np.float32)).to(device_for_shap)
                n = xb.shape[0]
                CHANNELS_LOCAL = all_windows_meta[0][2].shape[1]
                xb = xb.view(n, CHANNELS_LOCAL, WINDOW_SAMPLES)
                with torch.no_grad():
                    z = ae_cpu.encoder(xb)
                    z_pool = ae_cpu.pool(z).squeeze(-1)
                    logits = clf_cpu(z_pool)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                return probs

            # Run shap.Explainer (Independent masker) on CPU - less memory-hungry than permutation explainer
            try:
                masker = shap.maskers.Independent(bg)
                explainer = shap.Explainer(predict_proba_np_cpu, masker)
                sv_obj = explainer(expl_raw)
                vals = np.array(sv_obj.values)
            except Exception as e_sh:
                print("shap.Explainer failed or too heavy; falling back to KernelExplainer (CPU). Reason:", e_sh)
                # KernelExplainer is slower but more general; run on CPU with tiny expl set
                expl_kernel = shap.KernelExplainer(predict_proba_np_cpu, bg)
                # set nsamples small to limit compute
                vals = np.array(expl_kernel(expl_raw, nsamples=100).values)

            print("DEBUG: raw SHAP vals shape:", vals.shape, "expl_raw shape:", expl_raw.shape)

            # Normalize to (n_samples, n_features, n_classes)
            if vals.ndim == 3:
                n_samples_s, n_features_s, n_classes_s = vals.shape
            elif vals.ndim == 2:
                n_samples_s, n_features_s = vals.shape
                n_classes_s = 1
                vals = vals.reshape(n_samples_s, n_features_s, n_classes_s)
            else:
                raise RuntimeError(f"Unexpected SHAP shape: {vals.shape}")

            # Map features -> channels*time
            CHANNELS_READ = all_windows_meta[0][2].shape[1]
            expected = CHANNELS_READ * WINDOW_SAMPLES
            if n_features_s != expected:
                # attempt transpose or try to infer
                if n_features_s == WINDOW_SAMPLES * CHANNELS_READ:
                    pass
                else:
                    raise RuntimeError(f"SHAP features ({n_features_s}) != CHANNELS*WINDOW_SAMPLES ({expected}). Cannot map to channel names safely.")

            vals_reshaped = vals.reshape(n_samples_s, CHANNELS_READ, WINDOW_SAMPLES, n_classes_s)

            # Aggregate: mean absolute across time -> (n_samples, channels, n_classes), then mean across samples
            channel_by_sample_by_class = np.mean(np.abs(vals_reshaped), axis=2)
            channel_imp_mean = channel_by_sample_by_class.mean(axis=0)

            # Save CSV with descriptive names
            col_names = [f"class_{c}" for c in range(channel_imp_mean.shape[1])]
            df_imp = pd.DataFrame(channel_imp_mean, index=CHANNEL_NAMES, columns=col_names)
            csv_out = os.path.join(OUT_DIR, "shap_channel_importances.csv")
            df_imp.to_csv(csv_out)
            print("Saved channel importances CSV ->", csv_out)

            # Per-class barplots with descriptive channel names
            for c in range(channel_imp_mean.shape[1]):
                imp = channel_imp_mean[:, c]
                plt.figure(figsize=(12,5))
                x = np.arange(len(CHANNEL_NAMES))
                plt.bar(x, imp)
                plt.xticks(x, CHANNEL_NAMES, rotation=45, ha='right')
                plt.xlabel("Channels (VGRF sensors & totals)")
                plt.ylabel("Mean |SHAP| (aggregated across time & samples)")
                plt.title(f"SHAP Channel Importances (class {c})")
                plt.tight_layout()
                outp = os.path.join(OUT_DIR, f"shap_channel_importances_class{c}.png")
                plt.savefig(outp); plt.close()
                print("Saved", outp)

            # Heatmap for first class across samples
            chosen_class = 0
            heat = channel_by_sample_by_class[:, :, chosen_class]  # (n_samples, channels)
            plt.figure(figsize=(12,6))
            plt.imshow(heat, aspect='auto', origin='lower')
            plt.colorbar(label='|SHAP| aggregated over time')
            plt.xticks(np.arange(len(CHANNEL_NAMES)), CHANNEL_NAMES, rotation=45, ha='right')
            plt.xlabel('Channel'); plt.ylabel('Explainee sample index')
            plt.title(f"Per-sample per-channel |SHAP| (class {chosen_class})")
            heat_out = os.path.join(OUT_DIR, f"shap_raw_channel_heatmap_class{chosen_class}.png")
            plt.tight_layout(); plt.savefig(heat_out); plt.close()
            print("Saved", heat_out)

        except Exception as e:
            print("RAW SHAP failed or OOM prevented run:", e)
            import traceback; traceback.print_exc()
            print("Tip: reduce NUM_SHAP_RAW_BACKGROUND/NUM_SHAP_RAW_EXPL and ensure enough CPU RAM.")

    # 8) Integrated Gradients on raw windows (with descriptive Y-axis labels)
    if clf_model is not None:
        try:
            print("Computing Integrated Gradients (IG) for example windows ...")
            class CombinedModel(nn.Module):
                def _init_(self, ae_model, clf_model):
                    super()._init_()
                    self.ae = ae_model
                    self.clf = clf_model
                def forward(self, x):
                    z = self.ae.encoder(x)
                    z_pool = self.ae.pool(z).squeeze(-1)
                    logits = self.clf(z_pool)
                    return logits

            combined = CombinedModel(ae, clf_model).to(DEVICE).eval()
            ig = IntegratedGradients(combined)

            total = len(all_windows_meta)
            step = max(1, total // NUM_IG_SAMPLES)
            example_idxs = [i*step for i in range(NUM_IG_SAMPLES) if i*step < total]
            for j, idx in enumerate(example_idxs):
                s,e,arr,fname,subj = all_windows_meta[idx]
                arr_in = arr.T.copy()
                xb = torch.from_numpy(arr_in).unsqueeze(0).float().to(DEVICE)
                baseline = torch.zeros_like(xb).to(DEVICE)
                with torch.no_grad():
                    logits = combined(xb)
                    pred = int(logits.argmax(dim=1).item())
                attributions, delta = ig.attribute(xb, baselines=baseline, target=pred, return_convergence_delta=True)
                attr = attributions[0].cpu().numpy()
                plt.figure(figsize=(10,4))
                plt.imshow(attr, aspect='auto', origin='lower')
                plt.colorbar(label='IG attribution')
                plt.xlabel('Time samples')
                try:
                    plt.yticks(np.arange(len(CHANNEL_NAMES)), CHANNEL_NAMES)
                except Exception:
                    plt.ylabel('Channels')
                plt.title(f"IG heatmap: file={fname} subj={subj} pred_class={pred}")
                outname = os.path.join(OUT_DIR, f"ig_heatmap_{j}_{fname}_c{pred}.png")
                plt.tight_layout(); plt.savefig(outname); plt.close()
                print("Saved IG heatmap ->", outname)
        except Exception as e:
            print("IG failed:", e)
            import traceback; traceback.print_exc()

    print("All done. Outputs saved in:", OUT_DIR)


if __name__ == "__main__":
    main()