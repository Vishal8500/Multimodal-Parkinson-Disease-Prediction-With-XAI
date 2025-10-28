import os
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import torch
import shap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# =======================
# 0. CONFIG / PATHS
# =======================
SPEECH_FEATS_PATH = r"D:\SSDM\ssdm da-2\output_EFFICIENTb0_Speech\feature_vectors.npy"  # (Ns, 1280)
GAIT_FEATS_PATH   = r"D:\SSDM\ssdm da-2\output_gait\features.npy"                       # (306, 18)
HAND_FEATS_PATH   = r"D:\SSDM\ssdm da-2\output_resnet50\features.npy"                   # (Nh, d_hw) — from ResNet50
HAND_LABELS_PATH  = r"D:\SSDM\ssdm da-2\output_resnet50\labels.npy"                     # (Nh,)

SAVE_DIR = r"D:\SSDM\ssdm da-2\TRIMODAL"
os.makedirs(SAVE_DIR, exist_ok=True)

SPEECH_PCA_OUT = os.path.join(SAVE_DIR, "speech_pca.pkl")
HAND_PCA_OUT   = os.path.join(SAVE_DIR, "handwriting_pca.pkl")
XGB_PKL_OUT    = os.path.join(SAVE_DIR, "xgb_trimodal_model.pkl")
XGB_PTH_OUT    = os.path.join(SAVE_DIR, "xgb_trimodal_model.pth")  # raw booster for reference

# =======================
# 1. LOAD DATA
# =======================
speech_features = np.load(SPEECH_FEATS_PATH)   # (Ns, 1280)
gait_features   = np.load(GAIT_FEATS_PATH)     # (306, 18)
hand_features   = np.load(HAND_FEATS_PATH)     # (Nh, d_hw)
labels          = np.load(HAND_LABELS_PATH)    # (Nh,)

print("Speech:", speech_features.shape)
print("Gait:", gait_features.shape)
print("Handwriting:", hand_features.shape)
print("Labels:", labels.shape)

# =======================
# 2. ALIGN SUBJECTS (gait=306 reference)
# =======================
subjects = len(gait_features)  # 306

# ---- Speech: aggregate (mean) per-subject ----
samples_per_subject = len(speech_features) // subjects
assert samples_per_subject > 0, "Not enough speech samples to aggregate per subject."
speech_agg = [
    speech_features[i*samples_per_subject:(i+1)*samples_per_subject].mean(axis=0)
    for i in range(subjects)
]
speech_agg = np.array(speech_agg)  # (306, 1280)

# ---- PCA: Speech → 50D (save PCA) ----
pca_speech = PCA(n_components=50, random_state=42)
speech_reduced = pca_speech.fit_transform(speech_agg)  # (306, 50)
joblib.dump(pca_speech, SPEECH_PCA_OUT)
print("Saved Speech PCA ->", SPEECH_PCA_OUT)

# ---- Handwriting: stratified sample to 306 ----
hw_idx = np.arange(len(hand_features))
hw_idx_sel, _, y_sel, _ = train_test_split(
    hw_idx, labels, train_size=subjects, stratify=labels, random_state=42
)
handwriting_sel = hand_features[hw_idx_sel]  # (306, d_hw)
labels_sel = y_sel                           # (306,)

# If handwriting is already 2D (as in your ResNet-50 logits), PCA(2) will just rotate.
# If d_hw > 2, reduce to 2D; if d_hw == 1, keep 1D and we'll pad up to 2 later for fusion.
d_hw = handwriting_sel.shape[1]
target_hw_dim = 2 if d_hw >= 2 else d_hw

pca_hand = None
if d_hw >= 2:
    pca_hand = PCA(n_components=2, random_state=42)
    handwriting_reduced = pca_hand.fit_transform(handwriting_sel)  # (306, 2)
    joblib.dump(pca_hand, HAND_PCA_OUT)
    print("Saved Handwriting PCA ->", HAND_PCA_OUT)
else:
    handwriting_reduced = handwriting_sel  # (306, 1)
    # create an identity "PCA" saver for compatibility at inference (optional)
    # we’ll just skip saving if d_hw < 2
    print("Handwriting has <2 dims; skipping PCA save. Will pad at fusion.")

print("Aligned Speech:", speech_reduced.shape)
print("Aligned Gait:", gait_features.shape)
print("Aligned Handwriting (post-PCA if applied):", handwriting_reduced.shape)
print("Aligned Labels:", labels_sel.shape)

# =======================
# 3. EARLY FUSION  (50 + 18 + 2 = 70 expected)
# =======================
# If handwriting_reduced is 1D, pad a zero column to make it 2D so fused dimension matches 70.
if handwriting_reduced.shape[1] == 1:
    handwriting_reduced = np.hstack([handwriting_reduced, np.zeros((handwriting_reduced.shape[0], 1))])

X_fused = np.concatenate([speech_reduced, gait_features, handwriting_reduced], axis=1)
y = labels_sel
print("Fused features:", X_fused.shape)  # should be (306, 70)

# =======================
# 4. TRAIN/TEST SPLIT
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X_fused, y, test_size=0.4, random_state=455, stratify=y
)

# =======================
# 5. SCALE FEATURES (XGB is scale-insensitive, but we keep consistency)
# =======================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =======================
# 6. HANDLE CLASS IMBALANCE WITH SMOTE
# =======================
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print("After SMOTE -> Train shape:", X_train.shape, y_train.shape)

# =======================
# 7. scale_pos_weight (informative; XGB handles our balanced set)
# =======================
neg, pos = np.bincount(y_train)
scale = neg / pos if pos > 0 else 1.0
print(f"Class balance after SMOTE -> Healthy: {neg}, Parkinson: {pos}, scale_pos_weight: {scale:.2f}")

# =======================
# 8. TRAIN XGBOOST CLASSIFIER
# =======================
clf = XGBClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale,
    random_state=42,
    eval_metric="logloss"
)
clf.fit(X_train, y_train)

# =======================
# 9. PREDICTION (threshold = 0.5)
# =======================
y_proba = clf.predict_proba(X_test)[:, 1]
best_threshold = 0.5
y_pred = (y_proba >= best_threshold).astype(int)

# =======================
# 10. FINAL EVALUATION
# =======================
print(f"\nClassification Report (Trimodal Early Fusion + SMOTE + XGBoost, Threshold={best_threshold}):")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson"]))

# =======================
# 11. FEATURE NAMES for SHAP
# =======================
speech_names = [f"speech_{i+1}" for i in range(speech_reduced.shape[1])]   # 50
gait_names   = [f"gait_{i+1}" for i in range(gait_features.shape[1])]      # 18
hw_names     = [f"hw_{i+1}" for i in range(handwriting_reduced.shape[1])]  # 2
feature_names = speech_names + gait_names + hw_names
print("Total features:", len(feature_names))  # should be 70

# =======================
# 12. SAVE FEATURES & LABELS (pre-scaling)
# =======================
np.save(os.path.join(SAVE_DIR, "features.npy"), X_fused)
np.save(os.path.join(SAVE_DIR, "labels.npy"), y)
print(f"Saved features and labels in {SAVE_DIR}")

# =======================
# 13. SAVE MODELS
# =======================
joblib.dump(clf, XGB_PKL_OUT)
# Save raw booster (optional, for reference/portability)
torch.save(clf.get_booster().save_raw(), XGB_PTH_OUT)
print(f"Saved XGB model (.pkl and .pth) in {SAVE_DIR}")

# =======================
# 14. SHAP EXPLAINER (bar + dot)
# =======================
# Use a sample for speed
np.random.seed(42)
sel = np.random.choice(X_train.shape[0], min(100, X_train.shape[0]), replace=False)
X_explain = X_train[sel]

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_explain)

plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_explain, feature_names=feature_names, plot_type="bar", show=False)
plt.savefig(os.path.join(SAVE_DIR, "shap_summary_bar.png"), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_explain, feature_names=feature_names, plot_type="dot", show=False)
plt.savefig(os.path.join(SAVE_DIR, "shap_summary_dot.png"), dpi=300, bbox_inches='tight')
plt.close()

print("SHAP plots saved in", SAVE_DIR)

# =======================
# 15. LEARNING CURVE (Windows safe)
# =======================
train_sizes, train_scores, val_scores = learning_curve(
    clf, X_train, y_train,
    cv=3,
    scoring="accuracy",
    n_jobs=1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
plt.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation")
plt.xlabel("Training Size"); plt.ylabel("Accuracy")
plt.title("Learning Curve - Trimodal Model")
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "learning_curve.png"), dpi=300, bbox_inches='tight')
plt.close()

# =======================
# 16. PCA / t-SNE / UMAP VISUALS
# =======================
pca_vis = PCA(n_components=2, random_state=42)
X_pca = pca_vis.fit_transform(X_train)

plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_train, cmap="coolwarm", alpha=0.6)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA Projection of Trimodal Features")
plt.savefig(os.path.join(SAVE_DIR, "pca_scatter.png"), dpi=300, bbox_inches='tight')
plt.close()

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_train)
plt.figure(figsize=(7,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_train, cmap="coolwarm", alpha=0.6)
plt.xlabel("t-SNE1"); plt.ylabel("t-SNE2")
plt.title("t-SNE Projection of Trimodal Features")
plt.savefig(os.path.join(SAVE_DIR, "tsne_scatter.png"), dpi=300, bbox_inches='tight')
plt.close()

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_train)
plt.figure(figsize=(7,6))
plt.scatter(X_umap[:,0], X_umap[:,1], c=y_train, cmap="coolwarm", alpha=0.6)
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.title("UMAP Projection of Trimodal Features")
plt.savefig(os.path.join(SAVE_DIR, "umap_scatter.png"), dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations saved in", SAVE_DIR)
