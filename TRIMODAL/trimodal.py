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
# 1. LOAD DATA
# =======================
speech_features = np.load(r"D:\SSDM\ssdm da-2\output_EFFICIENTb0_Speech\feature_vectors.npy")   # (1740, 1280)
gait_features   = np.load(r"D:\SSDM\ssdm da-2\output_gait\features.npy")                        # (306, 18)
handwriting_features = np.load(r"D:\SSDM\ssdm da-2\output_resnet50\features.npy")               # (3264, d_hw)
labels = np.load(r"D:\SSDM\ssdm da-2\output_resnet50\labels.npy")                               # (3264,)

print("Speech:", speech_features.shape)
print("Gait:", gait_features.shape)
print("Handwriting:", handwriting_features.shape)
print("Labels:", labels.shape)

# =======================
# 2. ALIGN SUBJECTS (gait = 306 as reference)
# =======================
subjects = len(gait_features)   # 306 subjects

# Aggregate speech (≈5 samples per subject)
samples_per_subject = len(speech_features) // subjects
speech_agg = [speech_features[i*samples_per_subject:(i+1)*samples_per_subject].mean(axis=0)
              for i in range(subjects)]
speech_agg = np.array(speech_agg)

# PCA reduce speech to 50D
pca = PCA(n_components=50, random_state=42)
speech_reduced = pca.fit_transform(speech_agg)

# Stratified sample handwriting to match 306 subjects
hw_idx = np.arange(len(handwriting_features))
hw_idx, _, y_hw, _ = train_test_split(
    hw_idx, labels, train_size=subjects, stratify=labels, random_state=42
)
handwriting_features = handwriting_features[hw_idx]
labels = y_hw

print("Aligned Speech:", speech_reduced.shape)
print("Aligned Gait:", gait_features.shape)
print("Aligned Handwriting:", handwriting_features.shape)
print("Aligned Labels:", labels.shape)

# =======================
# 3. EARLY FUSION
# =======================
X_fused = np.concatenate([speech_reduced, gait_features, handwriting_features], axis=1)
y = labels
print("Fused features:", X_fused.shape)

# =======================
# 4. TRAIN/TEST SPLIT
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X_fused, y, test_size=0.4, random_state=455, stratify=y
)

# =======================
# 5. SCALE FEATURES
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
# 7. CALCULATE scale_pos_weight FOR XGBOOST
# =======================
neg, pos = np.bincount(y_train)
scale = neg / pos
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
# 9. PREDICTION WITH FIXED BEST THRESHOLD (0.5)
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
# 11. CREATE FEATURE NAMES FOR SHAP
# =======================
speech_names = [f"speech_{i+1}" for i in range(speech_reduced.shape[1])]
gait_names = [f"gait_{i+1}" for i in range(gait_features.shape[1])]
hw_names = [f"hw_{i+1}" for i in range(handwriting_features.shape[1])]
feature_names = speech_names + gait_names + hw_names
print("Total features:", len(feature_names))

# =======================
# 12. SAVE FEATURES & LABELS
# =======================
save_dir = r"D:\SSDM\ssdm da-2\TRIMODAL"
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, "features.npy"), X_fused)
np.save(os.path.join(save_dir, "labels.npy"), y)
print(f"Saved features and labels in {save_dir}")

# =======================
# 13. SAVE MODEL FILES
# =======================
pkl_path = os.path.join(save_dir, "xgb_trimodal_model.pkl")
joblib.dump(clf, pkl_path)
pth_path = os.path.join(save_dir, "xgb_trimodal_model.pth")
torch.save(clf.get_booster().save_raw(), pth_path)
print(f"Saved XGB model (.pkl and .pth) in {save_dir}")

# =======================
# 14. SHAP EXPLAINER
# =======================
X_explain = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_explain)

# Summary bar plot
plt.figure(figsize=(12,6))
shap.summary_plot(shap_values, X_explain, feature_names=feature_names, plot_type="bar", show=False)
plt.savefig(os.path.join(save_dir, "shap_summary_bar.png"), dpi=300, bbox_inches='tight')
plt.close()

# Summary dot plot
plt.figure(figsize=(12,6))
shap.summary_plot(shap_values, X_explain, feature_names=feature_names, plot_type="dot", show=False)
plt.savefig(os.path.join(save_dir, "shap_summary_dot.png"), dpi=300, bbox_inches='tight')
plt.close()

print("SHAP plots saved in", save_dir)

# =======================
# 15. LEARNING CURVE (Windows safe)
# =======================
train_sizes, train_scores, val_scores = learning_curve(
    clf, X_train, y_train,
    cv=3,
    scoring="accuracy",
    n_jobs=1,  # <- important fix
    train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
plt.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Trimodal Model")
plt.legend()
plt.savefig(os.path.join(save_dir, "learning_curve.png"), dpi=300, bbox_inches='tight')
plt.close()

# =======================
# 16. PCA VISUALIZATION
# =======================
pca_vis = PCA(n_components=2)
X_pca = pca_vis.fit_transform(X_train)

plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_train, cmap="coolwarm", alpha=0.6)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA Projection of Trimodal Features")
plt.savefig(os.path.join(save_dir, "pca_scatter.png"), dpi=300, bbox_inches='tight')
plt.close()

# =======================
# 17. T-SNE VISUALIZATION
# =======================
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_train)

plt.figure(figsize=(7,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_train, cmap="coolwarm", alpha=0.6)
plt.xlabel("t-SNE1"); plt.ylabel("t-SNE2")
plt.title("t-SNE Projection of Trimodal Features")
plt.savefig(os.path.join(save_dir, "tsne_scatter.png"), dpi=300, bbox_inches='tight')
plt.close()

# =======================
# 18. UMAP VISUALIZATION
# =======================
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_train)

plt.figure(figsize=(7,6))
plt.scatter(X_umap[:,0], X_umap[:,1], c=y_train, cmap="coolwarm", alpha=0.6)
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.title("UMAP Projection of Trimodal Features")
plt.savefig(os.path.join(save_dir, "umap_scatter.png"), dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations saved in", save_dir)
