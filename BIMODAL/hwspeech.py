import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import os, joblib, pickle

# =====================
# 0. OUTPUT DIR
# =====================
out_dir = r"D:\SSDM\ssdm da-2\BIMODAL"
os.makedirs(out_dir, exist_ok=True)

# =====================
# 1. LOAD DATA
# =====================
audio_features = np.load(r"D:\SSDM\ssdm da-2\output_EFFICIENTb0_Speech\feature_vectors.npy")
image_features = np.load(r"D:\SSDM\ssdm da-2\output_resnet50\features.npy")
image_labels   = np.load(r"D:\SSDM\ssdm da-2\output_resnet50\labels.npy")

# Ensure alignment
min_len = min(len(audio_features), len(image_features), len(image_labels))
audio_features = audio_features[:min_len]
image_features = image_features[:min_len]
labels = image_labels[:min_len]   # ✅ Use only IMAGE labels

print("Audio:", audio_features.shape)
print("Image:", image_features.shape)
print("Labels:", labels.shape)

# =====================
# 2. EARLY FUSION (Concatenate features)
# =====================
X_fused = np.concatenate([audio_features, image_features], axis=1)

# =====================
# 3. TRAIN/TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_fused, labels, test_size=0.3, random_state=455, stratify=labels
)

# =====================
# 4. SCALE FEATURES
# =====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================
# 5. HANDLE CLASS IMBALANCE WITH SMOTE
# =====================
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

print("After SMOTE -> Train shape:", X_train.shape, y_train.shape)

# =====================
# 6. CALCULATE scale_pos_weight FOR XGBOOST
# =====================
neg, pos = np.bincount(y_train)
scale = neg / pos
print(f"Class balance after SMOTE -> Healthy: {neg}, Parkinson: {pos}, scale_pos_weight: {scale:.2f}")

# =====================
# 7. TRAIN XGBOOST CLASSIFIER
# =====================
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

# =====================
# 8. PREDICTION WITH BEST THRESHOLD (0.45)
# =====================
y_proba = clf.predict_proba(X_test)[:, 1]
best_threshold = 0.45
y_pred = (y_proba >= best_threshold).astype(int)

# =====================
# 9. FINAL EVALUATION
# =====================
print(f"\nClassification Report (Bimodal Early Fusion + SMOTE + XGBoost, Threshold={best_threshold}):")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson"]))

# =====================
# 10. SAVE EVERYTHING
# =====================
# Save ONE feature vector file + ONE label file
np.save(os.path.join(out_dir, "bimodal_features.npy"), X_test)
np.save(os.path.join(out_dir, "bimodal_labels.npy"), y_test)

# Save model + scaler + threshold in a dict
model_package = {
    "classifier": clf,
    "scaler": scaler,
    "best_threshold": best_threshold
}

# Save as .pkl
joblib.dump(model_package, os.path.join(out_dir, "bimodal_model.pkl"))

# Save as .pth
with open(os.path.join(out_dir, "bimodal_model.pth"), "wb") as f:
    pickle.dump(model_package, f)

print(f"\n✅ Saved: ONE PKL, ONE PTH, features + labels to {out_dir}")
