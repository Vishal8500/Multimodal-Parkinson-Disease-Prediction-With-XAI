import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
import torch

# =====================
# 1. LOAD DATA
# =====================
handwriting_features = np.load(r"D:\SSDM\ssdm da-2\output_resnet50\features.npy")   # (3264, 2)
gait_features        = np.load(r"D:\SSDM\ssdm da-2\output_gait\features.npy")       # (306, 18)
labels               = np.load(r"D:\SSDM\ssdm da-2\output_resnet50\labels.npy")     # (3264,)

print("Handwriting:", handwriting_features.shape)
print("Gait:", gait_features.shape)
print("Labels:", labels.shape)

# =====================
# 2. AGGREGATE HANDWRITING PER SUBJECT
# =====================
subjects = gait_features.shape[0]  # 306
samples_per_subject = len(handwriting_features) // subjects
print("Samples per subject (handwriting):", samples_per_subject)

handwriting_agg = []
labels_agg = []
for i in range(subjects):
    start = i * samples_per_subject
    end = start + samples_per_subject
    handwriting_agg.append(handwriting_features[start:end].mean(axis=0))
    labels_agg.append(labels[start])  # assume consistent labels per subject

handwriting_agg = np.array(handwriting_agg)  # (306, 2)
labels_agg = np.array(labels_agg)            # (306,)
print("Aggregated handwriting:", handwriting_agg.shape)
print("Aggregated labels:", labels_agg.shape)

# =====================
# 3. EARLY FUSION
# =====================
X_fused = np.concatenate([handwriting_agg, gait_features], axis=1)  # (306, 20)
y = labels_agg
print("Fused features:", X_fused.shape)

# =====================
# 4. TRAIN/TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_fused, y, test_size=0.7, random_state=455, stratify=y
)

# =====================
# 5. SCALE + ADD NOISE
# =====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inject Gaussian noise
X_train = X_train + np.random.normal(0, 0.5, X_train.shape)
X_test  = X_test  + np.random.normal(0, 0.5, X_test.shape)

# =====================
# 6. TRAIN SIMPLER XGBOOST
# =====================
clf = XGBClassifier(
    n_estimators=50,
    learning_rate=0.2,
    max_depth=2,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    eval_metric="logloss"
)
clf.fit(X_train, y_train)

# =====================
# 7. PREDICTION WITH STRICTER THRESHOLD
# =====================
y_proba = clf.predict_proba(X_test)[:, 1]
best_threshold = 0.7 
y_pred = (y_proba >= best_threshold).astype(int)

# =====================
# 8. FINAL EVALUATION
# =====================
print(f"\nClassification Report (Bimodal Handwriting+Gait, Threshold={best_threshold}):")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson"]))

# =====================
# 9. SAVE FEATURES & LABELS (ONE FILE EACH)
# =====================
save_dir = r"D:\SSDM\ssdm da-2\BIMODAL"  # ← save location
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "features.npy"), X_fused)
np.save(os.path.join(save_dir, "labels.npy"), y)

print(f"Saved combined features to {os.path.join(save_dir,'features.npy')}")
print(f"Saved labels to {os.path.join(save_dir,'labels.npy')}")

# =====================
# 10. SAVE MODEL FILES
# =====================
# Save as .pkl using joblib (native format for sklearn/xgboost)
pkl_path = os.path.join(save_dir, "xgb_bimodal_model.pkl")
joblib.dump(clf, pkl_path)
print(f"Saved XGB model to {pkl_path}")

# Save as .pth (PyTorch-style raw booster bytes)
pth_path = os.path.join(save_dir, "xgb_bimodal_model.pth")
torch.save(clf.get_booster().save_raw(), pth_path)
print(f"Saved raw booster to {pth_path}")
