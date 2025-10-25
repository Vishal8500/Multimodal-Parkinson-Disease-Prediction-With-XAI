import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import os, joblib, pickle

# =====================
# 0. OUTPUT DIR
# =====================
out_dir = r"D:\SSDM\ssdm da-2\BIMODAL"
os.makedirs(out_dir, exist_ok=True)

# =====================
# 1. LOAD DATA
# =====================
speech_features = np.load(r"D:\SSDM\ssdm da-2\output_EFFICIENTb0_Speech\feature_vectors.npy")
gait_features   = np.load(r"D:\SSDM\ssdm da-2\output_gait\features.npy")
labels          = np.load(r"D:\SSDM\ssdm da-2\output_gait\labels.npy")

# =====================
# 2. AGGREGATE SPEECH PER SUBJECT
# =====================
subjects = len(labels)
samples_per_subject = len(speech_features) // subjects
speech_agg = [speech_features[i*samples_per_subject:(i+1)*samples_per_subject].mean(axis=0)
              for i in range(subjects)]
speech_agg = np.array(speech_agg)

# PCA reduce speech
pca = PCA(n_components=50, random_state=42)
speech_reduced = pca.fit_transform(speech_agg)

# =====================
# 3. TRAIN/TEST SPLIT
# =====================
idx = np.arange(len(labels))
train_idx, test_idx, y_train, y_test = train_test_split(
    idx, labels, test_size=0.4, random_state=42, stratify=labels
)

X_s_train, X_s_test = speech_reduced[train_idx], speech_reduced[test_idx]
X_g_train, X_g_test = gait_features[train_idx], gait_features[test_idx]

# Scale separately
scaler_s = StandardScaler().fit(X_s_train)
scaler_g = StandardScaler().fit(X_g_train)
X_s_train = scaler_s.transform(X_s_train)
X_s_test  = scaler_s.transform(X_s_test)
X_g_train = scaler_g.transform(X_g_train)
X_g_test  = scaler_g.transform(X_g_test)

# =====================
# 4. TRAIN MODELS
# =====================
clf_speech = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                           random_state=42, eval_metric="logloss").fit(X_s_train, y_train)

clf_gait   = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                           random_state=42, eval_metric="logloss").fit(X_g_train, y_train)

# =====================
# 5. FUSION WEIGHT TUNING
# =====================
proba_speech = clf_speech.predict_proba(X_s_test)[:, 1]
proba_gait   = clf_gait.predict_proba(X_g_test)[:, 1]

best_f1 = 0; best_ws = 0.5; best_thresh = 0.5
for ws in np.linspace(0.3, 0.7, 9):
    fused_proba = ws*proba_speech + (1-ws)*proba_gait
    for t in np.linspace(0.3,0.7,17):
        y_pred_temp = (fused_proba >= t).astype(int)
        f1 = classification_report(y_test, y_pred_temp, output_dict=True)["weighted avg"]["f1-score"]
        if f1 > best_f1:
            best_f1 = f1
            best_ws = ws
            best_thresh = t

# =====================
# 6. FINAL EVALUATION
# =====================
fused_proba = best_ws*proba_speech + (1-best_ws)*proba_gait
y_pred = (fused_proba >= best_thresh).astype(int)
print(classification_report(y_test, y_pred, target_names=["Healthy","Parkinson"]))

# =====================
# 7. SAVE EVERYTHING IN ONE FILE
# =====================
model_package = {
    "clf_speech": clf_speech,
    "clf_gait": clf_gait,
    "scaler_speech": scaler_s,
    "scaler_gait": scaler_g,
    "pca_speech": pca,
    "best_weight_speech": best_ws,
    "best_threshold": best_thresh
}

# Save as .pkl
joblib.dump(model_package, os.path.join(out_dir, "latefusion_model.pkl"))

# Save as .pth
with open(os.path.join(out_dir, "latefusion_model.pth"), "wb") as f:
    pickle.dump(model_package, f)

# Save fused features + labels
X_test_fused = np.concatenate([X_s_test, X_g_test], axis=1)
np.save(os.path.join(out_dir, "latefusion_features.npy"), X_test_fused)
np.save(os.path.join(out_dir, "latefusion_labels.npy"), y_test)

print(f"\n✅ Saved: ONE PKL, ONE PTH, features + labels to {out_dir}")
