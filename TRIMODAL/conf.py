import os
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

# ======================
# 1. LOAD MODEL + DATA
# ======================
# Paths
save_dir = r"D:\SSDM\ssdm da-2\TRIMODAL"
model_path = os.path.join(save_dir, "xgb_trimodal_model.pkl")
features_path = os.path.join(save_dir, "features.npy")
labels_path = os.path.join(save_dir, "labels.npy")

# Load model
clf = joblib.load(model_path)

# Load features and labels
X = np.load(features_path)
y = np.load(labels_path)

# Split train/test again (same random_state used earlier for consistency)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=455, stratify=y
)

# ======================
# 2. PREDICT
# ======================
y_pred = clf.predict(X_test)

# ======================
# 3. CLASSIFICATION REPORT
# ======================
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson"]))

# ======================
# 4. CONFUSION MATRIX
# ======================
cm = confusion_matrix(y_test, y_pred)
labels = ["Healthy", "Parkinson"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ======================
# 5. CONFUSION MATRIX WITH NORMALIZED VALUES (%)
# ======================
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix (%)")
plt.show()
