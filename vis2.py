import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Load your CSV files
# ===========================
cnn_logreg_df = pd.read_csv(r"D:\SSDM\ssdm da-2\logs\cnn_loso_logreg_per_fold.csv")
rf_sgkf_df = pd.read_csv(r"D:\SSDM\ssdm da-2\logs\rf_sgkf_per_fold.csv")
features_df = pd.read_csv(r"D:\SSDM\ssdm da-2\logs\features.csv")

# Make output folder
os.makedirs("figures", exist_ok=True)

# ===========================
# Prepare data
# ===========================
cnn_logreg_df["model"] = "CNN+LogReg"
rf_sgkf_df["model"] = "RandomForest"

cnn_perf = cnn_logreg_df[["acc", "f1", "auc", "model"]]
rf_perf = rf_sgkf_df[["acc", "f1", "auc", "model"]]
perf_df = pd.concat([cnn_perf, rf_perf], ignore_index=True)

# ===========================
# 1. F1 Violin Plot
# ===========================
plt.figure(figsize=(6, 5))
sns.violinplot(x="model", y="f1", data=perf_df, inner="box", palette="Set2")
plt.title("Distribution of F1 across folds")
plt.ylabel("F1")
plt.xlabel("")
plt.savefig("figures/f1_violin.png", dpi=300, bbox_inches="tight")
plt.close()

# ===========================
# 2. F1 Bar Plot with Error Bars
# ===========================
plt.figure(figsize=(6, 5))
sns.barplot(x="model", y="f1", data=perf_df, ci="sd", palette="Set1")
plt.title("Mean ± Std of F1")
plt.ylabel("F1")
plt.xlabel("")
plt.savefig("figures/f1_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# ===========================
# 3. Accuracy Violin Plot
# ===========================
plt.figure(figsize=(6, 5))
sns.violinplot(x="model", y="acc", data=perf_df, inner="box", palette="Set2")
plt.title("Distribution of Accuracy across folds")
plt.ylabel("Accuracy")
plt.xlabel("")
plt.savefig("figures/acc_violin.png", dpi=300, bbox_inches="tight")
plt.close()

# ===========================
# 4. ROC-AUC Distribution
# ===========================
plt.figure(figsize=(8, 5))
sns.kdeplot(rf_sgkf_df["auc"].dropna(), label="Random Forest", fill=True, alpha=0.4)
sns.kdeplot(cnn_logreg_df["auc"].dropna(), label="CNN+LogReg", fill=True, alpha=0.4)
plt.title("Distribution of AUC across folds")
plt.xlabel("AUC")
plt.legend()
plt.savefig("figures/auc_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# ===========================
# 5. Feature Correlation Heatmap
# ===========================
plt.figure(figsize=(12, 8))
corr = features_df.drop(columns=["subject", "trial", "label"]).corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig("figures/feature_corr_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Selected figures generated and saved in 'figures/' folder.")
