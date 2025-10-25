import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

feature_paths = {
    "image": r"D:\SSDM\ssdm da-2\output_resnet50\features.npy",
    "gait": r"D:\SSDM\ssdm da-2\output_gait\features.npy",
    "speech": r"D:\SSDM\ssdm da-2\output_EFFICIENTb0_Speech\feature_vectors.npy"
}

# Load features
features = {k: np.load(v) for k, v in feature_paths.items()}

# Reduce each modality to **mean feature vector across samples**
mean_feature_vectors = {k: v.mean(axis=0) for k, v in features.items()}  # shape: (num_features,)

# Compute correlation matrix between modalities
modalities = list(mean_feature_vectors.keys())
num_modalities = len(modalities)
corr_matrix = np.zeros((num_modalities, num_modalities))

for i, mod1 in enumerate(modalities):
    for j, mod2 in enumerate(modalities):
        # Use 1D vectors: mean across samples
        vec1 = mean_feature_vectors[mod1]
        vec2 = mean_feature_vectors[mod2]
        # To handle different lengths, truncate to min length
        min_len = min(len(vec1), len(vec2))
        corr_matrix[i, j] = np.corrcoef(vec1[:min_len], vec2[:min_len])[0, 1]

# Plot heatmap
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, xticklabels=modalities, yticklabels=modalities,
            cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Between Modalities' Feature Vectors (mean over samples)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "modality_correlation_heatmap.png"))
plt.close()

print(f"Correlation heatmap saved to {output_dir}/modality_correlation_heatmap.png")
