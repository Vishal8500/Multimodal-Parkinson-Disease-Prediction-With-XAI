import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# ========================
# Config
# ========================
models_info = {
    "image_model": {
        "file": r"D:\SSDM\ssdm da-2\output_resnet50\best_resnet50.pth",
        "framework": "torch",
        "features": r"D:\SSDM\ssdm da-2\output_resnet50\features.npy",
        "labels": r"D:\SSDM\ssdm da-2\output_resnet50\labels.npy"
    },
    "gait_model": {
        "file": r"D:\SSDM\ssdm da-2\parkinson\rf_model_pickle.pkl",
        "framework": "sklearn",
        "features": r"D:\SSDM\ssdm da-2\output_gait\features.npy",
        "labels": r"D:\SSDM\ssdm da-2\output_gait\labels.npy"
    },
    "speech_model": {
        "file": r"D:\SSDM\ssdm da-2\output_EFFICIENTb0_Speech\best_efficientnet_audio.pth",
        "framework": "torch",
        "features": r"D:\SSDM\ssdm da-2\output_EFFICIENTb0_Speech\feature_vectors.npy",
        "labels": r"D:\SSDM\ssdm da-2\output_EFFICIENTb0_Speech\labels.npy"
    },
    "image_gait_model": {
        "file": r"D:\SSDM\ssdm da-2\BIMODAL\HWGAIT\xgb_bimodal_model.pth",
        "framework": "torch",
        "features": r"D:\SSDM\ssdm da-2\BIMODAL\HWGAIT\features.npy",
        "labels": r"D:\SSDM\ssdm da-2\BIMODAL\HWGAIT\labels.npy"
    },
    "image_speech_model": {
        "file": r"D:\SSDM\ssdm da-2\BIMODAL\HWSPEECH\bimodal_model.pth",
        "framework": "torch",
        "features": r"D:\SSDM\ssdm da-2\BIMODAL\HWSPEECH\bimodal_features.npy",
        "labels": r"D:\SSDM\ssdm da-2\BIMODAL\HWSPEECH\bimodal_labels.npy"
    },
    "gait_speech_model": {
        "file": r"D:\SSDM\ssdm da-2\BIMODAL\GAITSPEECH\latefusion_model.pth",
        "framework": "torch",
        "features": r"D:\SSDM\ssdm da-2\BIMODAL\GAITSPEECH\latefusion_features.npy",
        "labels": r"D:\SSDM\ssdm da-2\BIMODAL\GAITSPEECH\latefusion_labels.npy"
    },
    "trimodal_model": {
        "file": r"D:\SSDM\ssdm da-2\TRIMODAL\xgb_trimodal_model.pth",
        "framework": "torch",
        "features": r"D:\SSDM\ssdm da-2\TRIMODAL\features.npy",
        "labels": r"D:\SSDM\ssdm da-2\TRIMODAL\labels.npy"
    }
}

output_dir = "tsne_umap_all_models"
os.makedirs(output_dir, exist_ok=True)

# Class names for coloring
class_names = ['HC', 'PD']
colors = ['tab:blue', 'tab:orange']

# ========================
# Function to plot 2D embeddings
# ========================
def plot_2d_embedding(embedding, labels, title, save_path):
    plt.figure(figsize=(8, 6))
    for cls_idx, cls_name in enumerate(class_names):
        idxs = np.where(labels == cls_idx)
        plt.scatter(embedding[idxs, 0], embedding[idxs, 1], label=cls_name, alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

# ========================
# Generate t-SNE and UMAP plots for all models
# ========================
for model_name, info in models_info.items():
    features = np.load(info["features"])
    labels = np.load(info["labels"])

    # t-SNE
    tsne_emb = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000).fit_transform(features)
    plot_2d_embedding(tsne_emb, labels, f"t-SNE 2D: {model_name}", os.path.join(output_dir, f"{model_name}_tsne.png"))

    # UMAP
    umap_emb = umap.UMAP(n_components=2, random_state=42).fit_transform(features)
    plot_2d_embedding(umap_emb, labels, f"UMAP 2D: {model_name}", os.path.join(output_dir, f"{model_name}_umap.png"))
