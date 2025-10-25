import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ----------------- Configuration -----------------
MODEL_PATH = "./output_resnet50/best_resnet50.pth"  # path to your .pth
DATA_DIR = "./Dataset"  # folder with 'healthy/' and 'parkinson/' subfolders
OUTPUT_DIR = "./gradcam_output"
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

num_classes = 2  # change if you have more classes
classes = ['healthy', 'parkinson']

# ----------------- Model -----------------
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ----------------- Grad-CAM -----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

# ----------------- Image Preprocessing -----------------
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------- Utility to overlay heatmap -----------------
def overlay_heatmap(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    if img_np.ndim == 2:  # grayscale
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    overlayed = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed

# ----------------- Generate Grad-CAM -----------------
target_layer = model.layer4[-1].conv3  # last conv layer
gradcam = GradCAM(model, target_layer)

for class_name in classes:
    class_folder = os.path.join(DATA_DIR, class_name)
    save_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(save_class_dir, exist_ok=True)

    for fname in os.listdir(class_folder):
        img_path = os.path.join(class_folder, fname)
        img = Image.open(img_path).convert("L")
        input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        cam, pred_class = gradcam(input_tensor)
        overlayed = overlay_heatmap(img, cam)

        save_path = os.path.join(save_class_dir, fname)
        cv2.imwrite(save_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))
        print(f"Saved Grad-CAM for {fname} -> predicted: {classes[pred_class]}")

print("All Grad-CAMs saved in:", OUTPUT_DIR)
