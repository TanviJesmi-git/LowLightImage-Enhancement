import os
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import enhance_net_nopool as ZeroDCE

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Loss Functions
def loss_function(orig, enhanced, curves):
    """Zero-DCE Loss Function (modified with identity + improved exposure control)"""

    # 1. Spatial Consistency Loss
    def spatial_consistency_loss(enhanced, orig):
        device = orig.device
        base_kernels = [
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],   # left
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]],   # right
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],   # up
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],   # down
        ]

        def make_kernel(mat):
            k = torch.tensor(mat, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            k = k.repeat(3, 1, 1, 1)
            return k

        kernels = [make_kernel(k) for k in base_kernels]

        def gradient(x, kernel):
            return F.conv2d(x, kernel, padding=1, groups=3)

        loss = 0
        for k in kernels:
            loss += torch.mean((gradient(enhanced, k) - gradient(orig, k)) ** 2)
        return loss

    # 2. Brightness-Aware Exposure Control Loss (improved)
    def exposure_control_loss(orig, enhanced, patch_size=16, mean_val=0.6, low_thresh=0.3, high_thresh=0.7):
        """
        Boosts faintly lit → dark inputs, avoids over-brightening already well-lit ones.
        """
        pool = nn.AvgPool2d(patch_size)
        mean_enh = pool(enhanced)
        mean_orig = pool(orig)

        # Weight: stronger push if darker, weaker if near well-lit
        weight = torch.clamp((high_thresh - mean_orig) / (high_thresh - low_thresh), 0, 1)

        return torch.mean(weight * (mean_enh - mean_val) ** 2)

    # 3. Color Constancy Loss
    def color_constancy_loss(enhanced):
        r, g, b = torch.split(enhanced, 1, dim=1)
        mean_r = torch.mean(r)
        mean_g = torch.mean(g)
        mean_b = torch.mean(b)
        return (torch.pow(mean_r - mean_g, 2) +
                torch.pow(mean_r - mean_b, 2) +
                torch.pow(mean_g - mean_b, 2))

    # 4. Illumination Smoothness Loss
    def illumination_smoothness_loss(curves):
        return (torch.mean(torch.abs(curves[:, :, :, :-1] - curves[:, :, :, 1:])) +
                torch.mean(torch.abs(curves[:, :, :-1, :] - curves[:, :, 1:, :])))

    # 5. Identity Loss
    def identity_loss(orig, enhanced, thresh=0.5):
        mask = (orig.mean(dim=[1,2,3]) > thresh).float().view(-1,1,1,1)
        return torch.mean(mask * (enhanced - orig)**2)

    
    # Combine losses
    loss_spa = spatial_consistency_loss(enhanced, orig)
    loss_exp = exposure_control_loss(orig, enhanced)
    loss_col = color_constancy_loss(enhanced)
    loss_tv  = illumination_smoothness_loss(curves)
    loss_idt = identity_loss(orig, enhanced)

    total_loss = loss_spa + 10 * loss_exp + 5 * loss_col + 200 * loss_tv + 5 * loss_idt
    return total_loss


# Dataset / Training Config
TRAIN_DIR = r"C:\Users\Tanvi Jesmi\Downloads\Zero-DCE\lowlightDataset\train"
VAL_DIR   = r"C:\Users\Tanvi Jesmi\Downloads\Zero-DCE\lowlightDataset\val"
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs("./Zero-DCE/Zero-DCE_files", exist_ok=True)


# Dataset Class
class LowLightDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = glob.glob(os.path.join(img_dir, "*"))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = LowLightDataset(TRAIN_DIR, transform=transform)
val_dataset   = LowLightDataset(VAL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Model & Optimizer
model = ZeroDCE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)


# Training & Validation Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images in train_loader:
        images = images.to(DEVICE)
        optimizer.zero_grad()

        enhance_image_1, enhanced_images, curves = model(images)
        loss = loss_function(images, enhanced_images, curves)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    brightness_vals = []
    with torch.no_grad():
        for images in val_loader:
            images = images.to(DEVICE)
            _, enhanced_images, curves = model(images)
            loss = loss_function(images, enhanced_images, curves)
            val_loss += loss.item()

            # Track average brightness
            mean_brightness = enhanced_images.mean().item()
            brightness_vals.append(mean_brightness)

    avg_val_loss = val_loss / len(val_loader)
    avg_brightness = np.mean(brightness_vals)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Avg Brightness: {avg_brightness:.3f}")

    # Save checkpoint
    torch.save(model.state_dict(),
               f'./Zero-DCE/Zero-DCE_files/zero_dce_epoch{epoch+1}.pth')
