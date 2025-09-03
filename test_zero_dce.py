import os
import torch
from PIL import Image
from torchvision import transforms
from model import enhance_net_nopool as ZeroDCE

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\Tanvi Jesmi\Downloads\Zero-DCE\Zero-DCE\Zero-DCE_code\Zero-DCE\Zero-DCE_files\zero_dce_epoch50.pth"   # stored model weights
INPUT_DIR = r"C:\Users\Tanvi Jesmi\Downloads\Zero-DCE\lowlightDataset\test"  # folder with low-light images
OUTPUT_DIR = r"C:\Users\Tanvi Jesmi\Downloads\Zero-DCE\lowlightDataset\enhanced"

# Create output folder if missing
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = ZeroDCE().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Inference loop
with torch.no_grad():
    for img_name in os.listdir(INPUT_DIR):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(INPUT_DIR, img_name)
        img = Image.open(img_path).convert("RGB")

        inp = transform(img).unsqueeze(0).to(DEVICE)
        _, enhanced_img, _ = model(inp)

        out = enhanced_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
        out = (out * 255).clip(0, 255).astype("uint8")

        save_path = os.path.join(OUTPUT_DIR, f"enh_{img_name}")
        Image.fromarray(out).save(save_path)

        print(f"Enhanced image saved: {save_path}")
