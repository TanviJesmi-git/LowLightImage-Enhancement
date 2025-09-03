import io
import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from torchvision import transforms

# Import your model
from model import enhance_net_nopool as ZeroDCE


# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your best trained model
MODEL_PATH = r"C:\Users\Tanvi Jesmi\Downloads\Zero-DCE\Zero-DCE\Zero-DCE_code\Zero-DCE\Zero-DCE_files\zero_dce_epoch45.pth"

model = ZeroDCE().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# FastAPI App
app = FastAPI(title="Image Enhancement API", description="Module 1: Zero-DCE Enhancer")

@app.get("/")
def root():
    return {"message": "Image Enhancement API is running. Visit /docs for Swagger UI."}


@app.post("/enhance_frame/")
async def enhance_frame(file: UploadFile = File(...)):
    """
    Receive an image frame, enhance it, and return the enhanced image.
    """
    # Load input frame
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    inp = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, enhanced_img, _ = model(inp)

    out = enhanced_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
    out = (out * 255).clip(0, 255).astype("uint8")
    out_img = Image.fromarray(out)

    # Convert to bytes
    buf = io.BytesIO()
    out_img.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
