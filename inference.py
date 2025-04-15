import torch
import cv2
import os
import numpy as np
from torchvision import transforms
from unet_gan import HybridUNetGenerator
from pathlib import Path

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained generator
generator = HybridUNetGenerator().to(device)
generator.load_state_dict(torch.load("saved_models_and_samples/best_generator.pth", map_location=device))
generator.eval()

# Transform for grayscale input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def colorize_image(grayscale_path, output_path):
    gray = cv2.imread(str(grayscale_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"⚠️ Failed to load image: {grayscale_path}")
        return
    gray_tensor = transform(gray).unsqueeze(0).to(device)
    with torch.no_grad():
        ab = generator(gray_tensor)
    color_lab = torch.cat([gray_tensor, ab], dim=1).squeeze(0).cpu().numpy()
    color_lab = (color_lab + 1) / 2
    color_lab = color_lab.transpose(1, 2, 0)
    color_lab[:, :, 0] *= 100
    color_lab[:, :, 1:] = color_lab[:, :, 1:] * 255 - 128

    bgr = cv2.cvtColor(color_lab.astype(np.float32), cv2.COLOR_LAB2BGR)
    cv2.imwrite(str(output_path), (bgr * 255).astype(np.uint8))
    print(f"✅ Saved: {output_path}")

def process_input(input_path, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    path = Path(input_path)

    if path.is_file():
        # Single image
        filename = path.name
        output_path = Path(output_dir) / f"colorized_{filename}"
        colorize_image(path, output_path)

    elif path.is_dir():
        # Multiple images in a folder
        for img_path in path.glob("*.[jpJP][pnPN]*[gG]"):  # Matches .jpg/.jpeg/.png
            output_path = Path(output_dir) / f"colorized_{img_path.name}"
            colorize_image(img_path, output_path)
    else:
        print("❌ Invalid input path. Provide an image or folder of images.")

# === USAGE ===
# Just change the below line to either an image path or folder path:
input_path = "Dataset/COCO_train2014_000000000009.jpg"       # e.g. "samples/test_gray.jpg" or "new_images/"
process_input(input_path)