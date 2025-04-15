import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import io
import os
import zipfile
from unet_gan import HybridUNetGenerator
from utils import lab_to_rgb_custom

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =  HybridUNetGenerator().to(device)
model.load_state_dict(torch.load("saved_models_and_samples/best_generator.pth", map_location=device))
model.eval()

# Transform
transform = T.Compose([
    T.Resize((256, 256)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor()
])

def process_and_colorize(image):
    gray = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        ab = model(gray)
    lab = torch.cat([gray, ab], dim=1).squeeze().cpu()
    color_img = lab_to_rgb_custom(lab)
    return color_img

def auto_resize(image):
    max_dim = max(image.size)
    if max_dim > 1024:
        return image.resize((256, 256))
    return image

# UI Setup
st.set_page_config(page_title="Image Auto-Colorization", layout="wide")
st.title("🎨 AI-Powered Image Auto-Colorization")
st.markdown("Upload grayscale image(s) and colorize them using our GAN-based model.")

uploaded_files = st.file_uploader("📤 Upload grayscale image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    colorized_images = []
    st.info("Processing images. Please wait...")
    progress = st.progress(0)
    
    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert("RGB")
        image = auto_resize(image)
        colorized = process_and_colorize(image)
        colorized_images.append((uploaded_file.name, image, colorized))
        progress.progress((i + 1) / len(uploaded_files))

    st.success("✅ All images processed successfully!")

    # Gallery Preview
    st.markdown("### 🖼️ Gallery Preview")
    for idx, (name, gray_img, color_img) in enumerate(colorized_images):
        st.markdown(f"**Image {idx + 1}: {name}**")
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray_img, caption="Grayscale", use_container_width=True)
        with col2:
            st.image(color_img, caption="Colorized", use_container_width=True)

        # Download individual
        buf = io.BytesIO()
        color_img.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download This Colorized Image",
            data=buf.getvalue(),
            file_name=f"colorized_{name}",
            mime="image/png"
        )

    # Download all
    if len(colorized_images) > 1:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zipf:
            for name, _, img in colorized_images:
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                zipf.writestr(f"colorized_{name}", img_buf.getvalue())
        st.download_button(
            label="⬇️ Download All as ZIP",
            data=zip_buf.getvalue(),
            file_name="colorized_images.zip",
            mime="application/zip"
        )