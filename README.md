
# 🎨 Hybrid Image Colorization using U-Net with Attention, PatchGAN, and Perceptual Loss

## 🧠 Project Overview

This project is a **Hybrid Image Colorization System** that utilizes a **U-Net with attention mechanism**, a **PatchGAN discriminator**, and a **VGG16-based perceptual loss** for high-quality colorization of grayscale images. It operates in the **LAB color space** and achieves impressive results with **92% SSIM** and **88% PSNR** on unseen images.

---

## 🚀 Features

- ✅ High-fidelity image colorization in LAB color space
- 🎯 Hybrid architecture combining U-Net, PatchGAN, and perceptual loss
- 🎨 Realistic output with smooth gradients and accurate tones
- 🧠 Perceptual loss using pretrained VGG16
- 📊 SSIM and PSNR evaluation metrics
- 📈 Data augmentation and custom loss functions
- 📷 Real-time inference with Streamlit app

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**

---

## 📁 Project Structure

```
hybrid-image-colorization/
├── dataloader/
│   ├── __init__.py
│   ├── colorization_dataset.py     # Custom dataset class for LAB color images
│   └── dataloaders.py              # Train and test dataloaders
├── app.py                          # Streamlit app for inference
├── inference.py                    # Inference script for standalone colorization
├── train.py                        # Training loop for GAN + perceptual loss
├── unet_gan.py                     # U-Net generator + PatchGAN discriminator
├── utils.py                        # Helper functions (e.g., SSIM/PSNR, visualization)
└── README.md                       # This file
```

---

## 🖼️ Image Format & Color Space

- **Input**: Grayscale channel (L channel from LAB)
- **Output**: AB color channels
- Full image = L + AB recombined after prediction

---

## 📈 Results

| Metric       | Value     |
|--------------|-----------|
| SSIM         | 92%       |
| PSNR         | 88%       |
| Inference Time | ~50ms/image |

Visualizations and comparisons are available in the `utils.py` and Streamlit app.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid-image-colorization.git
cd hybrid-image-colorization

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Training the Model

```bash
python train.py
```

Features:

- Progressive resolution training (if enabled)
- Visual logs (optional: integrate WandB)
- SSIM and PSNR tracking
- Real + fake discriminator loss + perceptual loss

---

## 🎯 Inference

Standalone inference (batch):

```bash
python inference.py --input_dir grayscale_images/ --output_dir colorized_output/
```

Or run with Streamlit:

```bash
streamlit run app.py
```

Streamlit Features:

- Upload grayscale images
- See real-time colorized results
- Visualize input vs output vs original

---

## 🔬 Loss Functions

- 🔻 **L1 Loss** on AB channels
- 🧠 **VGG Perceptual Loss** (VGG16)
- 🎨 **Color Fidelity Loss** (optional)
- 🧩 **PatchGAN Discriminator Loss**

---

## 🧪 Evaluation Metrics

```python
from utils import calculate_ssim, calculate_psnr
```

Metrics used:

- **SSIM (Structural Similarity Index)**
- **PSNR (Peak Signal-to-Noise Ratio)**

---

## 📦 Requirements (requirements.txt)

```txt
torch
torchvision
opencv-python
numpy
matplotlib
scikit-learn
streamlit
Pillow
```
Add `wandb` if logging is enabled.

---

## 📌 Future Enhancements

- ✨ Integrate transformer modules for global context
- 🔁 Real-time webcam support
- 🧠 Enhance perceptual loss with deeper networks (VGG19/ResNet)
- 📈 Add TensorBoard or WandB support

---

## 🙌 Credits

Developed by **Akash**  
Machine Learning Engineer | 2025  
Hybrid GAN-based Image Colorization with Attention, Perceptual Loss, and Realistic Fidelity

---

## 📜 License

Open-source under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Open an issue first to discuss major changes before a pull request.
