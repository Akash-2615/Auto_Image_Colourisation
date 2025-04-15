# utils.py
import os
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import cv2

import cv2
import numpy as np
from PIL import Image

import cv2
import numpy as np
from PIL import Image

def lab_to_rgb_custom(lab_tensor):
    """
    Converts a LAB image tensor (3xHxW or 1x3xHxW) to a proper RGB PIL image.
    Assumes L is in [0, 1] → [0, 100]
    and a,b in [-1, 1] → [-128, 127]
    """
    if lab_tensor.dim() == 4:
        lab_tensor = lab_tensor.squeeze(0)

    lab_np = lab_tensor.detach().cpu().numpy()
    lab_np = np.transpose(lab_np, (1, 2, 0))  # CxHxW → HxWxC

    # Denormalize
    L = lab_np[:, :, 0] * 100                  # [0,1] → [0,100]
    ab = lab_np[:, :, 1:] * 110                # [-1,1] → roughly [-110,110] (more stable than 255 - 128)
    
    lab_img = np.concatenate([L[..., np.newaxis], ab], axis=2).astype(np.float32)

    # Convert LAB to RGB
    rgb = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    return Image.fromarray(rgb)

    # Denormalize LAB values
    lab_np[:, :, 0] = lab_np[:, :, 0] * 100  # L: [0,1] → [0,100]
    lab_np[:, :, 1:] = lab_np[:, :, 1:] * 255 - 128  # AB: [-1,1] → [-128,127]

    lab_np = lab_np.astype(np.float32)
    rgb = cv2.cvtColor(lab_np, cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    return Image.fromarray(rgb)
def visualize_sample_grid(L, fake_AB, real_AB, epoch, save_dir):
    """
    Save a grid of L channel, generated AB, and ground truth AB.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fake_color = torch.cat([L, fake_AB], dim=1)
    real_color = torch.cat([L, real_AB], dim=1)

    grid = torch.cat([fake_color, real_color], dim=0)
    grid = vutils.make_grid(grid, nrow=len(L), normalize=True)

    plt.figure(figsize=(12, 6))
    plt.title(f"Epoch {epoch} - Top: Fake, Bottom: Real")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_samples.png"))
    plt.close()


def plot_losses_live(g_losses, d_losses):
    """
    Plot generator and discriminator losses.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_losses.png")
    plt.close()
    
def visualize_sample_grid(batch, unnormalize=True):
    """
    Convert a batch of LAB images to RGB and return a grid.
    """
    import torchvision.utils as vutils
    batch = batch[:8]  # take first 8 samples
    if unnormalize:
        batch = (batch + 1) / 2
    return np.transpose(vutils.make_grid(batch, nrow=4).cpu().numpy(), (1, 2, 0))

def plot_losses_live(g_losses, d_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()