if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataloader.colorization_dataset import ColorizationDataset
from unet_gan import HybridUNetGenerator, PatchDiscriminator, train_colorization_gan
import torch.nn as nn
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset directory
image_dir = "Dataset"

# Load full dataset (grayscale previews optional)
print("Loading dataset...")
full_dataset = ColorizationDataset(image_dir=image_dir, image_size=256, verbose=True, save_grayscale=True)
print(f"Dataset loaded with {len(full_dataset)} images.")

# Split into train and validation
indices = list(range(len(full_dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

# DataLoaders
train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=0)

# Model initialization
print("Initializing model...")
generator = HybridUNetGenerator(input_channels=1, output_channels=2).to(device)
discriminator = PatchDiscriminator(input_channels=3).to(device)

# Loss functions
l1_loss_fn = nn.L1Loss()
bce_loss_fn = nn.BCEWithLogitsLoss()

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Save directory
save_dir = "./saved_models_and_samples"
os.makedirs(save_dir, exist_ok=True)

# Training loop
print("Starting training...")
train_colorization_gan(
    generator,
    discriminator,
    train_loader,
    val_loader=val_loader,
    epochs=30,
    device=device,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
    l1_loss_fn=l1_loss_fn,
    bce_loss_fn=bce_loss_fn,
    save_dir=save_dir,
    save_samples_every=5,
    enable_visualization=True
)