# colorization_dataset.py

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class ColorizationDataset(Dataset):
    def __init__(self, image_dir, image_size=256, verbose=True, save_grayscale=False, grayscale_dir="grayscale_preview"):
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)
                            if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.image_size = image_size
        self.verbose = verbose
        self.save_grayscale = save_grayscale
        self.grayscale_dir = grayscale_dir

        if save_grayscale and not os.path.exists(grayscale_dir):
            os.makedirs(grayscale_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.preprocessed_data = []
        self._preprocess()

    def _preprocess(self):
        iterator = tqdm(self.image_paths, desc="Loading & converting images", disable=not self.verbose)

        for path in iterator:
            try:
                image = Image.open(path).convert("RGB")
                image = self.transform(image)
                image_np = np.transpose(image.numpy(), (1, 2, 0))  # CxHxW → HxWxC

                # Convert to LAB
                image_lab = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
                image_lab /= 255.0  # Normalize to [0, 1]

                # Split channels
                L = image_lab[:, :, 0:1]          # Luminance
                AB = image_lab[:, :, 1:] - 0.5    # Chrominance (shift to [-0.5, 0.5])

                # Convert to Tensors
                L_tensor = transforms.ToTensor()(L)
                AB_tensor = transforms.ToTensor()(AB)

                if self.save_grayscale:
                    preview_path = os.path.join(self.grayscale_dir, os.path.basename(path))
                    cv2.imwrite(preview_path, (L[:, :, 0] * 255).astype(np.uint8))

                self.preprocessed_data.append((L_tensor, AB_tensor))

            except Exception as e:
                print(f"Skipping {path}: {e}")

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]