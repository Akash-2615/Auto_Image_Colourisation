# dataloader.py

from torch.utils.data import DataLoader
from dataloader.colorization_dataset import ColorizationDataset


def get_dataloader(image_dir,
                   image_size=256,
                   batch_size=16,
                   shuffle=True,
                   num_workers=0,
                   verbose=True,
                   save_grayscale=False):
    """
    Returns a DataLoader for the colorization dataset.
    """
    dataset = ColorizationDataset(
        image_dir=image_dir,
        image_size=image_size,
        verbose=verbose,
        save_grayscale=save_grayscale
    )

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True)

    return dataloader