"""
PyTorch Dataset and DataLoader pipelines for Plant Disease Classification.
Provides robust augmentations for training and deterministic transforms for validation/testing.
"""
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src.config import config

def get_transforms(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Construct torchvision transform pipeline.
    Training pipeline: RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip,
                       RandomRotation, ColorJitter, ToTensor, Normalize.
    Val/Test pipeline: Resize, CenterCrop, ToTensor, Normalize.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # ~256 for 224 input
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std)
        ])

class PlantVillageDataset(Dataset):
    """
    Custom Dataset for PlantVillage crop leaf disease images.
    Reads image filepaths and labels from a pre-generated metadata DataFrame or CSV.
    """
    def __init__(
        self,
        df_or_csv: str | Path | pd.DataFrame,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None
    ):
        if isinstance(df_or_csv, (str, Path)):
            self.df = pd.read_csv(df_or_csv)
        else:
            self.df = df_or_csv.copy()

        if max_samples is not None and max_samples < len(self.df):
            # Sample deterministically for smoke tests / debugging
            self.df = self.df.sample(n=max_samples, random_state=config.seed).reset_index(drop=True)

        self.transform = transform
        self.image_paths = self.df["filepath"].values
        self.labels = self.df["label"].values.astype(int)
        self.class_names = self.df["class_name"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image at {path}: {e}")

        if self.transform is not None:
            tensor_img = self.transform(img)
        else:
            tensor_img = transforms.ToTensor()(img)

        label = self.labels[idx]
        return tensor_img, label, path

def get_dataloaders(
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Instantiate Train, Validation, and Test DataLoaders.
    """
    batch_size = batch_size or config.batch_size
    num_workers = num_workers if num_workers is not None else config.num_workers

    train_transform = get_transforms(config.image_size, is_train=True)
    eval_transform = get_transforms(config.image_size, is_train=False)

    train_ds = PlantVillageDataset(config.train_csv, transform=train_transform, max_samples=max_samples)
    val_ds = PlantVillageDataset(config.val_csv, transform=eval_transform, max_samples=max_samples // 4 if max_samples else None)
    test_ds = PlantVillageDataset(config.test_csv, transform=eval_transform, max_samples=max_samples // 4 if max_samples else None)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.pin_memory
    )

    return train_loader, val_loader, test_loader

def preprocess_single_image(pil_or_path: str | Path | Image.Image | np.ndarray, image_size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Utility to load and preprocess a single image for inference and Grad-CAM visualization.
    Returns: (preprocessed_tensor [1, 3, H, W], rgb_numpy_uint8 [H, W, 3])
    """
    if isinstance(pil_or_path, (str, Path)):
        pil_img = Image.open(pil_or_path).convert("RGB")
    elif isinstance(pil_or_path, np.ndarray):
        pil_img = Image.fromarray(pil_or_path.astype(np.uint8)).convert("RGB")
    else:
        pil_img = pil_or_path.convert("RGB")

    rgb_resized = pil_img.resize((image_size, image_size))
    rgb_np = np.array(rgb_resized)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std)
    ])
    tensor = transform(pil_img).unsqueeze(0)
    return tensor, rgb_np

