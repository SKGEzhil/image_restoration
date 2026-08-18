"""Paired dataset + shared helpers for train.py and test.py."""

import json
import random
from pathlib import Path

import numpy as np
import torch


class NoisyLROnly(torch.utils.data.Dataset):
    """Loads only NoisyLR npy images (no GT required).

    Used for inference/submission when ground truth is unavailable.
    """

    def __init__(self, root, split="test"):
        self.root = Path(root)
        self.split = split
        lr_dir = self.root / split / "NoisyLR"
        self.names = sorted(p.name for p in lr_dir.glob("*.npy"))
        if len(self.names) == 0:
            raise FileNotFoundError(f"No samples found in {lr_dir}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        lr = np.load(self.root / self.split / "NoisyLR" / name)
        lr = torch.from_numpy(np.ascontiguousarray(lr)).unsqueeze(0).float()
        return lr, name


class PairedDataset(torch.utils.data.Dataset):
    """Loads paired (NoisyLR, GT) npy images for a given split.

    GT: 256x256 float32 in [0, 1]; NoisyLR: 128x128 float32 (unclamped).
    When crop_size is set, random crops are taken (LR: crop_size, GT: crop_size * scale_factor).
    """

    def __init__(self, root, split="train", augment=False, seed=None,
                 exclude_list=None, include_augmented_data=True,
                 augmentation_offset=3200, crop_size=None, scale_factor=2):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.rng = random.Random(seed)

        lr_dir = self.root / split / "NoisyLR"
        gt_dir = self.root / split / "GT"
        lr_files = sorted(p.name for p in lr_dir.glob("*.npy"))
        gt_files = sorted(p.name for p in gt_dir.glob("*.npy"))
        assert lr_files == gt_files, f"{split}: NoisyLR/GT filenames do not match"
        self.names = lr_files

        # Filter out excluded samples (non-destructive)
        if exclude_list is not None:
            exclude_path = Path(exclude_list)
            if exclude_path.exists():
                with open(exclude_path) as f:
                    excluded = set(json.load(f)["excluded"])
                before = len(self.names)
                self.names = [n for n in self.names if n not in excluded]
                print(f"Excluded {before - len(self.names)} samples "
                      f"({len(self.names)} remaining)")

        # Filter out augmented samples if not included
        if not include_augmented_data and split == "train":
            offset_str = f"{augmentation_offset:06d}"
            before = len(self.names)
            self.names = [n for n in self.names if n < offset_str]
            print(f"Excluded augmented data (>= {offset_str}): "
                  f"{before - len(self.names)} removed ({len(self.names)} remaining)")

        if len(self.names) == 0:
            raise FileNotFoundError(f"No samples found in {lr_dir}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        lr = np.load(self.root / self.split / "NoisyLR" / name)
        gt = np.load(self.root / self.split / "GT" / name)

        if self.crop_size is not None:
            lr, gt = self._random_crop(lr, gt)

        if self.augment:
            lr, gt = self._transform(lr, gt)

        lr = torch.from_numpy(np.ascontiguousarray(lr)).unsqueeze(0).float()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0).float()
        return lr, gt, name

    def _transform(self, lr, gt):
        if self.rng.random() < 0.5:
            lr = np.flip(lr, axis=0)
            gt = np.flip(gt, axis=0)
        if self.rng.random() < 0.5:
            lr = np.flip(lr, axis=1)
            gt = np.flip(gt, axis=1)
        k = self.rng.choice([0, 1, 2, 3])
        if k:
            lr = np.rot90(lr, k)
            gt = np.rot90(gt, k)
        return np.ascontiguousarray(lr), np.ascontiguousarray(gt)

    def _random_crop(self, lr, gt):
        """Random crop: LR to crop_size, GT to crop_size * scale_factor."""
        h, w = lr.shape
        cs = self.crop_size
        gt_cs = cs * self.scale_factor

        if h < cs or w < cs:
            return lr, gt  # image smaller than crop, skip

        i = self.rng.randint(0, h - cs)
        j = self.rng.randint(0, w - cs)

        lr_crop = lr[i:i + cs, j:j + cs]
        gt_i, gt_j = i * self.scale_factor, j * self.scale_factor
        gt_crop = gt[gt_i:gt_i + gt_cs, gt_j:gt_j + gt_cs]
        return lr_crop, gt_crop


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)