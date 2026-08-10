"""Paired dataset + shared helpers for train.py and test.py."""

import random
from pathlib import Path

import numpy as np
import torch


class PairedDataset(torch.utils.data.Dataset):
    """Loads paired (NoisyLR, GT) npy images for a given split.

    GT: 256x256 float32 in [0, 1]; NoisyLR: 128x128 float32 (unclamped).
    """

    def __init__(self, root, split="train", augment=False, seed=None):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.rng = random.Random(seed)

        lr_dir = self.root / split / "NoisyLR"
        gt_dir = self.root / split / "GT"
        lr_files = sorted(p.name for p in lr_dir.glob("*.npy"))
        gt_files = sorted(p.name for p in gt_dir.glob("*.npy"))
        assert lr_files == gt_files, f"{split}: NoisyLR/GT filenames do not match"
        self.names = lr_files

        if len(self.names) == 0:
            raise FileNotFoundError(f"No samples found in {lr_dir}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        lr = np.load(self.root / self.split / "NoisyLR" / name)
        gt = np.load(self.root / self.split / "GT" / name)

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