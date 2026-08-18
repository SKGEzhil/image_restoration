#!/usr/bin/env python3
"""Standalone inference script for KLA Hackathon 2026 image restoration.

Accepts degraded .npy images (128x128) and produces restored .npy images (256x256)
using SCUNetSR with test-time augmentation (8-aug average).

Usage:
    python run.py --input-dir test/NoisyLR --output-dir results/
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import SCUNetSR

# ─── Hardcoded configuration ────────────────────────────────────────────────
CHECKPOINT_PATH = "weights/best.pt"
BATCH_SIZE = 16
# SCUNetSR architecture parameters (must match training config)
MODEL_PARAMS = dict(
    in_nc=1,
    config=[2, 2, 2, 2, 2, 2, 2],
    dim=64,
    drop_path_rate=0.0,
    input_resolution=128,
    up_scale=2,
)


# =============================================================================
# Dataset
# =============================================================================


class NpyFolder(Dataset):
    """Loads .npy files from a directory."""

    def __init__(self, root):
        self.root = Path(root)
        self.names = sorted(p.name for p in self.root.glob("*.npy"))
        if len(self.names) == 0:
            raise FileNotFoundError(f"No .npy files found in {root}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        arr = np.load(self.root / name)
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).float()
        return tensor, name


# =============================================================================
# Model Loading
# =============================================================================


def load_model(checkpoint_path, device):
    """Load SCUNetSR model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SCUNetSR(**MODEL_PARAMS)
    state_dict = ckpt["model"]
    # Strip torch.compile prefixes if present
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    return model.to(device)


# =============================================================================
# Inference
# =============================================================================


@torch.no_grad()
def tta_inference(model, x, device):
    """Test-time augmentation: average over 4 rotations x 2 flips = 8 predictions."""
    preds = []
    for k in (0, 1, 2, 3):
        x_rot = torch.rot90(x, k, [2, 3])
        for do_flip in (False, True):
            x_in = torch.flip(x_rot, [3]) if do_flip else x_rot
            out = model(x_in.to(device)).clamp(0, 1)
            if do_flip:
                out = torch.flip(out, [3])
            out = torch.rot90(out, -k, [2, 3])
            preds.append(out.cpu())
    return torch.stack(preds, dim=0).mean(dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Image restoration inference — SCUNetSR with TTA"
    )
    parser.add_argument("input_dir", type=str,
                        help="Directory containing degraded .npy images")
    parser.add_argument("output_dir", type=str,
                        help="Directory to write restored .npy images")
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = load_model(str(checkpoint_path), device)
    model.eval()
    print(f"Model: SCUNetSR loaded from {checkpoint_path}")

    # Dataset and loader
    ds = NpyFolder(args.input_dir)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    print(f"Found {len(ds)} images in {args.input_dir}")

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    total_start = time.time()
    for batch_idx, (images, names) in enumerate(loader):
        batch_start = time.time()
        images = images.to(device)
        restored = tta_inference(model, images, device)
        batch_time = time.time() - batch_start

        for name, out in zip(names, restored):
            np.save(str(out_dir / name), out.squeeze(0).numpy())

        print(f"  Batch {batch_idx + 1}/{len(loader)}: "
              f"{len(names)} images, {batch_time:.2f}s")

    total_time = time.time() - total_start
    print(f"\nDone: {len(ds)} images restored in {total_time:.2f}s "
          f"({len(ds) / total_time:.1f} images/s)")
    print(f"Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
