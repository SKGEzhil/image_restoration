import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_sample(data_dir: str, index: int):
    gt = np.load(os.path.join(data_dir, "train", "GT", f"{index:06d}.npy"))
    noisy = np.load(os.path.join(data_dir, "train", "NoisyLR", f"{index:06d}.npy"))
    return gt, noisy


def visualize(index: int, gt: np.ndarray, noisy: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(gt, cmap="gray")
    axes[0].set_title(f"GT ({gt.shape[0]}x{gt.shape[1]})")
    axes[0].axis("off")

    axes[1].imshow(noisy, cmap="gray")
    axes[1].set_title(f"NoisyLR ({noisy.shape[0]}x{noisy.shape[1]})")
    axes[1].axis("off")

    fig.suptitle(f"Sample {index} — GT vs NoisyLR")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GT and NoisyLR of a data sample")
    parser.add_argument("index", type=int, nargs="?", default=0, help="Sample index (default: 0)")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent / "src" / "data"))
    args = parser.parse_args()

    sample_index = args.index
    gt, noisy = load_sample(args.data_dir, sample_index)
    print(f"GT: {gt.shape} {gt.dtype} range [{gt.min():.4f}, {gt.max():.4f}]")
    print(f"NoisyLR: {noisy.shape} {noisy.dtype} range [{noisy.min():.4f}, {noisy.max():.4f}]")
    visualize(sample_index, gt, noisy)