import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_sample(data_dir: str, index: int):
    gt = np.load(os.path.join(data_dir, "train", "GT", f"{index:06d}.npy"))
    noisy = np.load(os.path.join(data_dir, "train", "NoisyLR", f"{index:06d}.npy"))
    return gt, noisy


def load_test_sample(runs_dir: str, run_id: str, index: int):
    out = np.load(os.path.join(runs_dir, run_id, "outputs", f"{index:06d}.npy"))
    return out.squeeze()


def load_test_metrics(runs_dir: str, run_id: str, index: int):
    name = f"{index:06d}.npy"
    details_file = os.path.join(runs_dir, run_id, "test_details.json")
    if not os.path.exists(details_file):
        return None
    details = json.loads(Path(details_file).read_text())
    for entry in details.get("per_sample", []):
        if entry["name"] == name:
            return entry
    return None


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


def visualize_test(index: int, gt: np.ndarray, noisy: np.ndarray, pred: np.ndarray, run_id: str, metrics=None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gt, cmap="gray")
    axes[0].set_title(f"GT ({gt.shape[0]}x{gt.shape[1]})")
    axes[0].axis("off")

    axes[1].imshow(noisy, cmap="gray")
    axes[1].set_title(f"NoisyLR ({noisy.shape[0]}x{noisy.shape[1]})")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title(f"Output ({pred.shape[0]}x{pred.shape[1]})")
    axes[2].axis("off")

    title = f"Sample {index} — GT vs NoisyLR vs Output ({run_id})"
    if metrics:
        title += f"\nL1 {metrics['L1']:.4f} | SSIM {metrics['SSIM']:.4f} | PSNR {metrics['PSNR']:.2f} dB"
    fig.suptitle(title)
    if metrics:
        fig.text(0.99, 0.01,
                 f"L1: {metrics['L1']:.4f}   SSIM: {metrics['SSIM']:.4f}   PSNR: {metrics['PSNR']:.2f} dB",
                 ha="right", va="bottom", fontsize=10, color="dimgray")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GT and NoisyLR of a data sample")
    parser.add_argument("index", type=int, nargs="?", default=None, help="Sample index (default: first sample)")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent / "src" / "data"))
    parser.add_argument("--test", action="store_true",
                        help="Visualize a model output from runs/<run_id>/outputs against the test split")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run id whose outputs/ directory holds restored test samples (required with --test)")
    parser.add_argument("--runs-dir", type=str, default=str(Path(__file__).resolve().parent / "runs"))
    args = parser.parse_args()

    if args.test:
        if not args.run_id:
            parser.error("--test requires --run-id")
        if args.index is None:
            out_dir = os.path.join(args.runs_dir, args.run_id, "outputs")
            sample_index = int(sorted(p[:-4] for p in os.listdir(out_dir) if p.endswith(".npy"))[0])
        else:
            sample_index = args.index
        gt = np.load(os.path.join(args.data_dir, "test", "GT", f"{sample_index:06d}.npy"))
        noisy = np.load(os.path.join(args.data_dir, "test", "NoisyLR", f"{sample_index:06d}.npy"))
        pred = load_test_sample(args.runs_dir, args.run_id, sample_index)
        metrics = load_test_metrics(args.runs_dir, args.run_id, sample_index)
        print(f"GT: {gt.shape} {gt.dtype} range [{gt.min():.4f}, {gt.max():.4f}]")
        print(f"NoisyLR: {noisy.shape} {noisy.dtype} range [{noisy.min():.4f}, {noisy.max():.4f}]")
        print(f"Output: {pred.shape} {pred.dtype} range [{pred.min():.4f}, {pred.max():.4f}]")
        if metrics:
            print(f"Metrics: L1 {metrics['L1']:.4f} | SSIM {metrics['SSIM']:.4f} | PSNR {metrics['PSNR']:.2f} dB")
        else:
            print("Metrics: no test_details.json found for this run")
        visualize_test(sample_index, gt, noisy, pred, args.run_id, metrics)
    else:
        if args.index is None:
            sample_index = 0
        else:
            sample_index = args.index
        gt, noisy = load_sample(args.data_dir, sample_index)
        print(f"GT: {gt.shape} {gt.dtype} range [{gt.min():.4f}, {gt.max():.4f}]")
        print(f"NoisyLR: {noisy.shape} {noisy.dtype} range [{noisy.min():.4f}, {noisy.max():.4f}]")
        visualize(sample_index, gt, noisy)