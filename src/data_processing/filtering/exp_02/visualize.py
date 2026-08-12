"""Visualize flagged GT samples separately for noise and blur (exp_02).

Each dimension gets its own visualization folder with:
- Distribution histogram with threshold line
- Batched PNGs of flagged samples sorted worst->best
- Full ranked CSV

Run: python src/data_processing/exp_02/visualize.py --dimension noise
      python src/data_processing/exp_02/visualize.py --dimension blur --batch-size 50
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

EXP_DIR = Path(__file__).parent
FILTERING_DIR = EXP_DIR.parent
CONFIG_PATH = EXP_DIR / "config.yaml"
OUTPUT_DIR = EXP_DIR / "outputs"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_classifications(dimension):
    path = OUTPUT_DIR / dimension / "classifications.json"
    with open(path) as f:
        return json.load(f)


def get_flagged_sorted(classifications, dimension):
    """Return flagged samples sorted worst->best."""
    flagged = [s for s in classifications["samples"] if s["flagged"]]

    if dimension == "noise":
        flagged.sort(key=lambda s: s["noise_est"], reverse=True)
    elif dimension == "blur":
        flagged.sort(key=lambda s: s["lap_var"])

    return flagged


def plot_distribution(classifications, dimension, out_dir):
    """Histogram of the metric for this dimension with threshold line."""
    threshold = classifications["threshold"]
    samples = classifications["samples"]

    if dimension == "noise":
        vals = [s["noise_est"] for s in samples]
        metric_label = "Noise Estimation (HIGH = noisy)"
        thr_label = f"threshold = {threshold:.6f}"
    else:
        vals = [s["lap_var"] for s in samples]
        metric_label = "Laplacian Variance (LOW = blurry)"
        thr_label = f"threshold = {threshold:.6f}"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Count")
    ax.set_title(f"{dimension.upper()} Dimension -- {metric_label}")

    if dimension == "noise":
        ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=thr_label)
        ax.axvspan(threshold, max(vals), alpha=0.15, color="red")
    else:
        ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=thr_label)
        ax.axvspan(min(vals), threshold, alpha=0.15, color="red")

    flagged = sum(1 for s in samples if s["flagged"])
    total = len(samples)
    ax.legend(fontsize=11)
    ax.text(0.98, 0.95, f"Flagged: {flagged} / {total} ({flagged / total * 100:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(out_dir / "distribution.png", dpi=150)
    print(f"  Saved {out_dir / 'distribution.png'}")
    plt.close(fig)


def plot_batch(batch, batch_idx, total_batches, dimension, gt_dir, out_dir, cols=6):
    """Render one batch of flagged samples as a grid."""
    n = len(batch)
    rows_grid = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_grid, cols, figsize=(15, rows_grid * 2.5))
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.axis("off")

    for i, sample in enumerate(batch):
        r, c = divmod(i, cols)
        name = sample["filename"]
        img = np.load(gt_dir / name)
        axes[r, c].imshow(img, cmap="gray", vmin=0, vmax=1)

        if dimension == "noise":
            val_str = f"ne={sample['noise_est']:.5f}"
        else:
            val_str = f"lv={sample['lap_var']:.6f}"

        axes[r, c].set_title(f"{name}\n{val_str}", fontsize=6)

    fig.suptitle(
        f"{dimension.upper()} -- Batch {batch_idx + 1}/{total_batches} "
        f"(worst->best, {n} samples)",
        fontsize=13,
    )
    fig.tight_layout()
    fname = out_dir / f"sorted_batch_{batch_idx + 1:03d}.png"
    fig.savefig(fname, dpi=150)
    print(f"  Saved {fname}")
    plt.close(fig)


def plot_all_batches(flagged, dimension, gt_dir, out_dir, batch_size):
    """Generate batched PNGs for all flagged samples."""
    total = len(flagged)
    if total == 0:
        print(f"  No flagged samples for {dimension}.")
        return

    total_batches = math.ceil(total / batch_size)
    print(f"  {total} flagged samples -> {total_batches} batches of {batch_size}")

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        plot_batch(flagged[start:end], batch_idx, total_batches, dimension, gt_dir, out_dir)

    csv_path = out_dir / "sorted.csv"
    with open(csv_path, "w") as f:
        f.write("rank,filename,lap_var,noise_est,local_var,fft_hf_ratio,flagged\n")
        for i, s in enumerate(flagged):
            f.write(f"{i + 1},{s['filename']},{s['lap_var']:.8f},{s['noise_est']:.8f},"
                    f"{s['local_var']:.8f},{s['fft_hf_ratio']:.6f},{s['flagged']}\n")
    print(f"  Saved {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", choices=["noise", "blur"], required=True)
    parser.add_argument("--batch-size", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config()
    dim = args.dimension
    gt_dir = FILTERING_DIR / cfg["gt_dir"]
    dim_out_dir = OUTPUT_DIR / dim
    dim_out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = dim_out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    classifications = load_classifications(dim)
    flagged = get_flagged_sorted(classifications, dim)
    print(f"Loaded {len(classifications['samples'])} samples, "
          f"{len(flagged)} flagged for {dim}")

    print(f"\n--- Distribution ---")
    plot_distribution(classifications, dim, vis_dir)

    print(f"\n--- Batched Grids ---")
    plot_all_batches(flagged, dim, gt_dir, vis_dir, args.batch_size)


if __name__ == "__main__":
    main()
