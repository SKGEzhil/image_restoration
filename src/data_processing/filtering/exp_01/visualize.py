"""Visualize GT quality metrics -- distributions and sample grids.

Reads metrics.csv and gt_classifications.json, generates:
1. Histogram distributions for each metric
2. Scatter plots (lap_var vs noise_est colored by local_var)
3. Flagged sample grids -- all Mode B samples, batched, sorted best->worst
4. Borderline sample grid (samples near threshold boundaries)

Run: python src/data_processing/exp_01/visualize.py
      python src/data_processing/exp_01/visualize.py --mode flagged
      python src/data_processing/exp_01/visualize.py --mode flagged --batch-size 50
      python src/data_processing/exp_01/visualize.py --mode borderline
"""

import argparse
import csv
import json
import math
import sys
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


def load_metrics():
    metrics_csv = FILTERING_DIR / "metrics.csv"
    rows = []
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "filename": row["filename"],
                "lap_var": float(row["lap_var"]),
                "fft_hf_ratio": float(row["fft_hf_ratio"]),
                "noise_est": float(row["noise_est"]),
                "local_var": float(row["local_var"]),
            })
    return rows


def load_classifications():
    path = OUTPUT_DIR / "gt_classifications.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_flagged_rows(rows, classifications):
    """Return only Mode B rows, merged with classification reasons."""
    if classifications is None:
        return rows
    mode_b = {s["filename"]: s for s in classifications["samples"] if s["mode"] == "B"}
    flagged = []
    for r in rows:
        if r["filename"] in mode_b:
            entry = dict(r)
            entry["reasons"] = mode_b[r["filename"]]["reasons"]
            flagged.append(entry)
    return flagged


def sort_flagged_best_to_worst(flagged):
    """Sort flagged samples from best (least degraded) to worst (most degraded)."""
    if not flagged:
        return flagged

    lap = np.array([r["lap_var"] for r in flagged])
    noise = np.array([r["noise_est"] for r in flagged])

    lap_severity = 1.0 - (lap - lap.min()) / (lap.max() - lap.min() + 1e-10)
    noise_severity = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)

    severity = lap_severity + noise_severity
    order = np.argsort(severity)
    return [flagged[i] for i in order]


def plot_distributions(rows, out_dir):
    """Histograms for each metric."""
    metrics = {
        "lap_var": "Laplacian Variance (HIGH=sharp, LOW=blurry)",
        "fft_hf_ratio": "FFT HF Energy Ratio (HIGH=sharp, LOW=blurry)",
        "noise_est": "Noise Estimation (HIGH=noisy, LOW=clean)",
        "local_var": "Local Variance (HIGH=textured, LOW=flat)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (key, label) in zip(axes.flat, metrics.items()):
        vals = [r[key] for r in rows]
        ax.hist(vals, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(key)

        arr = np.array(vals)
        for p, ls in [(5, "--"), (25, ":"), (50, "-"), (75, ":"), (95, "--")]:
            thr = np.percentile(arr, p)
            ax.axvline(thr, color="red", linestyle=ls, alpha=0.5, label=f"p{p}")

        ax.legend(fontsize=7)

    fig.suptitle("GT Quality Metric Distributions (Training Set)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "distributions.png", dpi=150)
    print(f"  Saved {out_dir / 'distributions.png'}")
    plt.close(fig)


def plot_scatter(rows, out_dir):
    """Scatter: lap_var vs noise_est, colored by local_var."""
    lap = [r["lap_var"] for r in rows]
    noise = [r["noise_est"] for r in rows]
    local = [r["local_var"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(lap, noise, c=local, cmap="viridis", s=8, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Local Variance (texture)")
    ax.set_xlabel("Laplacian Variance (HIGH=sharp)")
    ax.set_ylabel("Noise Estimation (HIGH=noisy)")
    ax.set_title("GT Quality: Laplacian Variance vs Noise\n(colored by local variance)")

    lap_med = np.median(lap)
    noise_med = np.median(noise)
    ax.axvline(lap_med, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(noise_med, color="gray", linestyle="--", alpha=0.3)
    ax.text(lap_med * 1.5, noise_med * 0.3, "Sharp + Clean\n(valid)", ha="center",
            fontsize=10, color="green", fontweight="bold")
    ax.text(lap_med * 0.3, noise_med * 0.3, "Blurry + Clean\n(check)", ha="center",
            fontsize=10, color="orange", fontweight="bold")
    ax.text(lap_med * 1.5, noise_med * 1.8, "Sharp + Noisy\n(degraded)", ha="center",
            fontsize=10, color="red", fontweight="bold")
    ax.text(lap_med * 0.3, noise_med * 1.8, "Blurry + Noisy\n(degraded)", ha="center",
            fontsize=10, color="darkred", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "scatter_lap_vs_noise.png", dpi=150)
    print(f"  Saved {out_dir / 'scatter_lap_vs_noise.png'}")
    plt.close(fig)


def plot_flagged_batch(batch, batch_idx, total_batches, gt_dir, out_dir, cols=6):
    """Render one batch of flagged samples as a grid image."""
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
        reasons_str = ",".join(sample.get("reasons", []))
        axes[r, c].set_title(
            f"{name}\nlv={sample['lap_var']:.4f}\nne={sample['noise_est']:.4f}\n{reasons_str}",
            fontsize=6,
        )

    fig.suptitle(
        f"Flagged GT Samples -- Batch {batch_idx + 1}/{total_batches} "
        f"(best->worst, showing {n} samples)",
        fontsize=13,
    )
    fig.tight_layout()
    fname = out_dir / f"flagged_batch_{batch_idx + 1:03d}.png"
    fig.savefig(fname, dpi=150)
    print(f"  Saved {fname}")
    plt.close(fig)


def plot_flagged_all(rows, out_dir, batch_size, classifications):
    """All Mode B samples sorted best->worst, saved as batched PNGs."""
    cfg = load_config()
    gt_dir = FILTERING_DIR / cfg["gt_dir"]

    flagged = get_flagged_rows(rows, classifications)
    flagged = sort_flagged_best_to_worst(flagged)
    total = len(flagged)

    if total == 0:
        print("  No flagged samples found.")
        return

    total_batches = math.ceil(total / batch_size)
    print(f"  {total} flagged samples -> {total_batches} batches of {batch_size}")

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        plot_flagged_batch(flagged[start:end], batch_idx, total_batches, gt_dir, out_dir)

    csv_path = out_dir / "flagged_sorted.csv"
    with open(csv_path, "w") as f:
        f.write("rank,filename,lap_var,noise_est,local_var,reasons\n")
        for i, s in enumerate(flagged):
            reasons = "|".join(s.get("reasons", []))
            f.write(f"{i + 1},{s['filename']},{s['lap_var']:.8f},{s['noise_est']:.8f},"
                    f"{s['local_var']:.8f},{reasons}\n")
    print(f"  Saved {csv_path}")


def plot_borderline_grid(rows, out_dir, cfg):
    """Grid of samples near threshold boundaries."""
    gt_dir = FILTERING_DIR / cfg["gt_dir"]
    thresholds = {
        "lap_var": cfg.get("lap_var_blur_threshold"),
        "noise_est": cfg.get("noise_est_noisy_threshold"),
        "local_var": cfg.get("local_var_flat_threshold"),
    }
    margin = cfg.get("borderline_margin", 0.1)

    borderline = []
    for r in rows:
        reasons = []
        if thresholds["lap_var"] is not None:
            thr = thresholds["lap_var"]
            if abs(r["lap_var"] - thr) < thr * margin:
                reasons.append("lap_var")
        if thresholds["noise_est"] is not None:
            thr = thresholds["noise_est"]
            if abs(r["noise_est"] - thr) < thr * margin:
                reasons.append("noise_est")
        if thresholds["local_var"] is not None:
            thr = thresholds["local_var"]
            if abs(r["local_var"] - thr) < thr * margin:
                reasons.append("local_var")
        if reasons:
            borderline.append((r, reasons))

    if not borderline:
        print("  No borderline samples found (thresholds may be null). Skipping.")
        return

    n = min(len(borderline), 30)
    borderline = borderline[:n]
    cols = 6
    rows_grid = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_grid, cols, figsize=(15, rows_grid * 2.5))
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.axis("off")

    for i, (r, reasons) in enumerate(borderline):
        rc, c = divmod(i, cols)
        name = r["filename"]
        img = np.load(gt_dir / name)
        axes[rc, c].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[rc, c].set_title(
            f"{name}\n{','.join(reasons)}",
            fontsize=7,
        )

    fig.suptitle("Borderline Samples (near threshold boundaries)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "borderline_grid.png", dpi=150)
    print(f"  Saved {out_dir / 'borderline_grid.png'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "distributions", "scatter", "flagged", "borderline"],
                        default="all")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Number of flagged samples per PNG (default: 30)")
    args = parser.parse_args()

    cfg = load_config()
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_metrics()
    classifications = load_classifications()
    print(f"Loaded {len(rows)} samples")
    if classifications:
        n_b = classifications["summary"]["mode_b"]
        print(f"Mode B (flagged): {n_b} samples")

    if args.mode in ("all", "distributions"):
        print("\n--- Distribution Plots ---")
        plot_distributions(rows, VISUALIZATIONS_DIR)

    if args.mode in ("all", "scatter"):
        print("\n--- Scatter Plot ---")
        plot_scatter(rows, VISUALIZATIONS_DIR)

    if args.mode in ("all", "flagged"):
        print("\n--- Flagged Sample Grid (all Mode B, batched) ---")
        plot_flagged_all(rows, VISUALIZATIONS_DIR, args.batch_size, classifications)

    if args.mode in ("all", "borderline"):
        print("\n--- Borderline Sample Grid ---")
        plot_borderline_grid(rows, VISUALIZATIONS_DIR, cfg)


if __name__ == "__main__":
    main()
