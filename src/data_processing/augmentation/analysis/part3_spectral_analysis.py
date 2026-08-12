"""Part 3: Order Inference via Spectral Analysis.

Determines whether noise was applied before or after downsampling
by analyzing the frequency spectrum of residuals.
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import (
    align_images,
    downsample_candidate,
    ensure_output_dir,
    get_data_pairs,
    load_config,
    load_pair,
    radial_profile,
)


def load_part1_winner(output_dir: Path, filename: str) -> str:
    """Get the winning kernel for a specific image from Part 1 results."""
    results_path = output_dir / "part1_kernel_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            "Part 1 results not found. Run part1_kernel_matching.py first."
        )
    with open(results_path) as f:
        data = json.load(f)
    for entry in data["per_image"]:
        if entry["filename"] == filename:
            return entry["winner"]
    return "bicubic"


def generate_best_candidate(
    gt: np.ndarray, winner: str, factor: int, sigmas: list[float]
) -> np.ndarray:
    """Generate clean LR candidate using the winning kernel."""
    if winner.startswith("gaussian_"):
        sigma = float(winner.split("_")[1])
        return downsample_candidate(gt, "gaussian", sigma=sigma, factor=factor)
    return downsample_candidate(gt, winner, factor=factor)


def classify_spectral_signature(
    residual: np.ndarray, threshold: float
) -> dict:
    """Classify residual spectrum as pre- or post-downsample noise.

    Flat/white spectrum up to Nyquist → post-downsample (noise added after).
    Spectrum rolling off at high frequencies → pre-downsample (noise filtered).

    Returns dict with classification and diagnostic values.
    """
    # 2D power spectrum
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual)))
    power = spectrum ** 2

    # Radial profile
    profile = radial_profile(power)

    # Normalize
    if profile.max() > 0:
        profile = profile / profile.max()
    else:
        return {"label": "unknown", "ratio": 0.0, "profile": []}

    # Split into low and high frequency halves
    n = len(profile)
    low_freq = profile[: n // 2]
    high_freq = profile[n // 2 :]

    low_energy = float(np.mean(low_freq)) if len(low_freq) > 0 else 0.0
    high_energy = float(np.mean(high_freq)) if len(high_freq) > 0 else 0.0

    ratio = high_energy / (low_energy + 1e-10)

    # Classification
    if ratio > threshold:
        label = "post_downsample"  # flat spectrum, full bandwidth
    else:
        label = "pre_downsample"  # rolled off, low-pass filtered

    return {
        "label": label,
        "ratio": ratio,
        "low_freq_energy": low_energy,
        "high_freq_energy": high_energy,
        "profile": profile.tolist(),
    }


def run_part3(config: dict, limit: int | None = None) -> dict:
    """Run spectral analysis across all image pairs.

    Args:
        config: Pipeline configuration dict.
        limit: If set, only process first N images.

    Returns:
        Summary dict with pre/post proportions.
    """
    data_dir = Path(config["data_dir"])
    split = config["split"]
    factor = config["downsample_factor"]
    sigmas = config["gaussian_sigmas"]
    upsample = config["alignment_upsample"]
    threshold = config["spectral_ratio_threshold"]
    output_dir = ensure_output_dir(config)

    pairs = get_data_pairs(data_dir, split)
    if limit:
        pairs = pairs[:limit]
    total = len(pairs)

    results = []
    label_counts = {"pre_downsample": 0, "post_downsample": 0}

    for gt_path, lr_path in tqdm(pairs, desc="Part 3: Spectral analysis"):
        gt, lr = load_pair(gt_path, lr_path)
        name = gt_path.stem

        try:
            winner = load_part1_winner(output_dir, name)
        except (FileNotFoundError, ValueError):
            winner = "bicubic"

        candidate = generate_best_candidate(gt, winner, factor, sigmas)
        aligned, _ = align_images(lr, candidate, upsample_factor=upsample)
        residual = lr - aligned

        spectral = classify_spectral_signature(residual, threshold)
        label_counts[spectral["label"]] = label_counts[spectral["label"]] + 1

        results.append({
            "filename": name,
            "label": spectral["label"],
            "ratio": spectral["ratio"],
            "low_freq_energy": spectral["low_freq_energy"],
            "high_freq_energy": spectral["high_freq_energy"],
        })

    # Compute proportions
    classified = label_counts["pre_downsample"] + label_counts["post_downsample"]
    proportions = {}
    for label, count in label_counts.items():
        proportions[label] = {
            "count": count,
            "pct": round(count / total * 100, 2) if total > 0 else 0.0,
        }

    summary = {
        "total_images": total,
        "classified_images": classified,
        "spectral_proportions": proportions,
        "threshold_used": threshold,
    }

    # Save results
    with open(output_dir / "part3_spectral_results.json", "w") as f:
        json.dump({"per_image": results, "summary": summary}, f, indent=2)
    with open(output_dir / "part3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Part 3: Spectral analysis")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Process only N images")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = run_part3(config, limit=args.limit)

    print("\n=== Part 3: Spectral Analysis Results ===")
    print(f"Total images: {summary['total_images']}")
    print(f"Threshold used: {summary['threshold_used']}")
    for label, info in summary["spectral_proportions"].items():
        print(f"  {label:20s}: {info['count']:5d} ({info['pct']:.1f}%)")
