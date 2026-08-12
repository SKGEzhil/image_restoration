"""Part 1: Kernel Matching Test — infer downsampling method.

Tests bicubic, bilinear, and gaussian-blur+decimate kernels to find
which best matches the real NoisyLR from GT.
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import (
    align_images,
    compute_flat_mask,
    downsample_candidate,
    ensure_output_dir,
    get_data_pairs,
    load_config,
    load_pair,
)


def run_part1(config: dict, limit: int | None = None) -> dict:
    """Run kernel matching analysis across all image pairs.

    Args:
        config: Pipeline configuration dict.
        limit: If set, only process first N images (for quick testing).

    Returns:
        Summary dict with kernel win rates and sigma distribution.
    """
    data_dir = Path(config["data_dir"])
    split = config["split"]
    factor = config["downsample_factor"]
    sigmas = config["gaussian_sigmas"]
    flat_pct = config["flat_percentile"]
    flat_win = config["flat_window"]
    upsample = config["alignment_upsample"]
    output_dir = ensure_output_dir(config)

    pairs = get_data_pairs(data_dir, split)
    if limit:
        pairs = pairs[:limit]
    total = len(pairs)

    # Build candidate names: bicubic, bilinear, gaussian with each sigma
    candidate_names = ["bicubic", "bilinear"] + [f"gaussian_{s}" for s in sigmas]

    # Per-image results
    results = []
    win_counts = {name: 0 for name in candidate_names}
    gaussian_winning_sigmas = []

    # Flat mask cache (reuse across candidates for same image)
    flat_mask_cache = {}

    for gt_path, lr_path in tqdm(pairs, desc="Part 1: Kernel matching"):
        gt, lr = load_pair(gt_path, lr_path)
        name = gt_path.stem

        # Generate all candidates
        candidates = {}
        for cname in candidate_names:
            if cname.startswith("gaussian_"):
                sigma = float(cname.split("_")[1])
                candidates[cname] = downsample_candidate(
                    gt, "gaussian", sigma=sigma, factor=factor
                )
            else:
                candidates[cname] = downsample_candidate(gt, cname, factor=factor)

        # Use bicubic as reference for flat mask (as per expert spec)
        bicubic = candidates["bicubic"]
        flat_mask = compute_flat_mask(bicubic, window=flat_win, percentile=flat_pct)

        # Align each candidate to NoisyLR and compute residual energy in flat regions
        energies = {}
        aligned_candidates = {}
        for cname, candidate in candidates.items():
            aligned, shift_vec = align_images(lr, candidate, upsample_factor=upsample)
            aligned_candidates[cname] = aligned
            residual = lr - aligned
            energy = float(np.mean(residual[flat_mask] ** 2))
            energies[cname] = energy

        # Find winner
        winner = min(energies, key=energies.get)
        win_counts[winner] += 1

        if winner.startswith("gaussian_"):
            gaussian_winning_sigmas.append(float(winner.split("_")[1]))

        results.append({
            "filename": name,
            "winner": winner,
            "energies": energies,
        })

    # Build summary
    kernel_win_rates = {
        name: {"count": win_counts[name], "pct": round(win_counts[name] / total * 100, 2)}
        for name in candidate_names
    }

    # Determine overall winner
    overall_winner = max(win_counts, key=win_counts.get)

    summary = {
        "total_images": total,
        "overall_winner": overall_winner,
        "kernel_win_rates": kernel_win_rates,
        "gaussian_sigma_distribution": {},
    }

    if gaussian_winning_sigmas:
        sigmas_arr = np.array(gaussian_winning_sigmas)
        summary["gaussian_sigma_distribution"] = {
            "count": len(gaussian_winning_sigmas),
            "mean": float(np.mean(sigmas_arr)),
            "std": float(np.std(sigmas_arr)),
            "p5": float(np.percentile(sigmas_arr, 5)),
            "p25": float(np.percentile(sigmas_arr, 25)),
            "p50": float(np.percentile(sigmas_arr, 50)),
            "p75": float(np.percentile(sigmas_arr, 75)),
            "p95": float(np.percentile(sigmas_arr, 95)),
            "min": float(np.min(sigmas_arr)),
            "max": float(np.max(sigmas_arr)),
        }

    # Save results
    with open(output_dir / "part1_kernel_results.json", "w") as f:
        json.dump({"per_image": results, "summary": summary}, f, indent=2)
    with open(output_dir / "part1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Part 1: Kernel matching test")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Process only N images")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = run_part1(config, limit=args.limit)

    print("\n=== Part 1: Kernel Matching Results ===")
    print(f"Total images: {summary['total_images']}")
    print(f"Overall winner: {summary['overall_winner']}")
    for name, info in summary["kernel_win_rates"].items():
        print(f"  {name:20s}: {info['count']:5d} ({info['pct']:.1f}%)")
    if summary["gaussian_sigma_distribution"]:
        print(f"\nGaussian sigma stats:")
        for k, v in summary["gaussian_sigma_distribution"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
