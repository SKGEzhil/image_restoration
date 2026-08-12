"""Part 2: Noise Level Estimation — separate Gaussian vs speckle noise.

Uses intensity-binned variance analysis on residuals to decompose
noise into additive (Gaussian) and multiplicative (speckle) components.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from utils import (
    align_images,
    downsample_candidate,
    ensure_output_dir,
    get_data_pairs,
    load_config,
    load_pair,
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
    raise ValueError(f"Image {filename} not found in Part 1 results")


def generate_best_candidate(
    gt: np.ndarray, winner: str, factor: int, sigmas: list[float]
) -> np.ndarray:
    """Generate clean LR candidate using the winning kernel."""
    if winner.startswith("gaussian_"):
        sigma = float(winner.split("_")[1])
        return downsample_candidate(gt, "gaussian", sigma=sigma, factor=factor)
    return downsample_candidate(gt, winner, factor=factor)


def estimate_noise_per_image(
    gt: np.ndarray,
    lr: np.ndarray,
    winner: str,
    factor: int,
    sigmas: list[float],
    num_bins: int,
    upsample: int,
) -> dict | None:
    """Estimate noise parameters for a single image.

    Returns dict with (a, b, c) polynomial fit coefficients,
    or None if computation fails (e.g., too few valid bins).
    """
    candidate = generate_best_candidate(gt, winner, factor, sigmas)
    aligned, _ = align_images(lr, candidate, upsample_factor=upsample)
    residual = lr - aligned

    # Bin pixels by intensity (use aligned candidate as reference)
    ref = aligned
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_variances = []
    valid_centers = []

    for i in range(num_bins):
        mask = (ref >= bin_edges[i]) & (ref < bin_edges[i + 1])
        count = np.sum(mask)
        if count < 10:  # skip bins with too few pixels
            continue
        bin_var = float(np.var(residual[mask]))
        bin_variances.append(bin_var)
        valid_centers.append(bin_centers[i])

    if len(valid_centers) < 4:
        return None

    valid_centers = np.array(valid_centers)
    bin_variances = np.array(bin_variances)

    # Fit: var(I) = a + b*I + c*I^2
    coeffs = np.polyfit(valid_centers, bin_variances, 2)
    # polyfit returns [c, b, a] (highest degree first)
    c_coeff = float(coeffs[0])  # quadratic term (speckle)
    b_coeff = float(coeffs[1])  # linear term (shot noise)
    a_coeff = float(coeffs[2])  # constant term (Gaussian)

    # Ensure non-negative physical interpretation
    a_coeff = max(a_coeff, 0.0)
    c_coeff = max(c_coeff, 0.0)

    return {
        "a": a_coeff,
        "b": b_coeff,
        "c": c_coeff,
        "num_valid_bins": len(valid_centers),
        "bin_centers": valid_centers.tolist(),
        "bin_variances": bin_variances.tolist(),
    }


def run_part2(config: dict, limit: int | None = None) -> dict:
    """Run noise estimation analysis across all image pairs.

    Args:
        config: Pipeline configuration dict.
        limit: If set, only process first N images.

    Returns:
        Summary dict with sigma ranges and cluster proportions.
    """
    data_dir = Path(config["data_dir"])
    split = config["split"]
    factor = config["downsample_factor"]
    sigmas = config["gaussian_sigmas"]
    num_bins = config["num_intensity_bins"]
    num_clusters = config["num_clusters"]
    upsample = config["alignment_upsample"]
    output_dir = ensure_output_dir(config)

    pairs = get_data_pairs(data_dir, split)
    if limit:
        pairs = pairs[:limit]
    total = len(pairs)

    all_fits = []
    failed_count = 0

    for gt_path, lr_path in tqdm(pairs, desc="Part 2: Noise estimation"):
        gt, lr = load_pair(gt_path, lr_path)
        name = gt_path.stem

        try:
            winner = load_part1_winner(output_dir, name)
        except (FileNotFoundError, ValueError):
            # Fallback to bicubic if Part 1 results missing
            winner = "bicubic"

        result = estimate_noise_per_image(
            gt, lr, winner, factor, sigmas, num_bins, upsample
        )

        if result is None:
            failed_count += 1
            continue

        result["filename"] = name
        result["winner_kernel"] = winner
        all_fits.append(result)

    if len(all_fits) < num_clusters:
        raise RuntimeError(
            f"Too few successful fits ({len(all_fits)}) for clustering."
        )

    # Extract (a, b, c) for clustering
    abc_matrix = np.array([[f["a"], f["b"], f["c"]] for f in all_fits])

    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(abc_matrix)

    # Assign labels: identify cluster types by center values
    centers = kmeans.cluster_centers_
    # Sort clusters by 'a' (Gaussian) descending to label them
    sorted_indices = np.argsort(-centers[:, 0])  # highest a first

    cluster_assignments = {}
    label_names = []
    for rank, idx in enumerate(sorted_indices):
        a_val = centers[idx, 0]
        c_val = centers[idx, 2]
        # Determine type by relative dominance
        if a_val > 2 * c_val:
            ctype = "gaussian_only"
        elif c_val > 2 * a_val:
            ctype = "speckle_only"
        else:
            ctype = "mixed"
        label_names.append(ctype)
        cluster_assignments[int(idx)] = ctype

    # Map cluster labels to type names
    for i, fit in enumerate(all_fits):
        cluster_id = labels[i]
        fit["cluster"] = cluster_assignments[cluster_id]

    # Compute cluster proportions
    cluster_counts = {}
    for fit in all_fits:
        ct = fit["cluster"]
        cluster_counts[ct] = cluster_counts.get(ct, 0) + 1

    cluster_proportions = {
        ct: {"count": count, "pct": round(count / len(all_fits) * 100, 2)}
        for ct, count in cluster_counts.items()
    }

    # Compute sigma distributions
    a_values = abc_matrix[:, 0]
    c_values = abc_matrix[:, 2]

    gaussian_sigmas = np.sqrt(a_values)
    speckle_sigmas = np.sqrt(c_values)

    def distribution_stats(arr: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    summary = {
        "total_images": total,
        "successful_fits": len(all_fits),
        "failed_fits": failed_count,
        "gaussian_noise": {
            "sigma_distribution": distribution_stats(gaussian_sigmas),
            "sigma_range_p5_p95": [
                float(np.percentile(gaussian_sigmas, 5)),
                float(np.percentile(gaussian_sigmas, 95)),
            ],
        },
        "speckle_noise": {
            "sigma_distribution": distribution_stats(speckle_sigmas),
            "sigma_range_p5_p95": [
                float(np.percentile(speckle_sigmas, 5)),
                float(np.percentile(speckle_sigmas, 95)),
            ],
        },
        "cluster_proportions": cluster_proportions,
        "cluster_centers": {
            cluster_assignments[i]: centers[i].tolist()
            for i in range(num_clusters)
        },
    }

    # Save results
    with open(output_dir / "part2_noise_results.json", "w") as f:
        json.dump({"per_image": all_fits, "summary": summary}, f, indent=2)
    with open(output_dir / "part2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Part 2: Noise level estimation")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Process only N images")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = run_part2(config, limit=args.limit)

    print("\n=== Part 2: Noise Estimation Results ===")
    print(f"Total images: {summary['total_images']}")
    print(f"Successful fits: {summary['successful_fits']}")
    print(f"Failed fits: {summary['failed_fits']}")
    print(f"\nGaussian noise sigma range (p5-p95): {summary['gaussian_noise']['sigma_range_p5_p95']}")
    print(f"Speckle noise sigma range (p5-p95): {summary['speckle_noise']['sigma_range_p5_p95']}")
    print(f"\nCluster proportions:")
    for ct, info in summary["cluster_proportions"].items():
        print(f"  {ct:15s}: {info['count']:5d} ({info['pct']:.1f}%)")
