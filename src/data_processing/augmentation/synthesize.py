"""Synthesize synthetic NoisyLR images from GT images.

Applies degradation recipes (kernel + noise + order) sampled from
the18-recipe distribution defined in augmentation_config.yaml.

Usage:
    python synthesize.py
    python synthesize.py --config augmentation_config.yaml
    python synthesize.py --num-samples 100 --seed 123
    python synthesize.py --dry-run  # preview without generating
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Noise functions
# ---------------------------------------------------------------------------

def apply_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """Additive Gaussian noise: img + N(0, sigma)."""
    noise = np.random.normal(0, sigma, img.shape)
    return img + noise


def apply_speckle(img: np.ndarray, sigma: float) -> np.ndarray:
    """Multiplicative speckle noise: img * N(1, sigma)."""
    noise = np.random.normal(1, sigma, img.shape)
    return img * noise


def downsample(img: np.ndarray, kernel: str, blur_sigma: float | None = None) -> np.ndarray:
    """Downsample 2x using specified kernel."""
    h, w = img.shape
    new_h, new_w = h // 2, w // 2

    if kernel == "bicubic":
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif kernel == "bilinear":
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    elif kernel == "gaussian-decimate":
        if blur_sigma is None:
            raise ValueError("blur_sigma required for gaussian-decimate kernel")
        ksize = int(np.ceil(blur_sigma * 3)) * 2 + 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)
        return blurred[::2, ::2]
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def apply_noise_stack(
    img: np.ndarray,
    noise_type: str,
    sigma_g: float,
    sigma_s: float,
) -> tuple[np.ndarray, str]:
    """Apply noise(s) in random order for 'both', fixed for single.

    Returns (result, noise_order_applied).
    """
    if noise_type == "gaussian":
        return apply_gaussian(img, sigma_g), "gaussian_only"
    elif noise_type == "speckle":
        return apply_speckle(img, sigma_s), "speckle_only"
    else:  # both
        if np.random.random() < 0.5:
            img = apply_gaussian(img, sigma_g)
            img = apply_speckle(img, sigma_s)
            return img, "gaussian_first"
        else:
            img = apply_speckle(img, sigma_s)
            img = apply_gaussian(img, sigma_g)
            return img, "speckle_first"


# ---------------------------------------------------------------------------
# Recipe execution
# ---------------------------------------------------------------------------

def execute_recipe(
    gt: np.ndarray,
    recipe: dict,
    sigma_g: float,
    sigma_s: float,
    blur_sigma: float | None,
) -> tuple[np.ndarray, str]:
    """Apply degradation according to recipe.

    Returns (synthetic_lr, noise_order_description).
    """
    if recipe["order"] == "pre":
        # Apply noise to full-res GT, then downsample
        img, noise_order = apply_noise_stack(gt, recipe["noise"], sigma_g, sigma_s)
        img = downsample(img, recipe["kernel"], blur_sigma)
    else:  # post
        # Downsample first, then apply noise
        img = downsample(gt, recipe["kernel"], blur_sigma)
        img, noise_order = apply_noise_stack(img, recipe["noise"], sigma_g, sigma_s)

    return img, noise_order


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def build_recipe_distribution(recipes: list[dict]) -> tuple[list[dict], list[float]]:
    """Extract recipe weights for weighted sampling."""
    weights = [r["percentage"] / 100.0 for r in recipes]
    return recipes, weights


def sample_recipe(recipes: list[dict], weights: list[float]) -> dict:
    """Sample a recipe based on percentage distribution."""
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize to sum to 1
    idx = np.random.choice(len(recipes), p=weights)
    return recipes[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthesize synthetic NoisyLR images")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--num-samples", type=int, default=None, help="Override num_augmented_samples")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--exclude", type=str, default=None, help="Override exclude list path")
    parser.add_argument("--dry-run", action="store_true", help="Preview without generating")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else Path(__file__).parent / "augmentation_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override from CLI
    num_samples = args.num_samples or config["num_augmented_samples"]
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    offset = config.get("offset", 3200)

    # Set seed
    np.random.seed(seed)

    # Resolve paths
    config_dir = config_path.parent
    data_dir = (config_dir / config["data_dir"]).resolve()
    split = config["split"]
    gt_dir = data_dir / split / "GT"
    output_dir = config_dir / "outputs" / "synthetic"
    synth_gt_dir = output_dir / "GT"
    synth_lr_dir = output_dir / "NoisyLR"

    # List available GT images
    gt_files = sorted(p.name for p in gt_dir.glob("*.npy"))
    if len(gt_files) == 0:
        raise FileNotFoundError(f"No .npy files found in {gt_dir}")
    print(f"Found {len(gt_files)} GT images in {gt_dir}")

    # Load exclusion list
    exclude_path = args.exclude or config.get("exclude")
    excluded_count = 0
    if exclude_path:
        exclude_file = config_dir / exclude_path
        if exclude_file.exists():
            with open(exclude_file) as f:
                excluded = set(json.load(f)["excluded"])
            before = len(gt_files)
            gt_files = [f for f in gt_files if f not in excluded]
            excluded_count = before - len(gt_files)
            print(f"Excluded {excluded_count} samples ({len(gt_files)} remaining)")
        else:
            print(f"WARNING: Exclude file not found: {exclude_file}")

    if len(gt_files) == 0:
        raise FileNotFoundError("No GT images available after exclusion")

    # Build recipe distribution
    recipes = config["recipes"]
    recipes, weights = build_recipe_distribution(recipes)
    print(f"Using {len(recipes)} recipes (total weight: {sum(weights)*100:.1f}%)")

    # Noise ranges
    g_min, g_max = config["gaussian_sigma_range"]
    s_min, s_max = config["speckle_sigma_range"]
    b_min, b_max = config["blur_sigma_range"]

    print(f"\nAugmentation plan:")
    print(f"  Samples: {num_samples}")
    print(f"  Offset: {offset}")
    print(f"  Output: {output_dir}")
    print(f"  Seed: {seed}")
    print(f"  Gaussian sigma: [{g_min}, {g_max}]")
    print(f"  Speckle sigma: [{s_min}, {s_max}]")
    print(f"  Blur sigma: [{b_min}, {b_max}]")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        # Show recipe distribution
        print("\nRecipe distribution:")
        for r in recipes:
            count = int(round(r["percentage"] / 100.0 * num_samples))
            print(f"  Recipe {r['id']:2d}: {r['kernel']:18s} {r['noise']:8s} {r['order']:4s} "
                  f"{r['percentage']:5.1f}% -> ~{count} samples")
        return

    # Create output directories
    synth_gt_dir.mkdir(parents=True, exist_ok=True)
    synth_lr_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    start_time = time.time()
    log_entries = []

    for i in tqdm(range(num_samples), desc="Synthesizing"):
        # Sample GT image
        src_gt_name = np.random.choice(gt_files)
        gt = np.load(gt_dir / src_gt_name).astype(np.float64)

        # Sample recipe
        recipe = sample_recipe(recipes, weights)

        # Sample noise parameters
        sigma_g = np.random.uniform(g_min, g_max)
        sigma_s = np.random.uniform(s_min, s_max)
        blur_sigma = np.random.uniform(b_min, b_max) if recipe["kernel"] == "gaussian-decimate" else None

        # Execute recipe
        synthetic_lr, noise_order = execute_recipe(gt, recipe, sigma_g, sigma_s, blur_sigma)

        # Determine output filename
        out_name = f"{offset + i:06d}.npy"

        # Save
        np.save(synth_gt_dir / out_name, gt.astype(np.float32))
        np.save(synth_lr_dir / out_name, synthetic_lr.astype(np.float32))

        # Log entry
        log_entries.append({
            "synthetic_name": out_name,
            "source_gt": src_gt_name,
            "recipe_id": recipe["id"],
            "kernel": recipe["kernel"],
            "noise": recipe["noise"],
            "order": recipe["order"],
            "noise_order": noise_order,
            "sigma_g": float(sigma_g),
            "sigma_s": float(sigma_s),
            "blur_sigma": float(blur_sigma) if blur_sigma is not None else None,
        })

    elapsed = time.time() - start_time

    # Save augmentation log
    log = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path),
        "seed": seed,
        "num_samples": num_samples,
        "offset": offset,
        "source_gt_dir": str(gt_dir),
        "output_dir": str(output_dir),
        "elapsed_seconds": round(elapsed, 1),
        "excluded_count": excluded_count,
        "exclude_path": exclude_path,
        "noise_config": {
            "gaussian_sigma_range": config["gaussian_sigma_range"],
            "speckle_sigma_range": config["speckle_sigma_range"],
            "blur_sigma_range": config["blur_sigma_range"],
        },
        "recipe_distribution": [
            {"id": r["id"], "kernel": r["kernel"], "noise": r["noise"],
             "order": r["order"], "percentage": r["percentage"]}
            for r in recipes
        ],
        "samples": log_entries,
    }

    log_path = output_dir.parent / "augmentation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Generated {num_samples} synthetic pairs")
    print(f"  GT:  {synth_gt_dir}")
    print(f"  LR:  {synth_lr_dir}")
    print(f"  Log: {log_path}")

    # Print recipe distribution summary
    print(f"\nRecipe distribution:")
    recipe_counts = {}
    for entry in log_entries:
        key = f"R{entry['recipe_id']:02d}"
        recipe_counts[key] = recipe_counts.get(key, 0) + 1
    for key in sorted(recipe_counts):
        print(f"  {key}: {recipe_counts[key]} ({recipe_counts[key]/num_samples*100:.1f}%)")


if __name__ == "__main__":
    main()
