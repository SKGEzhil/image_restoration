"""Degradation Estimation Pipeline — Main orchestrator.

Runs Parts 1–4 sequentially to estimate downsampling kernel, noise
levels, and degradation order from GT/NoisyLR pairs.

Usage:
    python analyze_dataset.py                    # run all parts
    python analyze_dataset.py --part 1           # run only Part 1
    python analyze_dataset.py --part 2           # run only Part 2
    python analyze_dataset.py --resume           # skip completed parts
    python analyze_dataset.py --limit 50         # quick test on 50 images
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from utils import load_config, get_data_pairs, load_pair, ensure_output_dir


def explore_data(config: dict) -> None:
    """Print data value range statistics for a sample of images."""
    data_dir = Path(config["data_dir"])
    split = config["split"]
    pairs = get_data_pairs(data_dir, split)

    sample_indices = np.linspace(0, len(pairs) - 1, min(20, len(pairs)), dtype=int)
    gt_mins, gt_maxs, lr_mins, lr_maxs = [], [], [], []

    for i in sample_indices:
        gt, lr = load_pair(pairs[i][0], pairs[i][1])
        gt_mins.append(gt.min())
        gt_maxs.append(gt.max())
        lr_mins.append(lr.min())
        lr_maxs.append(lr.max())

    print("=== Data Value Range (sample of 20 images) ===")
    print(f"  GT:   min=[{min(gt_mins):.4f}, {max(gt_mins):.4f}], "
          f"max=[{min(gt_maxs):.4f}, {max(gt_maxs):.4f}]")
    print(f"  LR:   min=[{min(lr_mins):.4f}, {max(lr_mins):.4f}], "
          f"max=[{min(lr_maxs):.4f}, {max(lr_maxs):.4f}]")
    print(f"  GT shapes: {load_pair(pairs[0][0], pairs[0][1])[0].shape}")
    print(f"  LR shapes: {load_pair(pairs[0][0], pairs[0][1])[1].shape}")
    print(f"  Total pairs: {len(pairs)}")


def part_is_done(output_dir: Path, part_num: int) -> bool:
    """Check if a part's summary file exists."""
    summary_map = {
        1: "part1_summary.json",
        2: "part2_summary.json",
        3: "part3_summary.json",
        4: "analysis_report.json",
    }
    return (output_dir / summary_map[part_num]).exists()


def main():
    parser = argparse.ArgumentParser(
        description="Degradation Estimation Pipeline"
    )
    parser.add_argument(
        "--part", choices=["1", "2", "3", "4", "all"], default="all",
        help="Which part to run (default: all)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: same directory)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only N images (for quick testing)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip parts that already have output files"
    )
    parser.add_argument(
        "--explore-only", action="store_true",
        help="Only print data value ranges, don't run analysis"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)

    # Print data ranges first
    explore_data(config)

    if args.explore_only:
        return

    parts = [int(args.part)] if args.part != "all" else [1, 2, 3, 4]

    start = time.time()
    for part_num in parts:
        if args.resume and part_is_done(output_dir, part_num):
            print(f"\n--- Part {part_num}: SKIPPED (already done) ---")
            continue

        print(f"\n{'='*50}")
        print(f"--- Part {part_num}: Starting ---")
        print(f"{'='*50}")

        t0 = time.time()

        if part_num == 1:
            from part1_kernel_matching import run_part1
            summary = run_part1(config, limit=args.limit)
            print(f"\nWinner: {summary['overall_winner']}")

        elif part_num == 2:
            from part2_noise_estimation import run_part2
            summary = run_part2(config, limit=args.limit)
            print(f"\nGaussian sigma (p5-p95): {summary['gaussian_noise']['sigma_range_p5_p95']}")
            print(f"Speckle sigma (p5-p95): {summary['speckle_noise']['sigma_range_p5_p95']}")

        elif part_num == 3:
            from part3_spectral_analysis import run_part3
            summary = run_part3(config, limit=args.limit)
            for label, info in summary["spectral_proportions"].items():
                print(f"  {label}: {info['pct']:.1f}%")

        elif part_num == 4:
            from part4_consolidate import consolidate
            report, md_path = consolidate(config)
            print(f"\nReport saved to {md_path}")

        elapsed = time.time() - t0
        print(f"--- Part {part_num}: Done in {elapsed:.1f}s ---")

    total_elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"Pipeline complete in {total_elapsed:.1f}s")
    print(f"Results in: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
