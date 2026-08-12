"""Compute GT quality metrics for all training samples.

Computes lap_var, fft_hf_ratio, noise_est, and local_var for every
GT image, saves to metrics.csv in the filtering folder.

Run: python src/data_processing/filtering/compute_gt_metrics.py
      python src/data_processing/filtering/compute_gt_metrics.py --gt-dir /path/to/GT
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

FILTERING_DIR = Path(__file__).parent
DEFAULT_GT_DIR = FILTERING_DIR / ".." / ".." / "data" / "train" / "GT"
OUTPUT_CSV = FILTERING_DIR / "metrics.csv"

sys.path.insert(0, str(FILTERING_DIR))
from metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR,
                        help="Path to GT images directory (default: src/data/train/GT)")
    parser.add_argument("--hf-cutoff", type=float, default=0.5,
                        help="FFT high-frequency cutoff (default: 0.5)")
    parser.add_argument("--local-var-window", type=int, default=16,
                        help="Local variance window size (default: 16)")
    args = parser.parse_args()

    gt_dir = args.gt_dir.resolve()
    gt_files = sorted(gt_dir.glob("*.npy"))
    print(f"Found {len(gt_files)} GT samples in {gt_dir}")

    results = []
    t0 = time.time()
    for path in tqdm(gt_files, desc="Computing metrics"):
        img = np.load(path)
        m = compute_all_metrics(img, hf_cutoff=args.hf_cutoff,
                                local_var_window=args.local_var_window)
        m["filename"] = path.name
        results.append(m)

    elapsed = time.time() - t0
    print(f"\nComputed metrics for {len(results)} samples in {elapsed:.1f}s")

    header = "filename,lap_var,fft_hf_ratio,noise_est,local_var"
    with open(OUTPUT_CSV, "w") as f:
        f.write(header + "\n")
        for r in results:
            f.write(f"{r['filename']},{r['lap_var']:.8f},{r['fft_hf_ratio']:.6f},"
                    f"{r['noise_est']:.8f},{r['local_var']:.8f}\n")
    print(f"Saved to {OUTPUT_CSV}")

    # Summary statistics
    lap_vars = [r["lap_var"] for r in results]
    noise_ests = [r["noise_est"] for r in results]
    local_vars = [r["local_var"] for r in results]
    fft_hfs = [r["fft_hf_ratio"] for r in results]

    print(f"\n--- Summary Statistics ---")
    for name, vals in [("lap_var", lap_vars), ("fft_hf_ratio", fft_hfs),
                       ("noise_est", noise_ests), ("local_var", local_vars)]:
        arr = np.array(vals)
        print(f"  {name:15s}  min={arr.min():.6f}  median={np.median(arr):.6f}  "
              f"mean={arr.mean():.6f}  max={arr.max():.6f}  std={arr.std():.6f}")
        for p in [1, 5, 10, 25, 75, 90, 95, 99]:
            print(f"{'':18s}  p{p:02d}={np.percentile(arr, p):.6f}")


if __name__ == "__main__":
    main()
