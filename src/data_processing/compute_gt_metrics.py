"""Compute GT quality metrics for all training samples.

Reads config.yaml for parameters, computes lap_var, fft_hf_ratio,
noise_est, and local_var for every GT image, saves to CSV.

Run: python src/data_processing/compute_gt_metrics.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_all_metrics

CONFIG_PATH = Path(__file__).parent / "config.yaml"
OUTPUT_CSV = Path(__file__).parent / ".." / "data" / "train" / "gt_quality_metrics.csv"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    gt_dir = Path(__file__).parent / cfg["data_dir"]
    gt_files = sorted(gt_dir.glob("*.npy"))
    print(f"Found {len(gt_files)} GT samples in {gt_dir}")

    hf_cutoff = cfg.get("fft_hf_ratio_cutoff", 0.5)
    local_var_window = cfg.get("local_var_window", 16)

    results = []
    t0 = time.time()
    for path in tqdm(gt_files, desc="Computing metrics"):
        img = np.load(path)
        m = compute_all_metrics(img, hf_cutoff=hf_cutoff, local_var_window=local_var_window)
        m["filename"] = path.name
        results.append(m)

    elapsed = time.time() - t0
    print(f"\nComputed metrics for {len(results)} samples in {elapsed:.1f}s")

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
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
        # percentiles
        for p in [1, 5, 10, 25, 75, 90, 95, 99]:
            print(f"{'':18s}  p{p:02d}={np.percentile(arr, p):.6f}")


if __name__ == "__main__":
    main()
