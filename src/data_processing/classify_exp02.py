"""Classify GT samples separately for noise and blur (exp_02).

Each dimension is evaluated independently:
  Noise: noise_est > threshold → flagged
  Blur:  lap_var < threshold → flagged

Run: python src/data_processing/classify_exp02.py
      python src/data_processing/classify_exp02.py --mode manual
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

EXP02_DIR = Path(__file__).parent / "exp_02"
CONFIG_PATH = EXP02_DIR / "config.yaml"


def load_metrics():
    cfg = yaml.safe_load(open(CONFIG_PATH))
    csv_path = EXP02_DIR / cfg["metrics_csv"]
    rows = []
    with open(csv_path) as f:
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


def find_natural_gap(values, n_bins=50):
    """Find the largest gap in a log-transformed histogram."""
    values = np.array(values)
    log_vals = np.log10(values + 1e-10)
    hist, bin_edges = np.histogram(log_vals, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    nonzero = np.where(hist > 0)[0]
    if len(nonzero) < 2:
        return float(np.median(values)), 0.0, False

    max_gap = 0
    max_gap_pos = 0
    for i in range(len(nonzero) - 1):
        gap = nonzero[i + 1] - nonzero[i]
        if gap > max_gap:
            max_gap = gap
            max_gap_pos = (nonzero[i] + nonzero[i + 1]) // 2

    gap_width_frac = max_gap / len(hist)
    gap_center_log = float(bin_centers[max_gap_pos])
    gap_center = float(10 ** gap_center_log)

    midpoint_log = float(np.median(log_vals))
    at_edge = abs(gap_center_log - midpoint_log) > 1.5 * float(np.std(log_vals))
    has_gap = gap_width_frac > 0.1 and not at_edge

    return gap_center, float(gap_width_frac), has_gap


def auto_threshold(values, direction):
    """Find threshold from distribution gap, fallback to percentile."""
    gap_center, gap_frac, has_gap = find_natural_gap(values)

    if has_gap:
        return gap_center, {"threshold": gap_center, "method": "natural_gap",
                            "gap_width": round(gap_frac, 3)}

    if direction == "upper":
        thr = np.percentile(values, 85)
        method = "percentile_p85"
    else:
        thr = np.percentile(values, 15)
        method = "percentile_p15"

    return thr, {"threshold": thr, "method": method,
                 "warning": "No natural gap — distribution is continuous"}


def classify_noise(rows, threshold):
    """Classify each sample for noise: noise_est >= threshold → flagged."""
    results = []
    flagged = 0
    for r in rows:
        is_noisy = r["noise_est"] >= threshold
        entry = dict(r)
        entry["flagged"] = bool(is_noisy)
        entry["severity"] = float(r["noise_est"])
        results.append(entry)
        if is_noisy:
            flagged += 1
    return results, flagged


def classify_blur(rows, threshold):
    """Classify each sample for blur: lap_var <= threshold → flagged."""
    results = []
    flagged = 0
    for r in rows:
        is_blurry = r["lap_var"] <= threshold
        entry = dict(r)
        entry["flagged"] = bool(is_blurry)
        entry["severity"] = float(r["lap_var"])
        results.append(entry)
        if is_blurry:
            flagged += 1
    return results, flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    rows = load_metrics()
    print(f"Loaded {len(rows)} samples from {EXP02_DIR / cfg['metrics_csv']}")

    # --- Noise classification ---
    print("\n=== Noise Dimension ===")
    noise_ests = [r["noise_est"] for r in rows]
    if args.mode == "auto" or cfg.get("noise_threshold") is None:
        noise_thr, noise_analysis = auto_threshold(noise_ests, direction="upper")
        print(f"  Auto threshold: {noise_thr:.8f}")
    else:
        noise_thr = cfg["noise_threshold"]
        noise_analysis = {"threshold": noise_thr, "method": "manual"}
        print(f"  Manual threshold: {noise_thr:.8f}")

    noise_results, noise_flagged = classify_noise(rows, noise_thr)
    print(f"  Flagged: {noise_flagged} / {len(rows)} ({noise_flagged / len(rows) * 100:.1f}%)")

    noise_out = {
        "dimension": "noise",
        "threshold": float(noise_thr),
        "analysis": noise_analysis,
        "summary": {"total": len(rows), "flagged": noise_flagged,
                     "pct": round(noise_flagged / len(rows) * 100, 1)},
        "samples": noise_results,
    }
    noise_dir = EXP02_DIR / "noise"
    noise_dir.mkdir(exist_ok=True)
    with open(noise_dir / "classifications.json", "w") as f:
        json.dump(noise_out, f, indent=2)
    print(f"  Saved {noise_dir / 'classifications.json'}")

    # --- Blur classification ---
    print("\n=== Blur Dimension ===")
    lap_vars = [r["lap_var"] for r in rows]
    if args.mode == "auto" or cfg.get("blur_threshold") is None:
        blur_thr, blur_analysis = auto_threshold(lap_vars, direction="lower")
        print(f"  Auto threshold: {blur_thr:.8f}")
    else:
        blur_thr = cfg["blur_threshold"]
        blur_analysis = {"threshold": blur_thr, "method": "manual"}
        print(f"  Manual threshold: {blur_thr:.8f}")

    blur_results, blur_flagged = classify_blur(rows, blur_thr)
    print(f"  Flagged: {blur_flagged} / {len(rows)} ({blur_flagged / len(rows) * 100:.1f}%)")

    blur_out = {
        "dimension": "blur",
        "threshold": float(blur_thr),
        "analysis": blur_analysis,
        "summary": {"total": len(rows), "flagged": blur_flagged,
                     "pct": round(blur_flagged / len(rows) * 100, 1)},
        "samples": blur_results,
    }
    blur_dir = EXP02_DIR / "blur"
    blur_dir.mkdir(exist_ok=True)
    with open(blur_dir / "classifications.json", "w") as f:
        json.dump(blur_out, f, indent=2)
    print(f"  Saved {blur_dir / 'classifications.json'}")

    # Update config with discovered thresholds
    cfg["noise_threshold"] = float(noise_thr)
    cfg["blur_threshold"] = float(blur_thr)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"\nUpdated {CONFIG_PATH}")


if __name__ == "__main__":
    main()
