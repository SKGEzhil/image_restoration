"""Classify GT samples separately for noise and blur (exp_02).

Each dimension is evaluated independently:
  Noise: noise_est > threshold -> flagged
  Blur:  lap_var < threshold -> flagged

Run: python src/data_processing/exp_02/classify.py
      python src/data_processing/exp_02/classify.py --mode manual
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).parent
FILTERING_DIR = EXP_DIR.parent
CONFIG_PATH = EXP_DIR / "config.yaml"
OUTPUT_DIR = EXP_DIR / "outputs"


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


def classify_noise(rows, threshold):
    """Classify each sample for noise: noise_est >= threshold -> flagged."""
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
    """Classify each sample for blur: lap_var <= threshold -> flagged."""
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
    parser.add_argument("--mode", choices=["auto", "manual"], default="manual")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Import shared threshold helpers
    sys.path.insert(0, str(FILTERING_DIR))
    from thresholds import auto_threshold

    rows = load_metrics()
    print(f"Loaded {len(rows)} samples")

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
    noise_dir = OUTPUT_DIR / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
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
    blur_dir = OUTPUT_DIR / "blur"
    blur_dir.mkdir(parents=True, exist_ok=True)
    with open(blur_dir / "classifications.json", "w") as f:
        json.dump(blur_out, f, indent=2)
    print(f"  Saved {blur_dir / 'classifications.json'}")

    print(f"\n--- Thresholds ---")
    print(f"  noise_threshold: {noise_thr:.8f}")
    print(f"  blur_threshold:  {blur_thr:.8f}")

    if args.mode == "auto":
        print(f"\n[auto mode] Thresholds NOT saved to config. To use them, add to config.yaml:")
        print(f"  noise_threshold: {float(noise_thr)}")
        print(f"  blur_threshold: {float(blur_thr)}")
    else:
        print(f"\nUsing thresholds from {CONFIG_PATH}")


if __name__ == "__main__":
    main()
