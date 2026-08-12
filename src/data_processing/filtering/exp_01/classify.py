"""Classify GT samples as Mode A (valid) or Mode B (degraded).

Reads metrics.csv, applies thresholds from config.yaml,
and outputs classification results.

Threshold modes:
- Auto: analyzes metric distributions for natural gaps, picks thresholds there
- Manual: uses thresholds set in config.yaml

Run: python src/data_processing/exp_01/classify.py
      python src/data_processing/exp_01/classify.py --mode auto
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
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


def classify_sample(lap_var, noise_est, local_var, thresholds):
    """Classify a single sample.

    Inequalities:
        lap_var < thresholds["lap_var_blur"]  -> blurry
        noise_est > thresholds["noise_est_noisy"] -> noisy
        local_var < thresholds["local_var_flat"] -> flat

    Returns: (mode, reasons)
    """
    is_flat = local_var < thresholds["local_var_flat"]
    is_blurry = lap_var < thresholds["lap_var_blur"]
    is_noisy = noise_est > thresholds["noise_est_noisy"]

    if is_flat:
        if is_noisy:
            return "B", ["noisy_flat"]
        else:
            return "A", []
    else:
        reasons = []
        if is_blurry:
            reasons.append("blurry")
        if is_noisy:
            reasons.append("noisy")
        if reasons:
            return "B", reasons
        else:
            return "A", []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "manual"], default="manual",
                        help="auto: find thresholds from distributions; manual: use config.yaml values")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    rows = load_metrics()
    print(f"Loaded {len(rows)} samples")

    # Import shared threshold helpers
    sys.path.insert(0, str(FILTERING_DIR))
    from thresholds import auto_threshold

    if args.mode == "auto":
        print("\n--- Auto-detecting thresholds from distributions ---")
        thresholds = {}
        analyses = {}

        for name, vals, direction in [
            ("lap_var_blur", [r["lap_var"] for r in rows], "lower"),
            ("noise_est_noisy", [r["noise_est"] for r in rows], "upper"),
            ("local_var_flat", [r["local_var"] for r in rows], "lower"),
        ]:
            thr, analysis = auto_threshold(vals, direction)
            thresholds[name] = thr
            analyses[name] = analysis
            print(f"  {name}: {analysis['method']} -> {thr:.8f}")

        print(f"\n--- Thresholds ---")
        print(f"  lap_var_blur:     {thresholds['lap_var_blur']:.8f}  (below = blurry)")
        print(f"  noise_est_noisy:  {thresholds['noise_est_noisy']:.8f}  (above = noisy)")
        print(f"  local_var_flat:   {thresholds['local_var_flat']:.8f}  (below = flat)")
        print(f"\n[auto mode] Thresholds NOT saved to config. To use them, add to config.yaml:")
        print(f"  lap_var_blur_threshold: {float(thresholds['lap_var_blur'])}")
        print(f"  noise_est_noisy_threshold: {float(thresholds['noise_est_noisy'])}")
        print(f"  local_var_flat_threshold: {float(thresholds['local_var_flat'])}")
    else:
        thresholds = {
            "lap_var_blur": cfg["lap_var_blur_threshold"],
            "noise_est_noisy": cfg["noise_est_noisy_threshold"],
            "local_var_flat": cfg["local_var_flat_threshold"],
        }
        if any(v is None for v in thresholds.values()):
            print("ERROR: Some thresholds are null in config.yaml. "
                  "Run with --mode auto first, or set them manually.")
            sys.exit(1)
        analyses = {k: {"threshold": v, "method": "manual"} for k, v in thresholds.items()}
        print(f"\nUsing manual thresholds from config.yaml")

    print(f"\n--- Thresholds ---")
    print(f"  lap_var_blur:     {thresholds['lap_var_blur']:.8f}  (below = blurry)")
    print(f"  noise_est_noisy:  {thresholds['noise_est_noisy']:.8f}  (above = noisy)")
    print(f"  local_var_flat:   {thresholds['local_var_flat']:.8f}  (below = flat)")

    # Classify all samples
    results = []
    mode_a = 0
    mode_b = 0
    for r in rows:
        mode, reasons = classify_sample(
            r["lap_var"], r["noise_est"], r["local_var"], thresholds
        )
        r["mode"] = mode
        r["reasons"] = reasons
        results.append(r)
        if mode == "A":
            mode_a += 1
        else:
            mode_b += 1

    print(f"\n--- Classification Results ---")
    print(f"  Mode A (valid):   {mode_a} ({mode_a/len(results)*100:.1f}%)")
    print(f"  Mode B (degraded): {mode_b} ({mode_b/len(results)*100:.1f}%)")

    # Breakdown of Mode B reasons
    reason_counts = {}
    for r in results:
        if r["mode"] == "B":
            for reason in r["reasons"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if reason_counts:
        print(f"\n--- Mode B Breakdown ---")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason:15s}  {count}")

    # Save classification output
    output = {
        "thresholds": thresholds,
        "analyses": analyses,
        "summary": {
            "total": len(results),
            "mode_a": mode_a,
            "mode_b": mode_b,
            "mode_a_pct": round(mode_a / len(results) * 100, 1),
            "mode_b_pct": round(mode_b / len(results) * 100, 1),
            "reason_counts": reason_counts,
        },
        "samples": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "gt_classifications.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved classifications to {out_path}")


if __name__ == "__main__":
    main()
