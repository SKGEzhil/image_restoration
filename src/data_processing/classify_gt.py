"""Classify GT samples as Mode A (valid) or Mode B (degraded).

Reads gt_quality_metrics.csv, applies thresholds from config.yaml,
and outputs classification results.

Threshold modes:
- Auto: analyzes metric distributions for natural gaps, picks thresholds there
- Manual: uses thresholds set in config.yaml

Run: python src/data_processing/classify_gt.py
      python src/data_processing/classify_gt.py --mode auto
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
METRICS_CSV = Path(__file__).parent / ".." / "data" / "train" / "gt_quality_metrics.csv"
OUTPUT_JSON = Path(__file__).parent / ".." / "data" / "train" / "gt_classifications.json"


def load_metrics():
    rows = []
    with open(METRICS_CSV) as f:
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
    """Find the largest gap in a histogram's bin edges.

    Returns (gap_center, gap_width, has_gap) where has_gap=False
    means the distribution is smooth with no clear separation.
    """
    values = np.array(values)

    # Log-transform for right-skewed distributions to spread out the bulk
    # (add small constant to avoid log(0))
    log_vals = np.log10(values + 1e-10)
    hist, bin_edges = np.histogram(log_vals, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find largest gap between consecutive non-zero bins
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

    # Convert back from log space
    gap_center_log = float(bin_centers[max_gap_pos])
    gap_center = float(10 ** gap_center_log)

    # Only consider meaningful gaps: >10% of range AND not at extreme edges
    # (gaps at the very tail are just outlier boundaries)
    midpoint_log = float(np.median(log_vals))
    at_edge = abs(gap_center_log - midpoint_log) > 1.5 * float(np.std(log_vals))
    has_gap = gap_width_frac > 0.1 and not at_edge

    return gap_center, float(gap_width_frac), has_gap


def auto_thresholds(rows):
    """Analyze distributions and pick thresholds at natural gaps."""
    lap_vars = [r["lap_var"] for r in rows]
    noise_ests = [r["noise_est"] for r in rows]
    local_vars = [r["local_var"] for r in rows]

    thresholds = {}
    analyses = {}

    for name, vals, direction in [
        ("lap_var_blur", lap_vars, "lower"),      # lower = blurrier
        ("noise_est_noisy", noise_ests, "upper"),   # upper = noisier
        ("local_var_flat", local_vars, "lower"),    # lower = flatter
    ]:
        gap_center, gap_frac, has_gap = find_natural_gap(vals)

        if has_gap:
            thresholds[name] = gap_center
            analyses[name] = {
                "threshold": gap_center,
                "gap_width_fraction": round(gap_frac, 3),
                "method": "natural_gap",
            }
            print(f"  {name}: natural gap at {gap_center:.6f} (gap spans {gap_frac:.1%} of range)")
        else:
            # No natural gap — use percentile-based fallback but warn
            if direction == "lower":
                thr = np.percentile(vals, 15)
                analyses[name] = {
                    "threshold": thr,
                    "method": "percentile_fallback_p15",
                    "warning": "No natural gap found. Distribution is continuous. "
                               "Consider soft loss-weighting instead of hard exclusion.",
                }
            else:
                thr = np.percentile(vals, 85)
                analyses[name] = {
                    "threshold": thr,
                    "method": "percentile_fallback_p85",
                    "warning": "No natural gap found. Distribution is continuous. "
                               "Consider soft loss-weighting instead of hard exclusion.",
                }
            thresholds[name] = thr
            print(f"  {name}: NO natural gap — fallback to percentile threshold {thr:.6f}")

    return thresholds, analyses


def classify_sample(lap_var, noise_est, local_var, thresholds):
    """Classify a single sample.

    Inequalities (explicit):
        lap_var < thresholds["lap_var_blur"]  → blurry
        noise_est > thresholds["noise_est_noisy"] → noisy
        local_var < thresholds["local_var_flat"] → flat

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
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto",
                        help="auto: find thresholds from distributions; manual: use config.yaml values")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    rows = load_metrics()
    print(f"Loaded {len(rows)} samples")

    if args.mode == "auto":
        print("\n--- Auto-detecting thresholds from distributions ---")
        thresholds, analyses = auto_thresholds(rows)
        # Update config.yaml with discovered thresholds (convert numpy to native Python)
        cfg["lap_var_blur_threshold"] = float(thresholds["lap_var_blur"])
        cfg["noise_est_noisy_threshold"] = float(thresholds["noise_est_noisy"])
        cfg["local_var_flat_threshold"] = float(thresholds["local_var_flat"])
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"\nUpdated {CONFIG_PATH} with auto-detected thresholds")
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
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved classifications to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
