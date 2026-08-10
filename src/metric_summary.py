"""Summarize per-sample SSIM, PSNR, and L1 from test_details.json."""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_BUCKETS = {
    "SSIM": [0.9, 0.8, 0.7],
    "PSNR": [40.0, 35.0, 30.0, 25.0],
    "L1": [0.01, 0.02, 0.05, 0.1],
}


def resolve_details_file(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "test_details.json"
    if not p.exists():
        raise FileNotFoundError(f"Could not find details file: {p}")
    return p


def load_metrics(details_file: Path):
    details = json.loads(details_file.read_text())
    per_sample = details.get("per_sample", [])
    if not per_sample:
        raise ValueError(f"No per_sample entries found in {details_file}")
    return {
        "SSIM": np.array([float(row["SSIM"]) for row in per_sample], dtype=np.float64),
        "PSNR": np.array([float(row["PSNR"]) for row in per_sample], dtype=np.float64),
        "L1": np.array([float(row["L1"]) for row in per_sample], dtype=np.float64),
    }


def metric_summary(values: np.ndarray):
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def bucket_counts(values: np.ndarray, thresholds, metric_name: str):
    thresholds = list(thresholds)

    if metric_name == "SSIM":
        labels = [
            (">= 0.90", values >= thresholds[0]),
            ("0.80 - 0.90", (values >= thresholds[1]) & (values < thresholds[0])),
            ("0.70 - 0.80", (values >= thresholds[2]) & (values < thresholds[1])),
            ("< 0.70", values < thresholds[2]),
        ]
        return labels

    if metric_name == "PSNR":
        labels = []
        upper = None
        for low in thresholds:
            if upper is None:
                labels.append((f">= {low:g} dB", values >= low))
            else:
                labels.append((f"{low:g} - {upper:g} dB", (values >= low) & (values < upper)))
            upper = low
        labels.append((f"< {thresholds[-1]:g} dB", values < thresholds[-1]))
        return labels

    if metric_name == "L1":
        labels = []
        upper = None
        for low in thresholds:
            if upper is None:
                labels.append((f"< {low:g}", values < low))
            else:
                labels.append((f"{upper:g} - {low:g}", (values >= upper) & (values < low)))
            upper = low
        labels.append((f">= {thresholds[-1]:g}", values >= thresholds[-1]))
        return labels

    raise ValueError(f"Unsupported metric: {metric_name}")


def print_report(metrics, buckets):
    total = next(iter(metrics.values())).size
    print(f"samples: {total}")
    print()

    for name in ("SSIM", "PSNR", "L1"):
        values = metrics[name]
        summary = metric_summary(values)
        print(f"{name}:")
        print(
            f"  mean={summary['mean']:.6f}  median={summary['median']:.6f}  "
            f"std={summary['std']:.6f}  min={summary['min']:.6f}  "
            f"p25={summary['p25']:.6f}  p75={summary['p75']:.6f}  max={summary['max']:.6f}"
        )
        for label, mask in bucket_counts(values, buckets[name], name):
            count = int(mask.sum())
            pct = 100.0 * count / total
            print(f"  {label:<16} {count:>6} ({pct:5.1f}%)")
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize SSIM, PSNR, and L1 distributions from test_details.json"
    )
    parser.add_argument(
        "path",
        help="Path to test_details.json or a run directory containing it",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    details_file = resolve_details_file(args.path)
    metrics = load_metrics(details_file)
    print(f"details_file: {details_file}")
    print_report(metrics, DEFAULT_BUCKETS)


if __name__ == "__main__":
    main()
