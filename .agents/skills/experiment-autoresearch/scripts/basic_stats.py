"""Basic statistical analysis for experiment metric series.

Given a series of values, compute slope, convergence, and trend direction.
The agent feeds metric arrays here and uses the results for decisions.
"""

import argparse
import json
from typing import Any

import numpy as np


def compute_slope(values: list[float], fraction: float = 0.2) -> dict:
    """Compute linear regression slope over the last fraction of points.
    Returns {slope, intercept, r_squared, num_points}.
    """
    if len(values) < 3:
        return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0, "num_points": len(values)}

    n = max(3, int(len(values) * fraction))
    recent = values[-n:]
    x = np.arange(len(recent))
    y = np.array(recent)

    if np.std(y) < 1e-9:
        return {"slope": 0.0, "intercept": float(y[0]), "r_squared": 1.0, "num_points": n}

    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    # R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "num_points": n,
    }


def compute_convergence(values: list[float], window: int = 5, threshold: float = 0.001) -> dict:
    """Check if a series has converged (flat in recent window).
    Returns {converged, recent_mean, recent_std, flat_for_n}.
    """
    if len(values) < window * 2:
        return {"converged": False, "recent_mean": None, "recent_std": None, "flat_for_n": 0}

    recent = values[-window:]
    prev = values[-window * 2 : -window]

    recent_mean = float(np.mean(recent))
    recent_std = float(np.std(recent))
    prev_mean = float(np.mean(prev))

    # Check if recent window is flat (std < threshold)
    is_flat = recent_std < threshold

    # Count how many consecutive windows are flat
    flat_for_n = 0
    for i in range(len(values) // window):
        start = -(i + 1) * window
        end = -i * window if i > 0 else None
        w = values[start:end]
        if len(w) == window and np.std(w) < threshold:
            flat_for_n += 1
        else:
            break

    return {
        "converged": is_flat,
        "recent_mean": recent_mean,
        "recent_std": recent_std,
        "flat_for_n": flat_for_n,
        "change_from_prev_window": recent_mean - prev_mean,
    }


def compute_trend(values: list[float]) -> str:
    """Simple trend classification."""
    if len(values) < 3:
        return "insufficient_data"

    slope_result = compute_slope(values, fraction=0.3)
    slope = slope_result["slope"]

    # Normalize by mean to get relative slope
    mean_val = np.mean(values)
    rel_slope = slope / abs(mean_val) if abs(mean_val) > 1e-9 else slope

    if rel_slope > 0.01:
        return "strongly_improving"
    elif rel_slope > 0.001:
        return "improving"
    elif rel_slope > -0.001:
        return "plateaued"
    elif rel_slope > -0.01:
        return "declining"
    else:
        return "strongly_declining"


def compute_instability(values: list[float]) -> dict:
    """Count spikes and oscillations."""
    if len(values) < 3:
        return {"spike_count": 0, "oscillation_score": 0.0, "max_jump": 0.0}

    diffs = np.abs(np.diff(values))
    median_diff = float(np.median(diffs))

    spikes = int(np.sum(diffs > 3 * median_diff)) if median_diff > 1e-9 else 0
    max_jump = float(np.max(diffs))

    # Oscillation: count sign changes in diffs
    sign_changes = int(np.sum((np.diff(values)[:-1] * np.diff(values)[1:]) < 0))
    oscillation_score = sign_changes / len(values)

    return {
        "spike_count": spikes,
        "oscillation_score": oscillation_score,
        "max_jump": max_jump,
    }


def analyze_series(values: list[float], metric_name: str = "metric") -> dict:
    """Full analysis of a single metric series."""
    return {
        "metric_name": metric_name,
        "count": len(values),
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "best": max(values) if values else None,
        "worst": min(values) if values else None,
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values)) if values else None,
        "slope": compute_slope(values),
        "convergence": compute_convergence(values),
        "trend": compute_trend(values),
        "instability": compute_instability(values),
    }


def extract_series_from_metrics(metrics: dict, prefix: str, metric_name: str) -> list[float]:
    """Extract a flat value list from the series format used by extract_metrics.py.
    Series format: [(step, value), (step, value), ...]
    """
    key = f"{prefix}_{metric_name}_series"
    series = metrics.get(key, [])
    return [v for _, v in series]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-file", help="JSON file from extract_metrics.py")
    parser.add_argument("--metric-array", help='JSON array of values, e.g. "[0.1, 0.2, 0.25]"')
    parser.add_argument("--metric-name", default="metric", help="Name for output labeling")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    values = None

    if args.metrics_file:
        data = json.loads(Path(args.metrics_file).read_text())
        # Try to auto-detect a series
        for key in data:
            if key.endswith("_series"):
                series = data[key]
                values = [v for _, v in series]
                args.metric_name = key.replace("_series", "")
                break
        if values is None:
            print(json.dumps({"error": "no _series key found in metrics file"}))
            return

    elif args.metric_array:
        values = json.loads(args.metric_array)

    else:
        print(json.dumps({"error": "provide --metrics-file or --metric-array"}))
        return

    result = analyze_series(values, args.metric_name)

    out = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
