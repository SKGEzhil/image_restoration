"""Threshold detection helpers for GT quality classification."""

import numpy as np


def find_natural_gap(values, n_bins=50):
    """Find the largest gap in a log-transformed histogram.

    Returns (gap_center, gap_width_fraction, has_gap) where has_gap=False
    means the distribution is smooth with no clear separation.
    """
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
    """Find threshold from distribution gap, fallback to percentile.

    direction: "upper" (high values = flagged, e.g. noise)
               "lower" (low values = flagged, e.g. blur)
    Returns: (threshold, analysis_dict)
    """
    gap_center, gap_frac, has_gap = find_natural_gap(values)

    if has_gap:
        return gap_center, {
            "threshold": gap_center,
            "method": "natural_gap",
            "gap_width": round(gap_frac, 3),
        }

    if direction == "upper":
        thr = np.percentile(values, 85)
        method = "percentile_p85"
    else:
        thr = np.percentile(values, 15)
        method = "percentile_p15"

    return thr, {
        "threshold": thr,
        "method": method,
        "warning": "No natural gap -- distribution is continuous",
    }
