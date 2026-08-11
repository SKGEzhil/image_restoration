"""Unit tests for GT quality classification logic.

Creates synthetic test patches and validates that the classification
inequalities produce correct Mode A / Mode B labels.

Run: python src/data_processing/unit_test_classification.py
"""

import sys
from pathlib import Path

import numpy as np

# allow import from same directory
sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Synthetic patch generators
# ---------------------------------------------------------------------------

def make_clean_sharp(size=256):
    """Random texture with strong edges — should be Mode A."""
    rng = np.random.RandomState(0)
    # checkerboard + noise for high-frequency content
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    checker = ((x // 16) + (y // 16)) % 2
    img = checker * 0.6 + rng.rand(size, size) * 0.1
    return np.clip(img, 0, 1).astype(np.float32)


def make_blurred(size=256):
    """Gaussian-blurred version — should be Mode B (blurry)."""
    sharp = make_clean_sharp(size)
    # box blur via uniform filter (simple, no scipy needed)
    from numpy.lib.stride_tricks import sliding_window_view
    w = 12
    if size >= w:
        padded = np.pad(sharp, w // 2, mode="reflect")
        windows = sliding_window_view(padded, (w, w))
        blurred = np.mean(windows, axis=(-2, -1))
    else:
        blurred = sharp
    return blurred.astype(np.float32)


def make_noisy(size=256):
    """Clean image with added Gaussian noise — should be Mode B (noisy)."""
    clean = make_clean_sharp(size)
    rng = np.random.RandomState(1)
    noisy = clean + rng.randn(size, size).astype(np.float32) * 0.15
    return np.clip(noisy, 0, 1).astype(np.float32)


def make_flat_clean(size=256):
    """Uniform region with slight gradient — should be Mode A (flat)."""
    x = np.linspace(0.45, 0.55, size)
    img = np.tile(x, (size, 1)).astype(np.float32)
    return img


def make_flat_noisy(size=256):
    """Uniform region with added noise — should be Mode B (flat + noisy)."""
    flat = make_flat_clean(size)
    rng = np.random.RandomState(2)
    noisy = flat + rng.randn(size, size).astype(np.float32) * 0.12
    return np.clip(noisy, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Classification logic (mirrors classify_gt.py)
# ---------------------------------------------------------------------------

def classify(lap_var, noise_est, local_var, thresholds):
    """Classify a sample as Mode A or Mode B.

    thresholds: dict with keys lap_var_blur, noise_est_noisy, local_var_flat
    Returns: (mode, reasons)
    """
    lap_thr = thresholds["lap_var_blur"]
    noise_thr = thresholds["noise_est_noisy"]
    local_thr = thresholds["local_var_flat"]

    is_flat = local_var < local_thr
    is_blurry = lap_var < lap_thr
    is_noisy = noise_est > noise_thr

    reasons = []
    if is_flat:
        if is_noisy:
            return "B", ["noisy_flat"]
        else:
            return "A", []
    else:
        if is_blurry:
            reasons.append("blurry")
        if is_noisy:
            reasons.append("noisy")
        if reasons:
            return "B", reasons
        else:
            return "A", []


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Unit Tests: GT Quality Classification Logic")
    print("=" * 70)

    # Generate synthetic patches
    patches = {
        "clean_sharp": make_clean_sharp(),
        "blurred": make_blurred(),
        "noisy": make_noisy(),
        "flat_clean": make_flat_clean(),
        "flat_noisy": make_flat_noisy(),
    }

    # Compute metrics for each patch
    print("\n--- Computed Metrics ---")
    metrics = {}
    for name, img in patches.items():
        m = compute_all_metrics(img)
        metrics[name] = m
        print(f"  {name:15s}  lap_var={m['lap_var']:.6f}  fft_hf={m['fft_hf_ratio']:.4f}  "
              f"noise_est={m['noise_est']:.6f}  local_var={m['local_var']:.6f}")

    # Derive thresholds from the metrics themselves (use midpoints for testing)
    lap_vals = [m["lap_var"] for m in metrics.values()]
    noise_vals = [m["noise_est"] for m in metrics.values()]
    local_vals = [m["local_var"] for m in metrics.values()]

    # For unit tests, use generous thresholds that should clearly separate
    # the synthetic cases. These are NOT the real-data thresholds.
    test_thresholds = {
        "lap_var_blur": np.median(lap_vals),       # midpoints
        "noise_est_noisy": np.median(noise_vals),
        "local_var_flat": np.median(local_vals),
    }

    print(f"\n--- Test Thresholds (derived from medians) ---")
    print(f"  lap_var_blur:      {test_thresholds['lap_var_blur']:.6f}")
    print(f"  noise_est_noisy:   {test_thresholds['noise_est_noisy']:.6f}")
    print(f"  local_var_flat:    {test_thresholds['local_var_flat']:.6f}")

    # Expected results
    expected = {
        "clean_sharp": ("A", "clean sharp — valid textured GT"),
        "blurred": ("B", "blurry — degraded GT"),
        "noisy": ("B", "noisy — degraded GT"),
        "flat_clean": ("A", "flat clean — valid flat GT"),
        "flat_noisy": ("B", "flat noisy — degraded GT"),
    }

    # Run classification
    print(f"\n--- Classification Results ---")
    all_passed = True
    for name, (exp_mode, exp_reason) in expected.items():
        m = metrics[name]
        mode, reasons = classify(m["lap_var"], m["noise_est"], m["local_var"], test_thresholds)
        passed = mode == exp_mode
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {name:15s}  expected={exp_mode}  got={mode}  {status}  ({exp_reason})")

    print()
    if all_passed:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED — check threshold logic or metric computation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
