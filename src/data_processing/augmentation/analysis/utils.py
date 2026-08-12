"""Shared utilities for degradation estimation pipeline."""

from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.ndimage import shift as ndimage_shift
from skimage.registration import phase_cross_correlation


def load_config(config_path: str | Path | None = None) -> dict:
    """Load config.yaml from the analysis directory."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_data_pairs(data_dir: str | Path, split: str) -> list[tuple[Path, Path]]:
    """Return sorted list of (gt_path, lr_path) pairs."""
    data_dir = Path(data_dir)
    gt_dir = data_dir / split / "GT"
    lr_dir = data_dir / split / "NoisyLR"

    gt_files = sorted(p.name for p in gt_dir.glob("*.npy"))
    lr_files = sorted(p.name for p in lr_dir.glob("*.npy"))
    assert gt_files == lr_files, f"{split}: GT/NoisyLR filenames mismatch"

    return [(gt_dir / name, lr_dir / name) for name in gt_files]


def load_pair(gt_path: Path, lr_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a GT/NoisyLR pair as float64 arrays."""
    gt = np.load(gt_path).astype(np.float64)
    lr = np.load(lr_path).astype(np.float64)
    return gt, lr


def downsample_candidate(
    gt: np.ndarray,
    kernel: str,
    sigma: float | None = None,
    factor: int = 2,
) -> np.ndarray:
    """Generate a clean LR candidate from GT using specified kernel.

    Kernels:
        - 'bicubic': cv2.INTER_CUBIC
        - 'bilinear': cv2.INTER_LINEAR
        - 'gaussian': GaussianBlur + decimate
    """
    h, w = gt.shape
    new_h, new_w = h // factor, w // factor

    if kernel == "bicubic":
        return cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif kernel == "bilinear":
        return cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    elif kernel == "gaussian":
        if sigma is None:
            raise ValueError("sigma required for gaussian kernel")
        ksize = int(np.ceil(sigma * 3)) * 2 + 1
        blurred = cv2.GaussianBlur(gt, (ksize, ksize), sigma)
        return blurred[::factor, ::factor]
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def align_images(
    reference: np.ndarray,
    target: np.ndarray,
    upsample_factor: int = 20,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Align target to reference using sub-pixel phase correlation.

    Returns (aligned_target, (shift_y, shift_x)).
    """
    shift_vec, _, _ = phase_cross_correlation(
        reference, target, upsample_factor=upsample_factor
    )
    aligned = ndimage_shift(target, shift_vec, order=1, mode="constant")
    return aligned, tuple(shift_vec)


def compute_flat_mask(
    candidate: np.ndarray,
    window: int = 7,
    percentile: float = 20.0,
) -> np.ndarray:
    """Identify flat/smooth regions via local variance.

    Returns boolean mask: True where regions are flat.
    """
    h, w = candidate.shape
    pad = window // 2
    padded = np.pad(candidate, pad, mode="reflect")

    # Compute local variance using sliding window
    local_var = np.zeros_like(candidate)
    for di in range(window):
        for dj in range(window):
            window_slice = padded[di : di + h, dj : dj + w]
            local_var += window_slice ** 2
    local_var /= window ** 2
    local_mean = np.zeros_like(candidate)
    for di in range(window):
        for dj in range(window):
            window_slice = padded[di : di + h, dj : dj + w]
            local_mean += window_slice
    local_mean /= window ** 2
    variance = local_var - local_mean ** 2

    threshold = np.percentile(variance, percentile)
    return variance <= threshold


def radial_profile(data: np.ndarray) -> np.ndarray:
    """Compute radial average of a 2D spectrum.

    Returns 1D array of energy vs. spatial frequency (normalized 0..1).
    """
    h, w = data.shape
    cy, cx = h // 2, w // 2

    # Distance from center (normalized)
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)

    # Bin by integer radius
    max_r = int(min(cy, cx))
    r_int = np.round(r * max_r).astype(int)
    r_int = np.clip(r_int, 0, max_r)

    profile = np.zeros(max_r + 1, dtype=np.float64)
    counts = np.zeros(max_r + 1, dtype=np.float64)
    np.add.at(profile, r_int, data)
    np.add.at(counts, r_int, 1)

    valid = counts > 0
    profile[valid] /= counts[valid]
    return profile


def ensure_output_dir(config: dict) -> Path:
    """Create and return the output directory."""
    analysis_dir = Path(__file__).parent
    output_dir = analysis_dir / config.get("output_dir", "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
