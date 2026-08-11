"""Shared metric computation functions for GT quality analysis."""

import numpy as np
from skimage.restoration import estimate_sigma


def laplacian_variance(img: np.ndarray) -> float:
    """Variance of Laplacian response — HIGH = sharp, LOW = blurry.

    Uses the standard 3x3 Laplacian kernel:
        [[ 0,  1,  0],
         [ 1, -4,  1],
         [ 0,  1,  0]]
    """
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    # manual 2D convolution via numpy
    h, w = img.shape
    padded = np.pad(img, 1, mode="reflect")
    lap = np.zeros_like(img, dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            lap += kernel[di, dj] * padded[di : di + h, dj : dj + w]
    return float(np.var(lap))


def fft_highfreq_ratio(img: np.ndarray, hf_cutoff: float = 0.5) -> float:
    """Ratio of high-frequency spectral energy to total energy.

    hf_cutoff: normalized frequency cutoff (0..1, where 1 = Nyquist).
    HIGH = sharp, LOW = blurry.
    """
    f = np.fft.fft2(img)
    mag = np.abs(f)
    total_energy = float(np.sum(mag**2))
    if total_energy == 0:
        return 0.0

    h, w = img.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    hf_mask = dist > hf_cutoff
    hf_energy = float(np.sum(mag[hf_mask] ** 2))
    return hf_energy / total_energy


def noise_estimation(img: np.ndarray) -> float:
    """Wavelet-based noise estimation — HIGH = noisy, LOW = clean."""
    sigma = estimate_sigma(img, channel_axis=None)
    return float(sigma)


def local_variance_mean(img: np.ndarray, window: int = 16) -> float:
    """Mean of local standard deviation over sliding window.

    HIGH = textured, LOW = flat/uniform.
    """
    h, w = img.shape
    if h < window or w < window:
        return float(np.std(img))

    # Use stride_tricks for efficient sliding window
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(img, (window, window))
    # local std for each window position
    local_std = np.std(windows, axis=(-2, -1))
    return float(np.mean(local_std))


def compute_all_metrics(
    img: np.ndarray,
    hf_cutoff: float = 0.5,
    local_var_window: int = 16,
) -> dict:
    """Compute all four quality metrics for a single image."""
    return {
        "lap_var": laplacian_variance(img),
        "fft_hf_ratio": fft_highfreq_ratio(img, hf_cutoff),
        "noise_est": noise_estimation(img),
        "local_var": local_variance_mean(img, local_var_window),
    }
