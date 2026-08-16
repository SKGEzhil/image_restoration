"""Evaluation metrics for test.py and validation.

These are separate from training losses (src/losses.py) and used only
for reporting/evaluation, not for gradient computation.
"""

import torch
from torchmetrics.image import PeakSignalNoiseRatio
import kornia
from dists_module import DISTS


class _DeviceMetrics:
    """Caches PSNR/DISTS metric modules, moving them to the input device lazily."""

    def __init__(self):
        self._psnr = None
        self._dists = None
        self._device = None

    def _get(self, device):
        if self._psnr is None or self._device != device:
            self._psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
            self._dists = DISTS(pretrained=True).to(device)
            self._dists.eval()
            self._device = device
        return self._psnr, self._dists


_metrics = _DeviceMetrics()


def compute_ssim(pred, gt):
    """pred/gt: 1-channel float tensors in [0, 1]. Returns mean SSIM."""
    kornia_ssim = kornia.losses.SSIMLoss(window_size=11, max_val=1.0)
    loss = 2 * kornia_ssim(pred.clamp(0.0, 1.0), gt)
    return 1.0 - loss


def compute_psnr(pred, gt):
    psnr, _ = _metrics._get(pred.device)
    return psnr(pred.clamp(0.0, 1.0), gt)


def compute_dists(pred, gt):
    """pred/gt: 1-channel float tensors in [0, 1]. Returns DISTS score (lower = more similar)."""
    _, dists_metric = _metrics._get(pred.device)
    return dists_metric(pred.clamp(0.0, 1.0), gt)
