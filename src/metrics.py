"""Evaluation metrics for test.py and validation.

These are separate from training losses (src/losses.py) and used only
for reporting/evaluation, not for gradient computation.
"""

import torch
from torchmetrics.image import PeakSignalNoiseRatio
import kornia
import lpips


class _DeviceMetrics:
    """Caches PSNR/LPIPS metric modules, moving them to the input device lazily."""

    def __init__(self):
        self._psnr = None
        self._lpips = None
        self._device = None

    def _get(self, device):
        if self._psnr is None or self._device != device:
            self._psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
            self._lpips = lpips.LPIPS(net="alex").to(device)
            self._lpips.eval()
            self._device = device
        return self._psnr, self._lpips


_metrics = _DeviceMetrics()


def compute_ssim(pred, gt):
    """pred/gt: 1-channel float tensors in [0, 1]. Returns mean SSIM."""
    kornia_ssim = kornia.losses.SSIMLoss(window_size=11, max_val=1.0)
    loss = 2 * kornia_ssim(pred.clamp(0.0, 1.0), gt)
    return 1.0 - loss


def compute_psnr(pred, gt):
    psnr, _ = _metrics._get(pred.device)
    return psnr(pred.clamp(0.0, 1.0), gt)


def compute_lpips(pred, gt):
    """pred/gt: 1-channel float tensors in [0, 1]. Returns mean LPIPS."""
    _, lpips_metric = _metrics._get(pred.device)
    pred_rgb = pred.clamp(0.0, 1.0).repeat(1, 3, 1, 1) * 2.0 - 1.0
    gt_rgb = gt.clamp(0.0, 1.0).repeat(1, 3, 1, 1) * 2.0 - 1.0
    return lpips_metric(pred_rgb, gt_rgb).mean()
