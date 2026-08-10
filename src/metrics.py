"""Shared loss + metrics so train.py and test.py measure identically."""

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class _DeviceMetrics:
    """Caches SSIM/PSNR metric modules, moving them to the input device lazily."""

    def __init__(self):
        self._ssim = None
        self._psnr = None
        self._device = None

    def _get(self, device):
        if self._ssim is None or self._device != device:
            self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            self._psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
            self._device = device
        return self._ssim, self._psnr


_metrics = _DeviceMetrics()


def compute_ssim(pred, gt):
    """pred/gt: 1-channel float tensors in [0, 1]. Returns mean SSIM."""
    ssim, _ = _metrics._get(pred.device)
    return ssim(pred.clamp(0.0, 1.0), gt)


def compute_psnr(pred, gt):
    _, psnr = _metrics._get(pred.device)
    return psnr(pred.clamp(0.0, 1.0), gt)


def build_loss(l1_weight=0.5, ssim_weight=0.5):
    """Returns loss(pred, gt) = l1_weight * L1 + ssim_weight * (1 - SSIM)."""

    def loss(pred, gt):
        l1 = torch.nn.functional.l1_loss(pred, gt)
        ssim_loss = 1.0 - compute_ssim(pred, gt)
        return l1_weight * l1 + ssim_weight * ssim_loss

    return loss


def separate_losses(pred, gt):
    """L1 and (1-SSIM) separately, for logging both components."""
    l1 = torch.nn.functional.l1_loss(pred, gt)
    ssim = compute_ssim(pred, gt)
    return l1, 1.0 - ssim, ssim