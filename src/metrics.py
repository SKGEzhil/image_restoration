"""Shared loss + metrics so train.py and test.py measure identically."""

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips


class _DeviceMetrics:
    """Caches SSIM/PSNR/LPIPS metric modules, moving them to the input device lazily."""

    def __init__(self):
        self._ssim = None
        self._psnr = None
        self._lpips = None
        self._device = None

    def _get(self, device):
        if self._ssim is None or self._device != device:
            self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            self._psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
            self._lpips = lpips.LPIPS(net="alex").to(device)
            self._lpips.eval()
            self._device = device
        return self._ssim, self._psnr, self._lpips


_metrics = _DeviceMetrics()


def compute_ssim(pred, gt):
    """pred/gt: 1-channel float tensors in [0, 1]. Returns mean SSIM."""
    ssim, _, _ = _metrics._get(pred.device)
    return ssim(pred.clamp(0.0, 1.0), gt)


def compute_psnr(pred, gt):
    _, psnr, _ = _metrics._get(pred.device)
    return psnr(pred.clamp(0.0, 1.0), gt)

def compute_lpips(pred, gt):
    """pred/gt: 1-channel float tensors in [0, 1]. Returns mean LPIPS."""
    _, _, lpips_metric = _metrics._get(pred.device)
    print(
        "LPIPS input ranges | "
        f"pred: [{pred.min().item():.4f}, {pred.max().item():.4f}] "
        f"gt: [{gt.min().item():.4f}, {gt.max().item():.4f}]"
    )
    pred_rgb = pred.clamp(0.0, 1.0).repeat(1, 3, 1, 1) * 2.0 - 1.0
    gt_rgb = gt.clamp(0.0, 1.0).repeat(1, 3, 1, 1) * 2.0 - 1.0
    return lpips_metric(pred_rgb, gt_rgb).mean()

def compute_freq_loss(pred, gt):
    pred_fft = torch.fft.rfft2(pred, norm='ortho')
    gt_fft = torch.fft.rfft2(gt, norm='ortho')
    return torch.nn.functional.l1_loss(pred_fft.abs(), gt_fft.abs())

def build_loss(l1_weight=0.5, ssim_weight=0.5, freq_weight=0.05):
    """Returns loss(pred, gt) = l1_weight * L1 + ssim_weight * (1 - SSIM)."""

    def loss(pred, gt):
        l1 = torch.nn.functional.l1_loss(pred, gt)
        ssim_loss = 1.0 - compute_ssim(pred, gt)
        freq_loss = compute_freq_loss(pred, gt)
        return (l1_weight * l1) + (ssim_weight * ssim_loss) + (freq_weight * freq_loss)

    return loss


def separate_losses(pred, gt):
    """L1 and (1-SSIM) separately, for logging both components."""
    l1 = torch.nn.functional.l1_loss(pred, gt)
    ssim = compute_ssim(pred, gt)
    freq_loss = compute_freq_loss(pred, gt)
    return l1, 1.0 - ssim, ssim, freq_loss