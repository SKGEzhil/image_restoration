"""Losses for CG-NAFNet (implementation.md §4).

Total loss (config-weighted):
    L = L_recon + lambda_ssim * L_ms_ssim + lambda_ortho * L_ortho
        + lambda_aux * L_aux_order
"""

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim


def l1_loss(pred, target):
    return F.l1_loss(pred, target)


def charbonnier_loss(pred, target, eps=1e-6):
    """Charbonnier (smooth L1) loss, common choice for restoration."""
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps))


_MIN_MS_SSIM_SIZE = (11 - 1) * (2 ** 4) + 1  # 161 with default 5 scales


def ms_ssim_loss(pred, target, data_range=1.0, size_average=True):
    """1 - MS-SSIM (pytorch-msssim).

    Falls back to single-scale SSIM for small tensors where MS-SSIM's 5-scale
    pyramid is mathematically undefined.
    """
    h, w = pred.shape[-2:]
    if h < _MIN_MS_SSIM_SIZE or w < _MIN_MS_SSIM_SIZE:
        return 1.0 - ssim(pred, target, data_range=data_range, size_average=size_average)
    return 1.0 - ms_ssim(pred, target, data_range=data_range, size_average=size_average)


def orthogonality_reg(prototype_bank):
    """Orthogonality regularizer on a prototype bank: ||P P^T - I||_F^2.

    Args:
        prototype_bank: (num_prototypes, proj_dim) — the *normalized* bank
            (normalize inside the caller with the same normalization used by
            PCGRMLite.forward so the penalty targets what routing sees).

    Returns:
        scalar tensor.
    """
    P = F.normalize(prototype_bank, dim=-1)
    gram = P @ P.t()  # (num_prototypes, num_prototypes)
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return F.mse_loss(gram, identity)


def auxiliary_order_loss(logits, order_labels):
    """Cross-entropy for the auxiliary order classifier (train-only head)."""
    return F.cross_entropy(logits, order_labels)


_LOSS_REGISTRY = {
    "l1": l1_loss,
    "charbonnier": charbonnier_loss,
}


def build_total_loss(
    pred,
    target,
    aux_logits=None,
    order_labels=None,
    model=None,
    recon_type="charbonnier",
    lambda_ssim=0.2,
    lambda_ortho=0.01,
    lambda_aux=0.1,
):
    """Compute the full weighted loss and return (total, component_dict)."""
    recon_fn = _LOSS_REGISTRY[recon_type]
    recon = recon_fn(pred, target)
    ssim = ms_ssim_loss(pred, target)

    total = recon + lambda_ssim * ssim
    components = {
        "loss_total": total.detach().item(),
        "recon": recon.detach().item(),
        "ssim": ssim.detach().item(),
        "ortho": 0.0,
        "aux": 0.0,
    }

    if model is not None and lambda_ortho > 0:
        ortho = 0.0
        for stage in _all_stage_modules(model):
            ortho = ortho + orthogonality_reg(stage.pcgrm.prototype)
        components["ortho"] = ortho.detach().item()
        total = total + lambda_ortho * ortho

    if aux_logits is not None and order_labels is not None and lambda_aux > 0:
        aux = auxiliary_order_loss(aux_logits, order_labels)
        components["aux"] = aux.detach().item()
        total = total + lambda_aux * aux

    components["loss_total"] = total.detach().item()
    return total, components


def _all_stage_modules(model):
    """Yield the routing _Stage module of every enc/dec stage."""
    for attr in ("enc_stages", "dec_stages"):
        stages = getattr(model, attr, None)
        if stages is not None:
            yield from stages