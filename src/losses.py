"""Configurable loss system for training.

Provides individual loss functions, a registry, combinators (additive,
geometric, uncertainty-weighted), presets, and a config-driven builder.

Usage in train.py:
    from losses import build_loss
    loss_fn = build_loss(config["loss_config"], device=device)
    total_loss = loss_fn(pred, gt, epoch=epoch)
    components = loss_fn.get_components(pred, gt)
"""

import math
import json
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

# ─── Individual Loss Functions ──────────────────────────────────────


def l1_loss(pred, gt):
    """L1 / Mean Absolute Error."""
    return F.l1_loss(pred, gt)


def l2_loss(pred, gt):
    """L2 / Mean Squared Error."""
    return F.mse_loss(pred, gt)


def charbonnier_loss(pred, gt, epsilon=1e-3):
    """Charbonnier loss: sqrt((pred-gt)^2 + epsilon^2)."""
    return torch.mean(torch.sqrt((pred - gt) ** 2 + epsilon ** 2))


def log_l1_loss(pred, gt, epsilon=1e-6):
    """Log-L1 loss: L1(log(pred+eps), log(gt+eps)). Speckle-aware."""
    return F.l1_loss(torch.log(pred.clamp(min=0) + epsilon),
                     torch.log(gt.clamp(min=0) + epsilon))


def ssim_loss(pred, gt, window_size=11):
    """2 * SSIMLoss. Requires kornia."""
    import kornia
    loss_fn = kornia.losses.SSIMLoss(window_size=window_size, max_val=1.0)
    return 2.0 * loss_fn(pred.clamp(0.0, 1.0), gt)

# TODO: check the kornia actual formula if it divides by 2
def ms_ssim_loss(pred, gt, window_size=11):
    """1 - MS-SSIM. Requires kornia >= 0.8."""
    import kornia
    loss_fn = kornia.losses.MS_SSIMLoss(window_size=window_size)
    return 1.0 - loss_fn(pred.clamp(0.0, 1.0), gt)


def gradient_loss(pred, gt, kernel="sobel"):
    """|∇pred - ∇gt| using Sobel or Laplacian kernels."""
    if kernel == "laplacian":
        return laplacian_loss(pred, gt)

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    pred_x = F.conv2d(pred, sobel_x, padding=1)
    pred_y = F.conv2d(pred, sobel_y, padding=1)
    gt_x = F.conv2d(gt, sobel_x, padding=1)
    gt_y = F.conv2d(gt, sobel_y, padding=1)

    return F.l1_loss(pred_x, gt_x) + F.l1_loss(pred_y, gt_y)


def laplacian_loss(pred, gt):
    """|Δpred - Δgt| using Laplacian kernel."""
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                       dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    pred_lap = F.conv2d(pred, lap, padding=1)
    gt_lap = F.conv2d(gt, lap, padding=1)
    return F.l1_loss(pred_lap, gt_lap)


def fft_loss(pred, gt):
    """FFT magnitude L1: ||FFT(pred)| - |FFT(gt)||."""
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    gt_fft = torch.fft.rfft2(gt, norm="ortho")
    return F.l1_loss(pred_fft.abs(), gt_fft.abs())


def ffl_loss(pred, gt, alpha=1.0):
    """Focal Frequency Loss: weighted L2 in freq domain.

    Weight = |FFT(pred) - FFT(gt)|^alpha (hard frequencies get more weight).
    """
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    gt_fft = torch.fft.rfft2(gt, norm="ortho")
    diff = pred_fft - gt_fft
    weight = (diff.abs() ** alpha).detach()
    return torch.mean(weight * (diff.real ** 2 + diff.imag ** 2))


def tv_loss(pred):
    """Total Variation: sum of absolute gradients. Regularization (pred only)."""
    return (torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])) +
            torch.mean(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])))


def dists_loss(pred, gt):
    """DISTS perceptual loss. Placeholder — requires pretrained model."""
    raise NotImplementedError(
        "DISTS loss not yet implemented. "
        "Needs: https://github.com/waveletsh/pytorch-DISTS"
    )


# ─── Registry ───────────────────────────────────────────────────────

LOSS_REGISTRY = OrderedDict([
    ("l1", l1_loss),
    ("l2", l2_loss),
    ("charbonnier", charbonnier_loss),
    ("log_l1", log_l1_loss),
    ("ssim", ssim_loss),
    ("ms_ssim", ms_ssim_loss),
    ("gradient", gradient_loss),
    ("laplacian", laplacian_loss),
    ("fft", fft_loss),
    ("ffl", ffl_loss),
    ("tv", tv_loss),
    ("dists", dists_loss),
    # TODO: "adversarial" — PatchGAN discriminator loss (late-stage only)
])


# ─── Combinator ─────────────────────────────────────────────────────

class LossCombinator:
    """Combines multiple losses based on mode.

    Modes:
    - additive: L = Σ wᵢ · Lᵢ  (all enabled losses)
    - geometric: geometric_terms combined as product of powers,
                 remaining enabled losses added normally
    - uncertainty: uncertainty_terms use learned σᵢ weighting,
                   remaining enabled losses added normally
    """

    def __init__(self, config, device=None):
        self.mode = config.get("mode", "additive")
        self.device = device
        self.epoch = 0
        self._components = {}

        # Parse enabled losses
        self._losses = []
        for name, cfg in config.get("losses", {}).items():
            if not cfg.get("enabled", False):
                continue
            if name not in LOSS_REGISTRY:
                raise ValueError(f"Unknown loss: {name}")

            fn = LOSS_REGISTRY[name]
            weight = cfg.get("weight", 1.0)
            ramp_epochs = cfg.get("ramp_epochs", 0)
            # Extra params (epsilon, alpha, kernel, window_size, etc.)
            skip_keys = {"enabled", "weight", "ramp_epochs", "note"}
            params = {k: v for k, v in cfg.items() if k not in skip_keys}
            self._losses.append({
                "name": name, "weight": weight, "fn": fn,
                "params": params, "ramp_epochs": ramp_epochs,
            })

        # Geometric mode: extract terms and auto-normalize exponents
        self._geo_terms = []
        self._geo_exps = []
        if self.mode == "geometric":
            geo_list = config.get("geometric_terms", [])
            # Find matching enabled losses
            available = {l["name"]: l for l in self._losses}
            raw_exps = []
            for t in geo_list:
                if t in available:
                    raw_exps.append(available[t]["weight"])
                    self._geo_terms.append(t)
            # Normalize exponents to sum=1
            total = sum(raw_exps)
            if total > 0:
                self._geo_exps = [e / total for e in raw_exps]

        # Uncertainty mode: learnable log(σ²)
        self._unc_terms = []
        self._log_sigma_sq = {}
        if self.mode == "uncertainty":
            unc_list = config.get("uncertainty_terms", [])
            init_sigma = config.get("uncertainty_init_sigma", 1.0)
            init_log_sq = math.log(init_sigma ** 2)
            available = {l["name"]: l for l in self._losses}
            for t in unc_list:
                if t in available:
                    self._unc_terms.append(t)
                    self._log_sigma_sq[t] = torch.tensor(
                        init_log_sq, dtype=torch.float32,
                        device=device, requires_grad=True,
                    )

    @property
    def log_sigma_sq_params(self):
        """Return learnable log(σ²) parameters for optimizer."""
        if self.mode != "uncertainty":
            return []
        return list(self._log_sigma_sq.values())

    def step(self, epoch):
        """Call once per epoch (for ramp-in scheduling)."""
        self.epoch = epoch

    def __call__(self, pred, gt, epoch=None):
        if epoch is not None:
            self.epoch = epoch

        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        self._components = {}

        # Compute all enabled losses
        loss_vals = {}
        for entry in self._losses:
            name = entry["name"]
            try:
                val = entry["fn"](pred, gt, **entry["params"])
            except TypeError:
                # Some losses (e.g. tv_loss) don't take gt
                val = entry["fn"](pred)
            loss_vals[name] = val
            self._components[name] = val.item()

        # Combine based on mode
        if self.mode == "geometric" and self._geo_terms:
            # Additive part: non-geo terms
            for entry in self._losses:
                name = entry["name"]
                if name in self._geo_terms:
                    continue
                w = self._apply_ramp(entry)
                total = total + w * loss_vals[name]

            # Geometric part: product of powers
            geo_product = torch.tensor(1.0, device=pred.device, dtype=pred.dtype)
            for name, exp in zip(self._geo_terms, self._geo_exps):
                val = loss_vals[name]
                geo_product = geo_product * (val + 1e-8) ** exp
            total = total + geo_product

        elif self.mode == "uncertainty" and self._unc_terms:
            # Additive part: non-uncertainty terms
            for entry in self._losses:
                name = entry["name"]
                if name in self._unc_terms:
                    continue
                w = self._apply_ramp(entry)
                total = total + w * loss_vals[name]

            # Uncertainty part: Σ (1/2σ²)·L + log(σ)
            for name in self._unc_terms:
                log_sq = self._log_sigma_sq[name]
                val = loss_vals[name]
                total = total + (1 / (2 * torch.exp(log_sq))) * val + log_sq

        else:
            # Additive mode (or fallback)
            for entry in self._losses:
                w = self._apply_ramp(entry)
                total = total + w * loss_vals[entry["name"]]

        return total

    def _apply_ramp(self, entry):
        """Apply linear ramp-in for the first ramp_epochs."""
        w = entry["weight"]
        ramp = entry["ramp_epochs"]
        if ramp > 0 and self.epoch < ramp:
            w = w * (self.epoch / ramp)
        return w

    def get_components(self, pred, gt):
        """Return {name: value} dict for logging."""
        with torch.no_grad():
            for entry in self._losses:
                name = entry["name"]
                try:
                    val = entry["fn"](pred, gt, **entry["params"])
                except TypeError:
                    val = entry["fn"](pred)
                self._components[name] = val.item()
        return dict(self._components)

    def get_params(self):
        """Return serializable state (for checkpointing)."""
        if self.mode != "uncertainty":
            return {}
        return {t: v.item() for t, v in self._log_sigma_sq.items()}

    def load_params(self, params):
        """Restore state from checkpoint."""
        if self.mode != "uncertainty":
            return
        for t, val in params.items():
            if t in self._log_sigma_sq:
                self._log_sigma_sq[t].data.fill_(val)

    def __repr__(self):
        names = [e["name"] for e in self._losses]
        return f"LossCombinator(mode={self.mode}, losses={names})"


# ─── Presets (loaded from loss_presets.json) ─────────────────────────

def _load_presets():
    presets_path = Path(__file__).parent / "loss_presets.json"
    with open(presets_path) as f:
        return json.load(f)

PRESETS = _load_presets()


# ─── Builder ────────────────────────────────────────────────────────

def _migrate_legacy(config):
    """Convert old flat-weight config to new format."""
    losses = {}
    if config.get("l1_weight", 0) > 0:
        losses["l1"] = {"enabled": True, "weight": config["l1_weight"]}
    if config.get("ssim_weight", 0) > 0:
        losses["ssim"] = {"enabled": True, "weight": config["ssim_weight"]}
    if config.get("freq_weight", 0) > 0:
        losses["fft"] = {"enabled": True, "weight": config["freq_weight"]}
    return {"mode": "additive", "losses": losses}


def build_loss(config, device=None):
    """Build loss combinator from config dict.

    Supports:
    - {"preset": "combo_1"} → use preset
    - {"mode": "additive", "losses": {...}} → manual config
    - {"l1_weight": 1, ...} → legacy format (auto-migrated)
    """
    if not config:
        # Empty config — fallback to old behavior
        config = {"l1_weight": 0.5, "ssim_weight": 0.5, "freq_weight": 0.05}

    if "preset" in config and config["preset"] is not None:
        preset_name = config["preset"]
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
        config = PRESETS[preset_name]
    elif "losses" not in config:
        config = _migrate_legacy(config)

    return LossCombinator(config, device=device)
