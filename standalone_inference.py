#!/usr/bin/env python3
"""Standalone inference script for KLA Hackathon 2026 image restoration.

Self-contained — does NOT import from src/. The SCUNetSR and NAFNetSR model
architectures are inlined below so evaluators can run this script with only
PyTorch and numpy installed.

Usage:
    python standalone_inference.py --input-dir test/NoisyLR --output-dir results/
    python standalone_inference.py --input-dir /data/test/NoisyLR --output-dir /data/restored --checkpoint weights/best.pt --batch-size 8
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Inlined Model: SCUNet / SCUNetSR
# =============================================================================

try:
    from timm.layers import trunc_normal_
except ImportError:
    def trunc_normal_(tensor, std=0.02):
        nn.init.trunc_normal_(tensor, std=std)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).uniform_()
        random_tensor = random_tensor < keep_prob
        return x / keep_prob * random_tensor


class WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, attn_type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.attn_type = attn_type

        self.embedding_layer = nn.Linear(input_dim, 3 * input_dim, bias=True)
        self.linear = nn.Linear(input_dim, output_dim)

        self.relative_position_params = nn.Parameter(
            torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size - 1)
        )
        trunc_normal_(self.relative_position_params, std=0.02)

        cord = torch.tensor(
            [[i, j] for i in range(window_size) for j in range(window_size)],
            dtype=torch.long,
        )
        relation = cord[:, None, :] - cord[None, :, :] + window_size - 1
        self.register_buffer("_rel_idx_h", relation[:, :, 0].long())
        self.register_buffer("_rel_idx_w", relation[:, :, 1].long())

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(
            h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device
        )
        if self.attn_type == "W":
            return attn_mask
        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = attn_mask.view(1, 1, h * w, p * p, p * p)
        return attn_mask

    def forward(self, x):
        if self.attn_type != "W":
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        B, H, W, C = x.shape
        w1 = H // self.window_size
        w2 = W // self.window_size
        p = self.window_size

        x = x.view(B, w1, p, w2, p, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, w1 * w2, p * p, C)

        qkv = self.embedding_layer(x)
        qkv = qkv.view(B, w1 * w2, p * p, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(3, 4, 0, 1, 2, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        rel_emb = self.relative_embedding()
        sim = sim + rel_emb.unsqueeze(1).unsqueeze(2)

        if self.attn_type != "W":
            attn_mask = self.generate_mask(w1, w2, p, shift=p // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = F.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = output.permute(1, 2, 3, 0, 4).contiguous().reshape(B, w1 * w2, p * p, C)
        output = self.linear(output)
        output = output.view(B, w1, w2, p, p, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        output = output.reshape(B, H, W, C)

        if self.attn_type != "W":
            output = torch.roll(output, shifts=(p // 2, p // 2), dims=(1, 2))
        return output

    def relative_embedding(self):
        return self.relative_position_params[:, self._rel_idx_h, self._rel_idx_w]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path,
                 attn_type="W", input_resolution=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attn_type = attn_type
        if input_resolution is not None and input_resolution <= window_size:
            self.attn_type = "W"

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.attn_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path,
                 attn_type="W", input_resolution=None):
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.attn_type = attn_type
        self.input_resolution = input_resolution

        if self.input_resolution is not None and self.input_resolution <= self.window_size:
            self.attn_type = "W"

        self.trans_block = Block(
            self.trans_dim, self.trans_dim, self.head_dim, self.window_size,
            self.drop_path, self.attn_type, self.input_resolution,
        )
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = trans_x.permute(0, 2, 3, 1).contiguous()
        trans_x = self.trans_block(trans_x)
        trans_x = trans_x.permute(0, 3, 1, 2).contiguous()
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        return x + res


class SCUNet(nn.Module):
    def __init__(self, in_nc=1, config=None, dim=64, drop_path_rate=0.0, input_resolution=256):
        super().__init__()
        if config is None:
            config = [2, 2, 2, 2, 2, 2, 2]
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)

        begin = 0
        self.m_down1 = nn.Sequential(
            *[
                ConvTransBlock(
                    dim // 2, dim // 2, self.head_dim, self.window_size,
                    dpr[i + begin], "W" if not i % 2 else "SW", input_resolution,
                )
                for i in range(config[0])
            ],
            nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False),
        )

        begin += config[0]
        self.m_down2 = nn.Sequential(
            *[
                ConvTransBlock(
                    dim, dim, self.head_dim, self.window_size,
                    dpr[i + begin], "W" if not i % 2 else "SW", input_resolution // 2,
                )
                for i in range(config[1])
            ],
            nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False),
        )

        begin += config[1]
        self.m_down3 = nn.Sequential(
            *[
                ConvTransBlock(
                    2 * dim, 2 * dim, self.head_dim, self.window_size,
                    dpr[i + begin], "W" if not i % 2 else "SW", input_resolution // 4,
                )
                for i in range(config[2])
            ],
            nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False),
        )

        begin += config[2]
        self.m_body = nn.Sequential(
            *[
                ConvTransBlock(
                    4 * dim, 4 * dim, self.head_dim, self.window_size,
                    dpr[i + begin], "W" if not i % 2 else "SW", input_resolution // 8,
                )
                for i in range(config[3])
            ]
        )

        begin += config[3]
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False),
            *[
                ConvTransBlock(
                    2 * dim, 2 * dim, self.head_dim, self.window_size,
                    dpr[i + begin], "W" if not i % 2 else "SW", input_resolution // 4,
                )
                for i in range(config[4])
            ],
        )

        begin += config[4]
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False),
            *[
                ConvTransBlock(
                    dim, dim, self.head_dim, self.window_size,
                    dpr[i + begin], "W" if not i % 2 else "SW", input_resolution // 2,
                )
                for i in range(config[5])
            ],
        )

        begin += config[5]
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False),
            *[
                ConvTransBlock(
                    dim // 2, dim // 2, self.head_dim, self.window_size,
                    dpr[i + begin], "W" if not i % 2 else "SW", input_resolution,
                )
                for i in range(config[6])
            ],
        )

        self.m_tail = nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False)
        self._init_weights()

    def forward(self, x0):
        h, w = x0.size()[-2:]
        padding_bottom = math.ceil(h / 64) * 64 - h
        padding_right = math.ceil(w / 64) * 64 - w
        if padding_bottom > 0 or padding_right > 0:
            x0 = F.pad(x0, (0, padding_right, 0, padding_bottom), mode="replicate")

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x[..., :h, :w]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


class SCUNetSR(SCUNet):
    def __init__(self, in_nc=1, config=None, dim=64, drop_path_rate=0.0,
                 input_resolution=256, up_scale=2):
        super().__init__(in_nc, config, dim, drop_path_rate, input_resolution)
        self.up_scale = up_scale
        self.m_tail = nn.Sequential(
            nn.Conv2d(dim, in_nc * (up_scale ** 2), 3, 1, 1, bias=True),
            nn.PixelShuffle(up_scale),
        )

    def forward(self, x0):
        h, w = x0.size()[-2:]
        padding_bottom = math.ceil(h / 64) * 64 - h
        padding_right = math.ceil(w / 64) * 64 - w
        x0_padded = x0
        if padding_bottom > 0 or padding_right > 0:
            x0_padded = F.pad(x0, (0, padding_right, 0, padding_bottom), mode="replicate")

        x1 = self.m_head(x0_padded)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., : h * self.up_scale, : w * self.up_scale]
        inp_hr = F.interpolate(x0, scale_factor=self.up_scale, mode="bilinear")
        return x + inp_hr


# =============================================================================
# Inlined Model: NAFNet / NAFNetSR
# =============================================================================


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(x, weight, bias, eps):
        n, c, h, w = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y_norm = (x - mu) / (var + eps).sqrt()
        y = weight.view(1, c, 1, 1) * y_norm + bias.view(1, c, 1, 1)
        return y, y_norm, var

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, eps = inputs
        y, y_norm, var = output
        ctx.eps = eps
        ctx.save_for_backward(y_norm, var, weight)

    @staticmethod
    def backward(ctx, grad_output, grad_y_norm, grad_var):
        eps = ctx.eps
        n, c, h, w = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, c, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = (g - y * mean_gy - mean_g) / (var + eps).sqrt()
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        y, _, _ = LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
        return y


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNetSR(nn.Module):
    def __init__(self, up_scale=2, width=32, num_blks=8, img_channel=1, drop_out_rate=0.0):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.body = nn.Sequential(*[NAFBlock(width, drop_out_rate=drop_out_rate) for _ in range(num_blks)])
        self.up = nn.Sequential(
            nn.Conv2d(width, img_channel * up_scale**2, 3, padding=1, bias=True),
            nn.PixelShuffle(up_scale),
        )
        self.up_scale = up_scale

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode="bilinear")
        feats = self.body(self.intro(inp))
        return self.up(feats) + inp_hr


# =============================================================================
# Model Loading
# =============================================================================


def load_model(checkpoint_path, device):
    """Load model from checkpoint, auto-detecting architecture."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("args", {})
    model_name = cfg.get("train_model", "scunet_sr")
    model_params = cfg.get("models", {}).get(model_name, {})

    if model_name in ("scunet_sr", "scunet_sr_real", "scunet", "scunet_real"):
        model = SCUNetSR(**model_params)
    elif model_name in ("nafnet",):
        model = NAFNetSR(**model_params)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    state_dict = ckpt["model"]
    # Strip torch.compile prefixes if present
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    return model.to(device), model_name


# =============================================================================
# Dataset
# =============================================================================


class NpyFolder(Dataset):
    """Loads .npy files from a directory."""

    def __init__(self, root):
        self.root = Path(root)
        self.names = sorted(p.name for p in self.root.glob("*.npy"))
        if len(self.names) == 0:
            raise FileNotFoundError(f"No .npy files found in {root}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        arr = np.load(self.root / name)
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).float()
        return tensor, name


# =============================================================================
# Inference
# =============================================================================


@torch.no_grad()
def tta_inference(model, x, device):
    """Test-time augmentation: average over 4 rotations x 2 flips = 8 predictions."""
    preds = []
    for k in (0, 1, 2, 3):
        x_rot = torch.rot90(x, k, [2, 3])
        for do_flip in (False, True):
            x_in = torch.flip(x_rot, [3]) if do_flip else x_rot
            out = model(x_in.to(device)).clamp(0, 1)
            if do_flip:
                out = torch.flip(out, [3])
            out = torch.rot90(out, -k, [2, 3])
            preds.append(out.cpu())
    return torch.stack(preds, dim=0).mean(dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone image restoration inference (self-contained, no src/ imports)"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing degraded .npy images")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write restored .npy images")
    parser.add_argument("--checkpoint", type=str, default="weights/best.pt",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, mps, or cpu (default: auto-detect)")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model, model_name = load_model(str(checkpoint_path), device)
    model.eval()
    print(f"Model: {model_name} loaded from {checkpoint_path}")

    # Dataset and loader
    ds = NpyFolder(args.input_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    print(f"Found {len(ds)} images in {args.input_dir}")

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    total_start = time.time()
    for batch_idx, (images, names) in enumerate(loader):
        batch_start = time.time()
        images = images.to(device)
        restored = tta_inference(model, images, device)
        batch_time = time.time() - batch_start

        for name, out in zip(names, restored):
            np.save(str(out_dir / name), out.squeeze(0).numpy())

        print(f"  Batch {batch_idx + 1}/{len(loader)}: "
              f"{len(names)} images, {batch_time:.2f}s")

    total_time = time.time() - total_start
    print(f"\nDone: {len(ds)} images restored in {total_time:.2f}s "
          f"({len(ds) / total_time:.1f} images/s)")
    print(f"Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
