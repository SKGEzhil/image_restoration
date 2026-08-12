"""Standard NAFBlock — the unmodified backbone primitive for CG-NAFNet.

Exact port of the megvii-research/NAFNet block (also mirrored in this repo's
legacy `src/models/nafnet.py`). Kept byte-for-byte faithful and deliberately
naive/dumb — all restoration-specific logic (cluster conditioning, prompts,
FiLM) lives in wrapper modules outside this file, so the block stays a clean,
swappable backbone primitive.
"""

import torch
import torch.nn as nn


class LayerNormFunction(torch.autograd.Function):
    """Channel-wise LayerNorm with fused backward (as in arch_util.py)."""

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
    """Chunk-and-multiply nonlinear activation (the "NAF" in NAFNet)."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Core building block (exact port of NAFNet_arch.NAFBlock)."""

    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * dw_expand

        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )

        # SimpleGate
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