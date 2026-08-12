"""NAFNet / NAFNetSR: Nonlinear Activation-Free Network.

Faithful port of the architecture from
    "Simple Baselines for Image Restoration" (Chen et al., ECCV 2022)
    https://arxiv.org/abs/2204.04676
    https://github.com/megvii-research/NAFNet
        basicsr/models/archs/{NAFNet_arch.py, NAFSSR_arch.py, arch_util.py}

- `NAFNet` is the flagship encoder-decoder restoration network from the paper.
- `NAFNetSR` is the same paper's super-resolution model (NAFSSR, single view):
  a body of NAFBlocks at LR resolution plus a PixelShuffle upsampling head and
  a bilinear-upsampled-input residual. That is what maps 128x128 -> 256x256.

For the SR + denoise task in this repo, `create_model_nafnet()` returns NAFNetSR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class NAFNet(nn.Module):
    """Flagship encoder-decoder NAFNet (exact port, same-size restoration)."""

    def __init__(self, img_channel=1, width=32, middle_blk_num=1,
                 enc_blk_nums=(1, 1, 4, 1), dec_blk_nums=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

    def forward(self, inp):
        _, _, h, w = inp.shape
        x = self.intro(self.check_image_size(inp))

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x) + self.check_image_size(inp)
        return x[:, :, :h, :w]


class NAFNetSR(nn.Module):
    """NAFNet super-resolution model (NAFSSR, single view).

    Body of NAFBlocks at LR resolution + PixelShuffle upsampling head and a
    bilinear-upsampled-input residual (exact port of NAFSSR_arch.NAFNetSR).

    Args:
        up_scale: spatial upsampling factor (2 -> 128x128 -> 256x256).
        width: base feature width.
        num_blks: number of NAFBlocks in the body.
        img_channel: input/output channels (1 for grayscale).
    """

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


def create_model_nafnet(width=32, num_blks=8, img_channel=1, drop_out_rate=0.0, **kwargs):
    """Factory for NAFNetSR. Matches create_model() signature from original model.py."""
    return NAFNetSR(
        up_scale=2,
        width=width,
        num_blks=num_blks,
        img_channel=img_channel,
        drop_out_rate=drop_out_rate,
    )
