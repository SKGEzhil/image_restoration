"""SCUNet / SCUNetSR: Swin-Conv-UNet for image denoising and restoration.

Faithful PyTorch port of the architecture from
    "Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis"
    (Zhang et al., Machine Intelligence Research 2023)
    https://arxiv.org/abs/2203.13278
    https://github.com/cszn/SCUNet

- `SCUNet` is the same-size denoising network from the paper.
- `SCUNetSR` adds a PixelShuffle upsampling head + bilinear residual,
  adapting the architecture for super-resolution (128x128 -> 256x256).

Architecture parameters (paper defaults):
    config=[2,2,2,2,2,2,2]   # SCUNet (synthetic Gaussian denoising)
    config=[4,4,4,4,4,4,4]   # SCUNet-real (practical blind denoising)
    dim=64, head_dim=32, window_size=8

NOTE: This implementation uses only native PyTorch ops (no einops) for
full compatibility with torch.compile(fullgraph=True).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


# ─── torch.compile-friendly DropPath (no dynamic imports) ─────────────


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    When paths are dropped, the residual connection is zeroed out.
    Compatible with torch.compile since all ops are traceable.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Generate a random mask with the same batch dimension as x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).uniform_()
        random_tensor = random_tensor < keep_prob
        output = x / keep_prob * random_tensor
        return output


# ─── Window Multi-head Self-Attention ───────────────────────────────


class WMSA(nn.Module):
    """Window Multi-head Self-Attention (W-MSA / SW-MSA) module.

    Uses native PyTorch ops only — compatible with torch.compile.
    """

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

        # Relative position bias table
        self.relative_position_params = nn.Parameter(
            torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size - 1)
        )
        trunc_normal_(self.relative_position_params, std=.02)

        # Precompute coordinate grid for relative embedding
        cord = torch.tensor(
            [[i, j] for i in range(window_size) for j in range(window_size)],
            dtype=torch.long,
        )
        relation = cord[:, None, :] - cord[None, :, :] + window_size - 1
        self.register_buffer("_rel_idx_h", relation[:, :, 0].long())
        self.register_buffer("_rel_idx_w", relation[:, :, 1].long())

    def generate_mask(self, h, w, p, shift):
        """Generate attention mask for shifted window."""
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
        # Native reshape instead of einops.rearrange
        attn_mask = attn_mask.view(1, 1, h * w, p * p, p * p)
        return attn_mask

    def forward(self, x):
        """
        Args:
            x: input tensor with shape [B, H, W, C]
        Returns:
            output: tensor shape [B, H, W, C]
        """
        if self.attn_type != "W":
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))

        B, H, W, C = x.shape
        w1 = H // self.window_size
        w2 = W // self.window_size
        p = self.window_size

        # reshape: (B, H, W, C) -> (B, w1, p, w2, p, C) -> (B, w1, w2, p, p, C)
        x = x.view(B, w1, p, w2, p, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        # reshape: (B, w1, w2, p, p, C) -> (B, w1*w2, p*p, C)
        x = x.reshape(B, w1 * w2, p * p, C)

        qkv = self.embedding_layer(x)
        # reshape: (B, nw, np, 3*input_dim) -> (3, n_heads, B, nw, np, head_dim)
        qkv = qkv.view(B, w1 * w2, p * p, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(3, 4, 0, 1, 2, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (n_heads, B, nw, np, head_dim)

        # Attention: einsum over head, batch, window, token dims
        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        # Add relative position bias
        rel_emb = self.relative_embedding()  # (n_heads, p*p, p*p)
        sim = sim + rel_emb.unsqueeze(1).unsqueeze(2)  # broadcast to (n_heads, B, nw, p*p, p*p)

        # Attention mask for shifted windows
        if self.attn_type != "W":
            attn_mask = self.generate_mask(w1, w2, p, shift=p // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = F.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        # reshape: (n_heads, B, nw, np, head_dim) -> (B, nw, np, n_heads*head_dim)
        output = output.permute(1, 2, 3, 0, 4).contiguous().reshape(B, w1 * w2, p * p, C)
        output = self.linear(output)
        # reshape: (B, w1*w2, p*p, C) -> (B, w1, w2, p, p, C) -> (B, w1*p, w2*p, C)
        output = output.view(B, w1, w2, p, p, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        output = output.reshape(B, H, W, C)

        if self.attn_type != "W":
            output = torch.roll(output, shifts=(p // 2, p // 2), dims=(1, 2))
        return output

    def relative_embedding(self):
        """Lookup relative position bias from the table."""
        return self.relative_position_params[:, self._rel_idx_h, self._rel_idx_w]


# ─── Swin Transformer Block ──────────────────────────────────────────


class Block(nn.Module):
    """Swin Transformer Block (W-MSA + MLP + LayerNorm + DropPath)."""

    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path,
                 attn_type="W", input_resolution=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert attn_type in ["W", "SW"]
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


# ─── Swin-Conv Block (the "SC" in SCUNet) ────────────────────────────


class ConvTransBlock(nn.Module):
    """Swin-Conv Block: fuses residual conv path + Swin Transformer path.

    Input -> 1x1 conv -> split -> [conv_block | trans_block] -> concat -> 1x1 conv + residual
    """

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

        assert self.attn_type in ["W", "SW"]
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
        # Native permute instead of einops Rearrange
        trans_x = trans_x.permute(0, 2, 3, 1).contiguous()
        trans_x = self.trans_block(trans_x)
        trans_x = trans_x.permute(0, 3, 1, 2).contiguous()
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x


# ─── SCUNet (same-size denoising) ─────────────────────────────────────


class SCUNet(nn.Module):

    def __init__(self, in_nc=1, config=None, dim=64, drop_path_rate=0.0, input_resolution=256):
        super().__init__()
        if config is None:
            config = [2, 2, 2, 2, 2, 2, 2]
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8

        # Stochastic depth decay rule
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

        x = x[..., :h, :w]
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


# ─── SCUNetSR (super-resolution variant, 128x128 -> 256x256) ────────


class SCUNetSR(SCUNet):
    """SCUNet with a PixelShuffle upsampling head for super-resolution.

    Processes features at LR resolution through the full SCUNet UNet body,
    then upsamples via PixelShuffle(2) and adds a bilinear-upsampled input
    residual (same pattern as NAFNetSR).
    """

    def __init__(self, in_nc=1, config=None, dim=64, drop_path_rate=0.0,
                 input_resolution=256, up_scale=2):
        super().__init__(in_nc, config, dim, drop_path_rate, input_resolution)
        self.up_scale = up_scale
        # Replace the same-size tail with an upsampling head
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

        # Crop to target HR resolution
        x = x[..., : h * self.up_scale, : w * self.up_scale]

        # Bilinear residual from original (unpadded) input
        inp_hr = F.interpolate(x0, scale_factor=self.up_scale, mode="bilinear")
        return x + inp_hr


# ─── Model factories ──────────────────────────────────────────────────


def create_model_scunet(in_nc=1, config=None, dim=64, drop_path_rate=0.0,
                        input_resolution=256, **kwargs):
    """Factory for same-size SCUNet."""
    if config is None:
        config = [2, 2, 2, 2, 2, 2, 2]
    return SCUNet(
        in_nc=in_nc,
        config=config,
        dim=dim,
        drop_path_rate=drop_path_rate,
        input_resolution=input_resolution,
    )


def create_model_scunet_sr(in_nc=1, config=None, dim=64, drop_path_rate=0.0,
                           input_resolution=256, up_scale=2, **kwargs):
    """Factory for SCUNetSR (super-resolution variant)."""
    if config is None:
        config = [2, 2, 2, 2, 2, 2, 2]
    return SCUNetSR(
        in_nc=in_nc,
        config=config,
        dim=dim,
        drop_path_rate=drop_path_rate,
        input_resolution=input_resolution,
        up_scale=up_scale,
    )
