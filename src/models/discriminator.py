"""PatchGAN discriminator with spectral normalization.

Based on the KAIR / BasicSR discriminator architectures used across
the SCUNet / Real-ESRGAN / SRGAN ecosystem.

NOTE: forward() and forward_features() are separate methods for
torch.compile compatibility — each has a static return signature.

References:
    https://github.com/cszn/KAIR/blob/master/models/network_discriminator.py
    https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/discriminator_arch.py
"""

import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator (70x70 receptive field when n_layers=3).

    Args:
        input_nc: number of input channels (1 for grayscale, 3 for RGB).
        ndf: base number of channels.
        n_layers: number of strided conv layers.
        use_spectral_norm: whether to apply spectral normalization.
    """

    def __init__(self, input_nc=1, ndf=64, n_layers=3, use_spectral_norm=True):
        super(PatchGANDiscriminator, self).__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        # First layer (no norm)
        conv0 = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        if use_spectral_norm:
            conv0 = spectral_norm(conv0)
        sequence = [[conv0, nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            conv = nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, bias=False)
            if use_spectral_norm:
                conv = spectral_norm(conv)
            sequence += [[conv, nn.BatchNorm2d(nf, affine=True), nn.LeakyReLU(0.2, True)]]

        # Penultimate layer (stride 1)
        nf_prev = nf
        nf = min(nf * 2, 512)
        conv = nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw, bias=False)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        sequence += [[conv, nn.BatchNorm2d(nf, affine=True), nn.LeakyReLU(0.2, True)]]

        # Final layer -> single-channel score map
        conv_last = nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)
        if use_spectral_norm:
            conv_last = spectral_norm(conv_last)
        sequence += [[conv_last]]

        self.model = nn.Sequential()
        for n in range(len(sequence)):
            self.model.add_module("child" + str(n), nn.Sequential(*sequence[n]))

        self.model.apply(self._weights_init)

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if "Conv" in classname:
            m.weight.data.normal_(0.0, 0.02)
        elif "BatchNorm2d" in classname:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        """Standard forward: returns discriminator logits only.

        Static return type (Tensor) for torch.compile compatibility.
        """
        return self.model(x)

    def forward_features(self, x):
        """Forward that returns logits + intermediate layer features.

        Static return type (Tensor, list[Tensor]) for torch.compile.
        Use this when feature-matching loss is needed.
        """
        features = []
        for module in self.model:
            x = module(x)
            features.append(x)
        return x, features


def create_discriminator(input_nc=1, ndf=64, n_layers=3, use_spectral_norm=True):
    """Factory for PatchGAN discriminator."""
    return PatchGANDiscriminator(
        input_nc=input_nc,
        ndf=ndf,
        n_layers=n_layers,
        use_spectral_norm=use_spectral_norm,
    )
