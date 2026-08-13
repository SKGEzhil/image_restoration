"""DISTS (Deep Image Structure and Texture Similarity) loss module.

Based on the paper:
    "Image Quality Assessment: Unifying Structure and Texture Similarity"
    by Keyan Ding, Kede Ma, Shiqi Wang, Eero P. Simoncelli (2020)

Reference implementation: https://github.com/chaofengc/IQA-PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class L2pooling(nn.Module):
    """L2 pooling with Hanning window filter."""

    def __init__(self, filter_size=5, stride=2, channels=None):
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer(
            'filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

    def forward(self, x):
        x = x ** 2
        out = F.conv2d(
            x, self.filter, stride=self.stride,
            padding=self.padding, groups=x.shape[1],
        )
        return (out + 1e-12).sqrt()


class DISTS(nn.Module):
    """DISTS perceptual metric/loss using VGG16 features.

    Computes structure similarity (S1) and texture similarity (S2)
    across multiple VGG16 stages with learned alpha/beta weights.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        vgg_features = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None).features

        self.stage1 = nn.Sequential(*list(vgg_features[:4]))
        self.stage2 = nn.Sequential(
            L2pooling(channels=64),
            *list(vgg_features[5:9])
        )
        self.stage3 = nn.Sequential(
            L2pooling(channels=128),
            *list(vgg_features[10:16])
        )
        self.stage4 = nn.Sequential(
            L2pooling(channels=256),
            *list(vgg_features[17:23])
        )
        self.stage5 = nn.Sequential(
            L2pooling(channels=512),
            *list(vgg_features[24:30])
        )

        # Freeze all VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        self.chns = [3, 64, 128, 256, 512, 512]

        # Learnable weights for structure (alpha) and texture (beta)
        self.alpha = nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))
        self.beta = nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

    def forward_once(self, x):
        """Extract features from all5 stages."""
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y):
        """Compute DISTS loss.

        Args:
            x: Predicted image tensor (N, C, H, W). Grayscale auto-repeated to RGB.
            y: Ground truth image tensor (N, C, H, W). Grayscale auto-repeated to RGB.

        Returns:
            DISTS score (lower = more similar).
        """
        # Handle grayscale input: repeat to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)

        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)

        dist1 = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        dist2 = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        c1 = 1e-6
        c2 = 1e-6

        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)

        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        score = 1 - (dist1 + dist2)
        return score.mean()


def create_dists_model(device, pretrained=True):
    """Factory: create DISTS model on device."""
    model = DISTS(pretrained=pretrained)
    return model.to(device).eval()
