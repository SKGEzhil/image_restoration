"""FiLM (Feature-wise Linear Modulation) conditioning layer (design doc §4.3).

Applies per-channel affine modulation to the stage feature map using the
degradation prompt: ``gamma * x + beta``. The linear projection is zero-inited
so the layer starts as an identity (gamma=1, beta=0) and does not destabilize
the freshly-initialized backbone.
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Args:
        prompt_dim: dimensionality of the degradation prompt.
        num_channels: channel count of the feature map to modulate.
    """

    def __init__(self, prompt_dim, num_channels):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.num_channels = num_channels
        self.gate = nn.Linear(prompt_dim, 2 * num_channels)
        # Zero-init: gamma = 1 for the offset-half by construction, beta = 0.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x, prompt):
        """x: (B, C, H, W), prompt: (B, prompt_dim) -> modulated (B, C, H, W)."""
        g = self.gate(prompt)  # (B, 2C)
        gamma, beta = g.chunk(2, dim=1)  # each (B, C)
        gamma = (1.0 + gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta