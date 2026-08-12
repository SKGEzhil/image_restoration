"""Cluster-conditioned degradation prompt (design doc §4.2).

The soft posterior ``alpha`` (over the stage's prototype bank) weights a set of
reparameterized latent codes ``mu_c + sigma_c * eps`` into a single degradation
prompt vector ``P`` used by the stage's FiLM layers.
"""

import torch
import torch.nn as nn


class DegradationPrompt(nn.Module):
    """Reparameterized cluster prior -> degradation prompt.

    Args:
        num_prototypes: number of clusters (matches PCGRMLite bank).
        prompt_dim: dimensionality of the prompt vector.
    """

    def __init__(self, num_prototypes, prompt_dim):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prompt_dim = prompt_dim
        # mu and log_sigma per prototype; log parameterization for stability.
        self.mu = nn.Parameter(torch.randn(num_prototypes, prompt_dim) * 0.1)
        self.log_sigma = nn.Parameter(torch.randn(num_prototypes, prompt_dim) * 0.1 - 3.0)

    def forward(self, alpha):
        """alpha: (B, num_prototypes) -> prompt P: (B, prompt_dim).

        Training: sample eps ~ N(0, I) fresh each call.
        Eval: deterministic, eps = 0 (reproducibility, design doc §3.3).
        """
        sigma = torch.exp(self.log_sigma)  # positive
        if self.training:
            eps = torch.randn(
                self.num_prototypes, self.prompt_dim,
                device=alpha.device, dtype=alpha.dtype,
            )
        else:
            eps = torch.zeros(self.num_prototypes, self.prompt_dim,
                              device=alpha.device, dtype=alpha.dtype)
        per_proto = self.mu + sigma * eps  # (num_prototypes, prompt_dim)
        # Weighted sum by soft posterior: P[b] = sum_c alpha[b, c] * per_proto[c, :]
        return alpha @ per_proto