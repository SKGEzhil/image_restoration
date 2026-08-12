"""Cluster-Guided Prior — lite PCGRM (design doc §4.1).

Soft posterior over a learnable prototype bank, computed from the global
(1x1 pooled) feature representation of a stage. All routing is soft — no
top-k, no argmax. The posterior is returned for the auxiliary order head /
validation and used to weight the reparameterized degradation prompt.
"""

import torch
import torch.nn as nn


class PCGRMLite(nn.Module):
    """Prototype-guided soft cluster posterior.

    Args:
        feat_dim: channel dimension of the input feature map.
        proj_dim: projection space dimensionality.
        num_prototypes: number of learnable clusters.
    """

    def __init__(self, feat_dim, proj_dim, num_prototypes):
        super().__init__()
        # Learnable prototype bank, initialized ORTHOGONALLY — a validated
        # finding from the reference design, not optional.
        self.prototype = nn.Parameter(torch.empty(num_prototypes, proj_dim))
        nn.init.orthogonal_(self.prototype)

        self.proj = nn.Linear(feat_dim, proj_dim)
        self.proj_dim = proj_dim
        self.num_prototypes = num_prototypes

    def forward(self, x):
        """x: (B, feat_dim, H, W) -> (alpha, x_hat).

        alpha: (B, num_prototypes) softmax over cosine similarities, sums to 1.
        x_hat: (B, proj_dim) L2-normalized projected embedding.
        """
        pooled = x.mean(dim=(2, 3))  # global average pool
        x_hat = self.proj(pooled)
        x_hat = torch.nn.functional.normalize(x_hat, dim=-1)

        # Keep the bank unit-norm each forward (prototype params stay L2=1).
        prototypes = torch.nn.functional.normalize(self.prototype, dim=-1)

        similarities = x_hat @ prototypes.T  # (B, num_prototypes)
        alpha = torch.softmax(similarities, dim=-1)
        return alpha, x_hat