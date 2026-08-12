"""Auxiliary degradation-order classifier (design doc §5, train-only).

Consumes the deepest-stage cluster posterior embedding (``x_hat``) and predicts
which of the 6 degradation orders was applied. This is an auxiliary training-
time loss only: it must be trivially removable with ZERO effect on the
restoration output path. The main model simply does not call this head.
"""

import torch
import torch.nn as nn


class AuxOrderHead(nn.Module):
    """Linear classifier over the deepest-stage posterior embedding.

    Args:
        feat_dim: dimension of ``x_hat`` (the projected, normalized embedding).
        num_orders: number of degradation orders (6).
    """

    def __init__(self, feat_dim, num_orders=6):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, num_orders)

    def forward(self, x_hat):
        """x_hat: (B, feat_dim) -> logits (B, num_orders)."""
        return self.classifier(x_hat)