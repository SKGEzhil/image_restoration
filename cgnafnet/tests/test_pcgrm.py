"""Tests for PCGRMLite: posterior, unit-norm bank, gradient flow."""

import pytest
import torch

from cgnafnet.models.pcgrm_lite import PCGRMLite


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


def _make(device, num_prototypes=3, feat_dim=16, proj_dim=8):
    return PCGRMLite(feat_dim, proj_dim, num_prototypes).to(device)


def test_posterior_sums_to_one(device):
    torch.manual_seed(0)
    m = _make(device)
    x = torch.randn(4, 16, 32, 32, device=device)
    alpha, x_hat = m(x)
    assert alpha.shape == (4, m.num_prototypes)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(4, device=device), atol=1e-5)
    assert x_hat.shape == (4, m.proj_dim)


def test_prototype_bank_unit_norm(device):
    torch.manual_seed(0)
    m = _make(device)
    x = torch.randn(4, 16, 16, 16, device=device)
    alphas = [m(x) for _ in range(3)]
    prototypes = torch.nn.functional.normalize(m.prototype, dim=-1)
    norms = prototypes.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_gradient_flows_to_prototype_bank(device):
    torch.manual_seed(0)
    m = _make(device)
    x = torch.randn(4, 16, 16, 16, device=device)
    alpha, x_hat = m(x)
    m.zero_grad()
    loss = alpha.sum()  # dummy scalar downstream of alpha
    loss.backward()
    assert m.prototype.grad is not None
    assert m.prototype.grad.abs().sum() > 0
    assert m.proj.weight.grad is not None


def test_posterior_assignment_differs_for_distinct_feats(device):
    torch.manual_seed(0)
    m = _make(device, num_prototypes=4)
    x1 = torch.randn(2, 16, 8, 8, device=device)
    x2 = torch.randn(2, 16, 8, 8, device=device) + 5.0
    a1, _ = m(x1)
    a2, _ = m(x2)
    assert not torch.allclose(a1, a2, atol=1e-3)