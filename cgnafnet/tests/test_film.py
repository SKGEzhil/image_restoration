"""Phase-2 module tests for DegradationPrompt + FiLM (implementation.md §3.3-3.4)."""

import pytest
import torch

from cgnafnet.models.degradation_prompt import DegradationPrompt
from cgnafnet.models.film import FiLM


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


def test_prompt_eval_deterministic(device):
    torch.manual_seed(0)
    p = DegradationPrompt(num_prototypes=3, prompt_dim=8).to(device)
    p.eval()
    alpha = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]], device=device)
    out1 = p(alpha)
    out2 = p(alpha)
    assert torch.equal(out1, out2)
    assert out1.shape == (2, 8)


def test_prompt_train_stochastic(device):
    torch.manual_seed(0)
    p = DegradationPrompt(num_prototypes=3, prompt_dim=8).to(device)
    p.train()
    alpha = torch.tensor([[0.5, 0.3, 0.2]], device=device)
    out1 = p(alpha)
    out2 = p(alpha)
    assert not torch.equal(out1, out2)
    assert out1.shape == (1, 8)


def test_prompt_weighted_mean_updates_mu(device):
    torch.manual_seed(0)
    p = DegradationPrompt(num_prototypes=3, prompt_dim=4).to(device)
    p.train()
    alpha = torch.softmax(torch.randn(1, 3), dim=-1).to(device)
    prompt = p(alpha)
    prompt.sum().backward()
    assert p.mu.grad is not None
    assert p.log_sigma.grad is not None


def test_film_shape(device):
    torch.manual_seed(0)
    film = FiLM(prompt_dim=8, num_channels=16).to(device)
    x = torch.randn(2, 16, 32, 32, device=device)
    prompt = torch.randn(2, 8, device=device)
    out = film(x, prompt)
    assert out.shape == x.shape


def test_film_identity_at_init(device):
    torch.manual_seed(0)
    film = FiLM(prompt_dim=8, num_channels=16).to(device)
    film.eval()
    x = torch.randn(2, 16, 8, 8, device=device)
    prompt = torch.randn(2, 8, device=device)
    out = film(x, prompt)
    # zero-init => gamma = 1, beta = 0 => identity
    assert torch.allclose(out, x, atol=1e-6)