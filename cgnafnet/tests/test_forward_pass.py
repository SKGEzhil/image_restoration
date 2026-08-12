"""Phase 2 checkpoint: full model forward + backward, no NaN, aux no-op.

Verifies implementation.md §3.5-3.6:
  - output shape matches input resolution,
  - backward runs on a dummy L1 loss,
  - no NaNs in output or gradients,
  - disabling the aux head leaves the restoration output bit-identical.
"""

import pytest
import torch

from cgnafnet.models.cg_nafnet import CGNAFNet

CONFIG = dict(
    img_channel=1,
    width=32,
    num_stages=4,
    blocks_per_stage=(2, 2, 4, 2),
    num_prototypes_per_stage=(3, 3, 3, 3),
    prompt_dim=64,
    proj_dim=64,
    aux_order_head=True,
)


def _make(with_aux=True):
    torch.manual_seed(0)
    cfg = dict(CONFIG)
    cfg["aux_order_head"] = with_aux
    return CGNAFNet(**cfg)


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


def test_forward_shape_and_no_nan(device):
    torch.manual_seed(0)
    m = _make()
    m.eval()
    x = torch.rand(1, 1, 256, 256, device=device)
    out = m(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_forward_odd_input_size(device):
    torch.manual_seed(0)
    m = _make()
    m.eval()
    x = torch.rand(1, 1, 260, 260, device=device)
    out = m(x)
    assert out.shape == x.shape


def test_backward_no_nan(device):
    torch.manual_seed(0)
    m = _make()
    m.train()
    x = torch.rand(2, 1, 128, 128, device=device)
    target = torch.rand_like(x)
    out = m(x)
    loss = torch.nn.functional.l1_loss(out, target)
    loss.backward()
    assert not torch.isnan(loss).item()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


def test_aux_head_returns_logits(device):
    torch.manual_seed(0)
    m = _make(with_aux=True)
    m.eval()
    x = torch.rand(2, 1, 128, 128, device=device)
    out, logits = m(x, return_aux=True)
    assert logits.shape == (2, 6)
    assert out.shape == x.shape
    assert m.aux_head is not None


def test_return_cluster_posteriors(device):
    torch.manual_seed(0)
    m = _make(with_aux=False)
    m.eval()
    x = torch.rand(1, 1, 128, 128, device=device)
    out, alphas = m(x, return_cluster_posteriors=True)
    assert len(alphas) == 2 * m.enc_stages.__len__()  # 4 enc + 4 dec
    for a in alphas:
        assert a.shape[0] == 1
        assert torch.allclose(a.sum(dim=-1), torch.ones(1, device=device), atol=1e-5)


def test_aux_head_noop_bit_identical(device):
    """Restoration output must be identical with aux head on/off (same seed)."""
    torch.manual_seed(0)
    x = torch.rand(1, 1, 128, 128, device=device)

    torch.manual_seed(0)
    m_on = _make(with_aux=True)
    m_on.eval()
    out_on, _ = m_on(x, return_aux=True)

    torch.manual_seed(0)
    m_off = _make(with_aux=False)
    m_off.eval()
    out_off = m_off(x, return_aux=False)

    assert torch.equal(out_on, out_off), "aux head affects restoration output"


def test_raw_residual_learning():
    """With zeroed backbone params (eval), output should be near input."""
    torch.manual_seed(0)
    m = _make(with_aux=False)
    m.eval()
    # zero all conv/film params so the network is a no-op
    for p in m.parameters():
        p.data.zero_()
    x = torch.rand(1, 1, 64, 64)
    out = m(x)
    assert torch.allclose(out, x, atol=1e-4)