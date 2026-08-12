"""Phase 1 checkpoint: verify the data pipeline.

Checks mandated by implementation.md §2.3:
  1. output shapes match input,
  2. no NaNs,
  3. severities continuously distributed across 1000 samples,
  4. all 6 degradation orders are producible.
"""

import pytest
import torch

from cgnafnet.data.compose import (
    NUM_ORDERS,
    ORDERS,
    OrderSampler,
    compose_degradation,
    sample_severity_and_order,
)
from cgnafnet.data.degradations import apply_downsample, apply_gaussian, apply_speckle

RANGES = {
    "speckle_sigma": [0.05, 0.4],
    "gaussian_sigma": [0.01, 0.15],
    "downsample_factor": [1.5, 4.0],
}

B, C, H, W = 4, 1, 256, 256


@pytest.fixture(scope="module")
def clean():
    torch.manual_seed(0)
    return torch.rand(B, C, H, W)


def test_speckle_shapes_and_no_nan(clean):
    out, params = apply_speckle(clean, 0.2)
    assert out.shape == clean.shape
    assert not torch.isnan(out).any()
    assert params["speckle_sigma"] == pytest.approx(0.2)


def test_gaussian_shapes_and_no_nan(clean):
    out, params = apply_gaussian(clean, 0.1)
    assert out.shape == clean.shape
    assert not torch.isnan(out).any()
    assert params["gaussian_sigma"] == pytest.approx(0.1)


def test_downsample_shapes(clean):
    for sf in (1.5, 2.0, 3.5, 4.0):
        out, params = apply_downsample(clean, sf)
        assert not torch.isnan(out).any()
        expected = (int((256 / sf) + 0.999), int((256 / sf) + 0.999))
        assert out.shape[-2:] == expected
        assert params["downsample_factor"] == pytest.approx(sf)


def test_speckle_broadcast_batch(clean):
    sigmas = torch.linspace(0.05, 0.4, B)
    out, params = apply_speckle(clean, sigmas)
    assert out.shape == clean.shape
    assert params["speckle_sigma"].shape == (B,)


def test_compose_fixed_size(clean):
    for order in ORDERS:
        sev, _ = sample_severity_and_order(B, RANGES)
        degraded, log = compose_degradation(clean, order, sev)
        assert degraded.shape == clean.shape, f"order {order} changed resolution"
        assert not torch.isnan(degraded).any()
        assert log["order"] == list(order)


def test_all_six_orders_producible():
    seen = {tuple(o) for o in ORDERS}
    assert len(seen) == 6
    assert NUM_ORDERS == 6
    sampler = OrderSampler()
    drawn = {tuple(ORDERS[i]) for i in sampler.sample(1000)}
    assert drawn == seen


def test_severities_continuously_distributed():
    sev, _ = sample_severity_and_order(1000, RANGES)
    for key, (lo, hi) in RANGES.items():
        vals = sev[key]
        assert vals.min() >= lo
        assert vals.max() <= hi
        uniq = torch.unique(vals)
        assert uniq.numel() > 800, f"{key}: severity values clustered, not continuous"


def test_severity_scalar_inputs(clean):
    sev = {"speckle_sigma": 0.2, "gaussian_sigma": 0.05, "downsample_factor": 2.0}
    degraded, log = compose_degradation(clean, ORDERS[0], sev)
    assert degraded.shape == clean.shape


def test_weights_sampler_runs():
    w = torch.ones(6)
    s = OrderSampler(w)
    idx = s.sample(50)
    assert len(idx) == 50
    assert all(0 <= i < 6 for i in idx)