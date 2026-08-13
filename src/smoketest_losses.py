"""Smoketest: run every loss preset for 10 training steps to verify it works."""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from losses import build_loss, PRESETS
from models import create_model


def run_preset(preset_name, model, device, steps=10):
    """Run forward+backward for `steps` iterations with the given preset."""
    config = {"preset": preset_name}
    loss_fn = build_loss(config, device=device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + loss_fn.log_sigma_sq_params,
        lr=1e-3,
    )

    history = []
    for step in range(steps):
        lr_img = torch.randn(2, 1, 128, 128, device=device)
        gt_img = torch.rand(2, 1, 256, 256, device=device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(lr_img)
        loss = loss_fn(pred, gt_img)
        loss.backward()
        optimizer.step()

        loss_fn.step(epoch=0)
        history.append(loss.item())

    return history


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}\n")

    model = create_model(name="nafnet", width=32, num_blks=8, drop_out_rate=0.0)
    model = model.to(device)
    model.train()

    results = {}
    for name in PRESETS:
        t0 = time.time()
        try:
            history = run_preset(name, model, device, steps=10)
            elapsed = time.time() - t0
            first, last = history[0], history[-1]
            results[name] = {
                "status": "PASS",
                "first_loss": f"{first:.6f}",
                "last_loss": f"{last:.6f}",
                "loss_decreased": first > last,
                "time_s": f"{elapsed:.1f}",
            }
        except Exception as e:
            elapsed = time.time() - t0
            results[name] = {
                "status": "FAIL",
                "error": str(e)[:200],
                "time_s": f"{elapsed:.1f}",
            }
        # Reset model weights for next preset to keep runs independent
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    # Summary
    passed = sum(1 for v in results.values() if v["status"] == "PASS")
    failed = sum(1 for v in results.values() if v["status"] == "FAIL")
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} presets")
    print(f"{'='*60}\n")

    for name, info in results.items():
        status = info["status"]
        icon = "✓" if status == "PASS" else "✗"
        line = f"  {icon} {name:30s} {status}"
        if status == "PASS":
            line += f"  loss {info['first_loss']} -> {info['last_loss']}  ({info['time_s']}s)"
        else:
            line += f"  ERROR: {info['error']}"
        print(line)

    print(f"\n{'='*60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
