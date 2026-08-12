"""Latency benchmark (implementation.md §7).

Measures end-to-end wall time for a trained CG-NAFNet on a single grayscale
image at the configured resolution (patch_size x patch_size). Reports mean /
median / p95 over N timed forwards after a short warmup, and exits non-zero
(loud fail) if p95 exceeds the configured budget.

Run:
    python -m cgnafnet.benchmark_latency \
        --config cgnafnet/configs/base.yaml \
        --checkpoint runs/<run_id>/best.pt
"""

import argparse
import time
from pathlib import Path

import torch

from cgnafnet.models.cg_nafnet import CGNAFNet
from cgnafnet.train import ROOT, load_config


def main(config_path, checkpoint_path, repeats=100, warmup=20, verbose=True):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"benchmark device: {device}")

    mcfg = cfg["model"]
    model = CGNAFNet(
        img_channel=1,
        width=mcfg["width"],
        num_stages=mcfg["num_stages"],
        blocks_per_stage=tuple(mcfg["blocks_per_stage"]),
        num_prototypes_per_stage=tuple(mcfg["num_prototypes_per_stage"]),
        prompt_dim=mcfg["prompt_dim"],
        proj_dim=mcfg.get("proj_dim", mcfg["prompt_dim"]),
        # Match training hparams so the state_dict loads; the head is never
        # called during inference (true no-op by design).
        aux_order_head=mcfg.get("aux_order_head", False),
    ).to(device)
    sd = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(sd["model"], strict=True)
    model.eval()

    p = cfg["data"]["patch_size"]
    x = torch.rand(1, 1, p, p, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times = sorted(times)
    mean = sum(times) / len(times)
    median = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95) - 1]
    if verbose:
        print(f"input: {p}x{p} grayscale")
        print(f"warmup: {warmup}, timed forwards: {repeats}")
        print(f"mean   : {mean * 1000:.2f} ms")
        print(f"median : {median * 1000:.2f} ms")
        print(f"p95    : {p95 * 1000:.2f} ms")

    budget = cfg["inference"]["latency_budget_ms"]
    ok = p95 * 1000 <= budget
    print(f"p95 {p95 * 1000:.2f} ms vs budget {budget:.2f} ms -> "
          f"{'PASS' if ok else 'FAIL (over budget)'}")
    if not ok:
        raise SystemExit(1)
    return mean, median, p95


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark CG-NAFNet latency")
    p.add_argument("--config", default=str(ROOT / "cgnafnet" / "configs" / "base.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.checkpoint, args.repeats, args.warmup)