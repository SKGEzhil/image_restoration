"""Learning rate finder for image restoration models.

Uses torch-lr-finder (https://github.com/davidtvs/pytorch-lr-finder) to sweep
learning rates and suggest an optimal starting LR for training.

Usage:
    python src/find_lr.py
    python src/find_lr.py --model scunet_sr --loss-preset scunet_l1
    python src/find_lr.py --end-lr 10 --num-iter 200
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; save plots to file only
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder as _LRFinder, TrainDataLoaderIter

from dataset import PairedDataset, get_device, set_seed
from losses import build_loss
from models import create_model

DEFAULT_CONFIG = Path(__file__).resolve().parent / "training_config.yaml"


# ─── Adapters for torch-lr-finder ────────────────────────────────────


class PairedDataLoaderIter(TrainDataLoaderIter):
    """Adapts PairedDataset batches (lr, gt, name) -> (inputs, labels)."""

    def inputs_labels_from_batch(self, batch):
        lr, gt, _name = batch
        return lr, gt


class LossWrapper:
    """Wraps LossCombinator so it's callable as criterion(pred, gt)."""

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, pred, gt):
        return self.loss_fn(pred, gt, epoch=0)


# ─── Config Loading ──────────────────────────────────────────────────


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    defaults = {
        "train_model": "nafnet",
        "models": {"nafnet": {"width": 32, "num_blks": 8, "drop_out_rate": 0.0}},
        "data_dir": str(Path(__file__).resolve().parent.parent / "data"),
        "batch_size": 8,
        "seed": 42,
        "loss_config": {"preset": "l1_ssim_baseline"},
        "exclude_samples": None,
        "include_augmented_data": True,
        "augmentation_offset": 3200,
        "num_workers": 0,
    }
    merged = {**defaults, **config}

    config_dir = Path(config_path).resolve().parent
    for key in ("data_dir",):
        if merged.get(key):
            p = Path(merged[key])
            if not p.is_absolute():
                merged[key] = str((config_dir / p).resolve())

    return merged


# ─── Main ────────────────────────────────────────────────────────────


def find_lr(args):
    config = load_config(args.config)

    model_name = args.model or config["train_model"]
    if args.loss_preset:
        config["loss_config"] = {"preset": args.loss_preset}

    set_seed(config["seed"])
    device = get_device()
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(0.9)

    # Model (no torch.compile)
    model_kwargs = config["models"].get(model_name, {})
    model = create_model(name=model_name, **model_kwargs).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} ({params / 1e6:.2f}M params) | Device: {device}")

    # Dataset + DataLoader
    train_ds = PairedDataset(
        config["data_dir"],
        split="train",
        augment=False,
        seed=config["seed"],
        exclude_list=config.get("exclude_samples"),
        include_augmented_data=config.get("include_augmented_data", True),
        augmentation_offset=config.get("augmentation_offset", 3200),
    )
    pin = device.type != "mps"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size or config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=pin,
        drop_last=True,
    )
    print(f"Train samples: {len(train_ds)} | Batch size: {train_loader.batch_size}")

    # Loss
    loss_fn = build_loss(config["loss_config"], device=device)
    criterion = LossWrapper(loss_fn)

    # Optimizer (model params + loss learnable params, same as train.py)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + loss_fn.log_sigma_sq_params,
        lr=args.start_lr,
    )

    # LR Finder
    lr_finder = _LRFinder(model, optimizer, criterion, device=device)
    print(f"Running LR range test: {args.start_lr} -> {args.end_lr} ({args.num_iter} iters)")

    lr_finder.range_test(
        train_loader,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iter=args.num_iter,
        step_mode=args.step_mode,
        smooth_f=args.smooth_f,
        diverge_th=float("inf"),
    )

    # Plot + save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    preset_name = config["loss_config"].get("preset", "custom")
    filename = f"{model_name}_{preset_name}_lr_finder.png"
    save_path = save_dir / filename

    ax, suggested_lr = lr_finder.plot(
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        log_lr=True,
        suggest_lr=True,
    )
    ax.set_title(f"LR Finder: {model_name} + {preset_name}")
    ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {save_path}")
    print(f"Suggested LR: {suggested_lr:.2e}")

    # Save results as JSON
    results = {
        "model": model_name,
        "preset": preset_name,
        "suggested_lr": float(suggested_lr),
        "num_iter": args.num_iter,
        "step_mode": args.step_mode,
        "batch_size": train_loader.batch_size,
        "train_samples": len(train_ds),
        "history": {
            "lr": [float(v) for v in lr_finder.history["lr"]],
            "loss": [float(v) for v in lr_finder.history["loss"]],
        },
    }
    json_path = save_dir / f"{model_name}_{preset_name}_lr_finder.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved: {json_path}")

    # Restore model + optimizer to initial state
    lr_finder.reset()
    print("Model and optimizer restored to initial state.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find optimal learning rate for image restoration models."
    )
    # Config
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG),
        help="Path to training config YAML (default: src/training_config.yaml)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name (overrides config)")
    parser.add_argument("--loss-preset", type=str, default=None, help="Loss preset (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")

    # LR Finder params
    parser.add_argument("--start-lr", type=float, default=1e-7, help="Starting LR")
    parser.add_argument("--end-lr", type=float, default=1.0, help="Ending LR")
    parser.add_argument("--num-iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--step-mode", type=str, default="exp", choices=["exp", "linear"], help="LR schedule mode")
    parser.add_argument("--smooth-f", type=float, default=0.05, help="Loss smoothing factor")
    parser.add_argument("--skip-start", type=int, default=10, help="Batches to skip at start of plot")
    parser.add_argument("--skip-end", type=int, default=5, help="Batches to skip at end of plot")

    # Output
    parser.add_argument("--save-dir", type=str, default="runs/lr_finder", help="Directory for output plots")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    find_lr(args)
