"""Train CG-NAFNet (implementation.md §5).

Standard PyTorch loop with:
  - config-driven Adam (lr=2e-4, betas=(0.9,0.999)), halved at a configured
    epoch fraction,
  - per-component loss logging (recon, ssim, ortho, aux) to CSV,
  - mean alpha-entropy per stage (falling entropy = clusters sharpening),
  - checkpointing every N epochs + keep best-by-validation-PSNR,
  - validation with deterministic prompt (eval mode, eps=0).

Run:
    python -m cgnafnet.train --config cgnafnet/configs/base.yaml
"""

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from cgnafnet.data.compose import NUM_ORDERS, ORDERS
from cgnafnet.data.dataset import CleanImageDataset, DegradedPairDataset
from cgnafnet.losses import build_total_loss
from cgnafnet.models.cg_nafnet import CGNAFNet

ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve clean_image_dirs against the repo root, matching both
    # "src/..." and "../src/..." spellings.
    resolved = []
    for d in cfg["data"]["clean_image_dirs"]:
        p = Path(d)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        resolved.append(str(p))
    cfg["data"]["clean_image_dirs"] = resolved
    _coerce_numerics(cfg)
    return cfg


def _coerce_numerics(obj):
    """PyYAML keeps strings like '2e-4' (no decimal point) as str; coerce numbers."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _coerce_numerics(v)
        return obj
    if isinstance(obj, list):
        return [_coerce_numerics(v) for v in obj]
    if isinstance(obj, str):
        lowered = obj.strip().lower()
        if lowered in ("true", "false", "null", "none", "yes", "no"):
            return obj
        try:
            return float(obj)
        except ValueError:
            return obj
    return obj


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_items(batch):
    """Collate (degraded, clean, log, name) keeping log as a list of dicts."""
    degraded = torch.stack([b[0] for b in batch], dim=0)
    clean = torch.stack([b[1] for b in batch], dim=0)
    logs = [b[2] for b in batch]
    names = [b[3] for b in batch]
    return degraded, clean, logs, names


def order_labels_from_logs(logs, device):
    """Map a list of per-sample ground-truth log dicts to order-label tensor."""
    indices = [ORDERS.index(tuple(log["order"])) for log in logs]
    return torch.tensor(indices, dtype=torch.long, device=device)


def mean_alpha_entropy(alphas):
    """mean over stages & batch of posterior entropy, H(alpha)."""
    if not alphas:
        return 0.0
    total = 0.0
    for a in alphas:
        log_a = torch.log(a.clamp_min(1e-8))
        total = total + (-a * log_a).sum(dim=-1).mean()
    return (total / len(alphas)).item()


def validate(model, val_loader, device, cfg):
    model.eval()
    total_psnr = 0.0
    total_l1 = 0.0
    n = 0
    with torch.no_grad():
        for degraded, clean, log, _ in tqdm(val_loader, desc="val", leave=False):
            degraded, clean = degraded.to(device), clean.to(device)
            out = model(degraded, return_aux=False)
            mse = (out - clean).pow(2).mean()
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
            total_psnr += psnr.item() * clean.shape[0]
            total_l1 += F.l1_loss(out, clean).item() * clean.shape[0]
            n += clean.shape[0]
    model.train()
    return total_psnr / n, total_l1 / n


def main(config_path=None, smoke=False):
    if config_path is None:
        config_path = ROOT / "cgnafnet" / "configs" / "base.yaml"
    cfg = load_config(config_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "runs" / f"cgnafnet_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    dcfg = cfg["data"]
    clean_dirs = dcfg["clean_image_dirs"]
    # split dirs into train/val (first two dirs; simple, config-driven later)
    split_frac = 0.8
    all_paths = []
    for d in clean_dirs:
        all_paths.extend(sorted(Path(d).glob("*.npy")))
    random.Random(seed).shuffle(all_paths)
    n_train = int(len(all_paths) * split_frac)
    train_paths, val_paths = all_paths[:n_train], all_paths[n_train:]
    print(f"train images: {len(train_paths)}, val images: {len(val_paths)}")

    train_ds = DegradedPairDataset(
        CleanImageDataset(train_paths),
        patch_size=dcfg["patch_size"],
        degradation_ranges=dcfg["degradation_ranges"],
        regional_mixing=dcfg.get("regional_mixing", False),
    )
    val_ds = DegradedPairDataset(
        CleanImageDataset(val_paths),
        patch_size=dcfg["patch_size"],
        degradation_ranges=dcfg["degradation_ranges"],
        regional_mixing=False,
    )

    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              collate_fn=collate_items)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_items)

    mcfg = cfg["model"]
    model = CGNAFNet(
        img_channel=1,
        width=mcfg["width"],
        num_stages=mcfg["num_stages"],
        blocks_per_stage=tuple(mcfg["blocks_per_stage"]),
        num_prototypes_per_stage=tuple(mcfg["num_prototypes_per_stage"]),
        prompt_dim=mcfg["prompt_dim"],
        proj_dim=mcfg.get("proj_dim", mcfg["prompt_dim"]),
        aux_order_head=mcfg["aux_order_head"],
    ).to(device)
    print(f"model params: {count_parameters(model):,}")

    lcfg = cfg["loss"]
    tcfg = cfg["train"]
    epochs = tcfg["epochs"]
    lr = tcfg["lr"]
    halve_at = int(epochs * tcfg["lr_halve_epoch_fraction"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=tuple(tcfg["betas"]))

    log_path = out_dir / "train_log.csv"
    fields = ["epoch", "step", "lr", "loss_total", "recon", "ssim", "ortho",
              "aux", "alpha_entropy", "val_psnr", "val_l1"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    best_psnr = float("-inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        if epoch > halve_at:
            for g in optimizer.param_groups:
                g["lr"] = lr * 0.5
        running = {k: 0.0 for k in ("loss_total", "recon", "ssim", "ortho", "aux")}
        entropies = []
        n_steps = 0
        model.train()
        for degraded, clean, log, _ in tqdm(train_loader, desc=f"train e{epoch}"):
            degraded, clean = degraded.to(device), clean.to(device)
            optimizer.zero_grad()

            if mcfg["aux_order_head"]:
                out, aux_logits, alphas = model(
                    degraded, return_aux=True, return_cluster_posteriors=True
                )
                labels = order_labels_from_logs(log, device)
            else:
                out, alphas = model(degraded, return_cluster_posteriors=True)
                aux_logits, labels = None, None

            loss, comps = build_total_loss(
                out, clean, aux_logits, labels, model,
                recon_type=lcfg["recon_type"],
                lambda_ssim=lcfg["lambda_ssim"],
                lambda_ortho=lcfg["lambda_ortho"],
                lambda_aux=lcfg["lambda_aux"],
            )
            loss.backward()
            optimizer.step()

            for k in running:
                running[k] += comps[k]
            n_steps += 1
            global_step += 1
            entropies.append(mean_alpha_entropy(alphas))

        epoch_entropy = sum(entropies) / max(1, len(entropies))
        val_psnr, val_l1 = validate(model, val_loader, device, cfg)

        row = {
            "epoch": epoch, "step": global_step, "lr": optimizer.param_groups[0]["lr"],
            "loss_total": running["loss_total"] / n_steps,
            "recon": running["recon"] / n_steps,
            "ssim": running["ssim"] / n_steps,
            "ortho": running["ortho"] / n_steps,
            "aux": running["aux"] / n_steps,
            "alpha_entropy": epoch_entropy,
            "val_psnr": val_psnr, "val_l1": val_l1,
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"e{epoch} loss={row['loss_total']:.4f} entropy={epoch_entropy:.4f} "
              f"val_psnr={val_psnr:.2f}")

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "config": cfg}
        if epoch % tcfg.get("save_every", 10) == 0:
            torch.save(ckpt, out_dir / f"epoch_{epoch}.pt")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(ckpt, out_dir / "best.pt")
        torch.save(ckpt, out_dir / "last.pt")

    print(f"done. best val PSNR: {best_psnr:.2f}. artifacts in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train CG-NAFNet")
    p.add_argument("--config", default=str(ROOT / "cgnafnet" / "configs" / "base.yaml"))
    p.add_argument("--smoke", action="store_true", help="single short batch run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        main(args.config, smoke=True)
    else:
        main(args.config)