"""Train NAFNet on paired NoisyLR (128x128) -> GT (256x256) data.

Loss = l1_weight * L1 + ssim_weight * (1 - SSIM). Writes everything to
runs/<run_id>/ and a tqdm console log; optional Weights & Biases tracking
(disabled until a real key is set in .env).
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedDataset, get_device, set_seed
from metrics import build_loss, compute_psnr, compute_ssim, separate_losses
from model import create_model


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "src" / "training_config.yaml"


def parse_args(config_path=None):
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    config_dir = config_path.resolve().parent
    defaults = {
        "data_dir": str(Path(__file__).resolve().parent.parent / "data"),
        "epochs": 3,
        "batch_size": 8,
        "lr": 1e-3,
        "val_every": 50,
        "wandb_log_step": 5,
        "l1_weight": 0.5,
        "ssim_weight": 0.5,
        "width": 32,
        "num_blks": 8,
        "drop_out_rate": 0.0,
        "num_workers": 2,
        "seed": 42,
        "run_name": None,
        "resume": None,
    }
    merged = {**defaults, **config}
    for key in ("data_dir", "resume"):
        if merged.get(key):
            p = Path(merged[key])
            if not p.is_absolute():
                merged[key] = str((config_dir / p).resolve())
    return argparse.Namespace(**merged)


def setup_logging(run_id):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"log_{run_id}.log"

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Every record (incl. per-step DEBUG) goes to the file; console gets INFO only.
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    logger.propagate = False

    return logger, log_file


def init_wandb(args, run_id, note=None):
    load_dotenv()
    key = os.getenv("WANDB_API_KEY")
    if not key or key == "YOUR_WANDB_API_KEY_HERE":
        logger = logging.getLogger("train")
        logger.info("[wandb] skipped: set a real WANDB_API_KEY in .env to enable tracking")
        return None
    os.environ["WANDB_API_KEY"] = key
    try:
        import wandb
        wandb.login(key=key, relogin=True)
        run = wandb.init(
            project="image-restoration",
            id=run_id,
            name=run_id,
            config=vars(args),
            notes=note,
            save_code=True,
        )
        return run
    except Exception as e:  # noqa: BLE001
        logger = logging.getLogger("train")
        logger.warning(f"[wandb] init failed ({e}); continuing without wandb")
        return None


def save_checkpoint(state, path):
    torch.save(state, path)


def log_samples(wandb_run, images, step):
    if wandb_run is None:
        return
    import numpy as np
    import wandb
    lr, gt, pred = images
    grid = []
    for i in range(lr.shape[0]):
        grid.append(wandb.Image(lr[i], caption=f"LR {i}"))
        grid.append(wandb.Image(gt[i], caption=f"GT {i}"))
        grid.append(wandb.Image(np.clip(pred[i], 0, 1), caption=f"Pred {i}"))
    wandb_run.log({"samples/lr_gt_pred": grid, "train/step": step}, step=step)


def validate(model, val_loader, device, l1_weight, ssim_weight):
    model.eval()
    total_l1 = total_ssim_loss = total_psnr = 0.0
    num = 0
    with torch.no_grad():
        for lr, gt, _ in val_loader:
            lr, gt = lr.to(device), gt.to(device)
            pred = model(lr)
            l1, ssim_loss, _ = separate_losses(pred, gt)
            b = lr.size(0)
            total_l1 += l1.item() * b
            total_ssim_loss += ssim_loss.item() * b
            total_psnr += compute_psnr(pred, gt).item() * b
            num += b
    model.train()
    val_l1 = total_l1 / num
    val_ssim_loss = total_ssim_loss / num
    return {
        "val/l1": val_l1,
        "val/ssim_loss": val_ssim_loss,
        "val/loss": l1_weight * val_l1 + ssim_weight * val_ssim_loss,
        "val/psnr": total_psnr / num,
    }


def main():
    args = parse_args()
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logging(run_id)
    logger.info(f"run_id={run_id}")
    logger.info(f"args={json.dumps(vars(args), indent=2)}")
    logger.info(f"log file: {log_file}")

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    set_seed(args.seed)
    device = get_device()
    logger.info(f"device={device}")
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(0.9)
    pin = device.type != "mps"

    train_ds = PairedDataset(args.data_dir, split="train", augment=True, seed=args.seed)
    val_ds = PairedDataset(args.data_dir, split="val")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch

    model = create_model(args).to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"model=NAFNet params={params/1e6:.2f}M")

    loss_fn = build_loss(args.l1_weight, args.ssim_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    epochs_elapsed = 0
    steps_elapsed = 0
    global_step = 0
    best_val_loss = float("inf")

    wandb_run = None
    if args.wandb_log_step > 0:
        wandb_run = init_wandb(args, run_id)
        if wandb_run is not None:
            wandb_run.summary["model_params"] = params
            wandb_run.summary["device"] = str(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        epochs_elapsed = ckpt["epoch"]
        steps_elapsed = ckpt["step"]
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"resumed from {args.resume} (epoch={epochs_elapsed} step={steps_elapsed})")

    start = time.time()
    sample_imgs = None
    lr_val = scheduler.get_last_lr()[0]
    for epoch in range(epochs_elapsed, args.epochs):
        pbar = tqdm(train_loader, total=steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{args.epochs}", unit="step")
        for step, (lr, gt, _) in enumerate(pbar):
            if epoch == epochs_elapsed and step < steps_elapsed:
                continue
            lr, gt = lr.to(device), gt.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(lr)
            loss = loss_fn(pred, gt)
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr_val = scheduler.get_last_lr()[0]
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.5f}", lr=f"{lr_val:.1e}")
            logger.debug(f"step {global_step} (epoch {epoch} batch {step}) "
                         f"loss={loss.item():.5f} lr={lr_val:.1e}")

            if sample_imgs is None:
                sample_imgs = (lr[:4, 0].cpu().numpy(), gt[:4, 0].cpu().numpy(),
                               pred[:4, 0].detach().cpu().numpy())
                log_samples(wandb_run, sample_imgs, global_step)

            if global_step % args.wandb_log_step == 0 and wandb_run is not None:
                l1, ssim_loss, ssim = separate_losses(pred, gt)
                wandb_run.log({
                    "train/loss": loss.item(),
                    "train/l1": l1.item(),
                    "train/ssim_loss": ssim_loss.item(),
                    "train/ssim": ssim.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                }, step=global_step)

            if global_step % args.val_every == 0:
                val_metrics = validate(model, val_loader, device, args.l1_weight, args.ssim_weight)
                logger.info(
                    f"[epoch {epoch}/{args.epochs} step {step + 1}/{steps_per_epoch} "
                    f"global_step {global_step}] "
                    f"val loss={val_metrics['val/loss']:.4f} "
                    f"val L1={val_metrics['val/l1']:.4f} "
                    f"val SSIM={1 - val_metrics['val/ssim_loss']:.4f} "
                    f"val PSNR={val_metrics['val/psnr']:.2f} dB"
                )
                if wandb_run is not None:
                    wandb_run.log(val_metrics | {"global_step": global_step}, step=global_step)
                if val_metrics["val/loss"] < best_val_loss:
                    best_val_loss = val_metrics["val/loss"]
                    save_checkpoint({
                        "model": model.state_dict(),
                        "epoch": epoch, "step": step + 1, "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    }, run_dir / "best.pt")
                    logger.info(f"best model updated (val loss={best_val_loss:.4f})")

        save_checkpoint({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1, "step": 0, "global_step": global_step,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }, run_dir / "last.pt")

    elapsed = time.time() - start
    logger.info(f"training finished in {elapsed/3600:.2f} h")
    logger.info(f"best val loss: {best_val_loss:.4f}")
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val_loss
        wandb_run.summary["train_time_h"] = round(elapsed / 3600, 2)
        wandb_run.summary["global_step"] = global_step
        wandb_run.finish()


if __name__ == "__main__":
    main()