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
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedDataset, get_device, set_seed
from losses import build_loss, lsgan_loss
from metrics import compute_psnr, compute_ssim
from models import create_model


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "src" / "training_config.yaml"


def parse_args(config_path=None):
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    config_dir = config_path.resolve().parent
    defaults = {
        "train_model": "nafnet",
        "models": {"nafnet": {"width": 32, "num_blks": 8, "drop_out_rate": 0.0}},
        "data_dir": str(Path(__file__).resolve().parent.parent / "data"),
        "epochs": 3,
        "batch_size": 8,
        "lr": 1e-3,
        "val_every": 50,
        "wandb_log_step": 5,
        "loss_config": {"preset": "l1_ssim_baseline"},
        "scheduler": {"type": "cosine"},
        "gan_training": {"enabled": False},
        "num_workers": 2,
        "seed": 42,
        "run_name": None,
        "resume": None,
        "exclude_samples": None,
        "include_augmented_data": True,
        "augmentation_offset": 3200,
    }
    merged = {**defaults, **config}

    # CLI overrides via argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--loss-preset", type=str, default=None,
                        help="Override loss_config.preset from CLI (e.g., combo_3)")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    cli_args, _ = parser.parse_known_args()

    if cli_args.loss_preset is not None:
        merged["loss_config"] = {"preset": cli_args.loss_preset}
    if cli_args.run_name is not None:
        merged["run_name"] = cli_args.run_name
    if cli_args.epochs is not None:
        merged["epochs"] = cli_args.epochs
    if cli_args.batch_size is not None:
        merged["batch_size"] = cli_args.batch_size
    if cli_args.lr is not None:
        merged["lr"] = cli_args.lr

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


def validate(model, val_loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num = 0
    with torch.inference_mode():
        for lr, gt, _ in val_loader:
            lr, gt = lr.to(device), gt.to(device)
            pred = model(lr)
            total_loss += loss_fn(pred, gt).item() * lr.size(0)
            total_psnr += compute_psnr(pred, gt).item() * lr.size(0)
            total_ssim += compute_ssim(pred, gt).item() * lr.size(0)
            num += lr.size(0)
    model.train()

    # Compute component losses on a sample batch for logging
    components = loss_fn.get_components(pred, gt)

    # Convert loss names to metric names for logging
    metrics = {}
    for k, v in components.items():
        if k in ("ssim", "ms_ssim"):
            metrics[f"val/{k}_loss"] = v
        else:
            metrics[f"val/{k}"] = v

    metrics["val/loss"] = total_loss / num
    metrics["val/psnr"] = total_psnr / num
    metrics["val/ssim"] = total_ssim / num
    return metrics


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

    train_ds = PairedDataset(args.data_dir, split="train", augment=True,
                              seed=args.seed, exclude_list=args.exclude_samples,
                              include_augmented_data=args.include_augmented_data,
                              augmentation_offset=args.augmentation_offset)
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

    model = create_model(name=args.train_model, **args.models[args.train_model]).to(device)
    if device.type == "cuda":
        model = torch.compile(model)
    else:
        logger.info("torch.compile skipped: only supported on CUDA")
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"model={args.train_model} params={params/1e6:.2f}M")

    # ─── GAN: discriminator setup ───────────────────────────────────────
    discriminator = None
    d_optimizer = None
    d_scheduler = None
    gan_cfg = getattr(args, "gan_training", {})
    use_gan = gan_cfg.get("enabled", False)
    if use_gan:
        from models.discriminator import create_discriminator
        d_cfg = gan_cfg.get("discriminator", {})
        in_ch = args.models[args.train_model].get("in_nc", 1)
        discriminator = create_discriminator(
            input_nc=in_ch,
            ndf=d_cfg.get("ndf", 64),
            n_layers=d_cfg.get("n_layers", 3),
            use_spectral_norm=d_cfg.get("use_spectral_norm", True),
        ).to(device)
        d_params = sum(p.numel() for p in discriminator.parameters())
        logger.info(f"discriminator params={d_params/1e6:.2f}M")
        d_lr = gan_cfg.get("discriminator_lr", 1e-4)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.9, 0.999))
        logger.info(f"GAN training enabled (D lr={d_lr})")

    # ─── Loss & Optimizer ─────────────────────────────────────────────
    loss_fn = build_loss(args.loss_config, device=device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + loss_fn.log_sigma_sq_params,
        lr=args.lr,
    )

    # ─── Scheduler selection ──────────────────────────────────────────
    sched_cfg = getattr(args, "scheduler", {})
    sched_type = sched_cfg.get("type", "cosine")
    if sched_type == "step":
        step_size = sched_cfg.get("step_size", 200000)
        gamma = sched_cfg.get("step_gamma", 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(f"scheduler=StepLR (step_size={step_size}, gamma={gamma})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
        logger.info(f"scheduler=CosineAnnealingLR (T_max={total_steps})")

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

    buf_size = args.wandb_log_step if args.wandb_log_step > 0 else 1
    buf_loss = deque(maxlen=buf_size)
    buf_ssim = deque(maxlen=buf_size)
    buf_components = {}  # dynamic buffer per active loss component

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        epochs_elapsed = ckpt["epoch"]
        steps_elapsed = ckpt["step"]
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        loss_fn.load_params(ckpt.get("loss_params", {}))
        if use_gan and "discriminator" in ckpt:
            discriminator.load_state_dict(ckpt["discriminator"])
            d_optimizer.load_state_dict(ckpt["d_optimizer"])
            if "d_scheduler" in ckpt:
                d_scheduler.load_state_dict(ckpt["d_scheduler"])
        logger.info(f"resumed from {args.resume} (epoch={epochs_elapsed} step={steps_elapsed})")

    start = time.time()
    sample_imgs = None
    lr_val = scheduler.get_last_lr()[0]
    for epoch in range(epochs_elapsed, args.epochs):
        loss_fn.step(epoch)
        pbar = tqdm(train_loader, total=steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{args.epochs}", unit="step")
        for step, (lr, gt, _) in enumerate(pbar):
            if epoch == epochs_elapsed and step < steps_elapsed:
                continue
            lr, gt = lr.to(device), gt.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(lr)

            # Pixel-space loss (L1, DISTS, etc.)
            loss_pixel = loss_fn(pred, gt)
            loss = loss_pixel

            # ─── GAN branch: generator adversarial loss ─────────────────
            if use_gan:
                d_fake = discriminator(pred)
                loss_adv = lsgan_loss(d_fake, target_is_real=True)
                w_adv = gan_cfg.get("loss_weights", {}).get("adv", 0.1)
                loss = loss + w_adv * loss_adv

            loss.backward()
            optimizer.step()
            scheduler.step()

            # ─── GAN branch: discriminator update ───────────────────────
            if use_gan and (global_step % gan_cfg.get("discriminator_update_freq", 1) == 0):
                d_optimizer.zero_grad(set_to_none=True)
                d_real = discriminator(gt)
                d_fake = discriminator(pred.detach())
                loss_d_real = lsgan_loss(d_real, target_is_real=True)
                loss_d_fake = lsgan_loss(d_fake, target_is_real=False)
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                d_optimizer.step()
                if d_scheduler is not None:
                    d_scheduler.step()

            lr_val = scheduler.get_last_lr()[0]
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.5f}", lr=f"{lr_val:.1e}")
            logger.debug(f"step {global_step} (epoch {epoch} batch {step}) "
                         f"loss={loss.item():.5f} lr={lr_val:.1e}")

            if sample_imgs is None:
                sample_imgs = (lr[:4, 0].cpu().numpy(), gt[:4, 0].cpu().numpy(),
                               pred[:4, 0].detach().cpu().numpy())
                log_samples(wandb_run, sample_imgs, global_step)

            components = loss_fn.get_components(pred, gt)
            buf_loss.append(loss.item())
            buf_ssim.append(compute_ssim(pred, gt).item())
            for name, val in components.items():
                if name not in buf_components:
                    buf_components[name] = deque(maxlen=buf_size)
                buf_components[name].append(val)

            # Buffer GAN metrics for logging
            if use_gan:
                if "adv" not in buf_components:
                    buf_components["adv"] = deque(maxlen=buf_size)
                buf_components["adv"].append(loss_adv.item())
                if "d_loss" not in buf_components:
                    buf_components["d_loss"] = deque(maxlen=buf_size)
                buf_components["d_loss"].append(loss_d.item())

            if global_step % args.wandb_log_step == 0 and wandb_run is not None:
                log_dict = {
                    "train/loss": sum(buf_loss) / len(buf_loss),
                    "train/ssim": sum(buf_ssim) / len(buf_ssim),
                    "train/lr": scheduler.get_last_lr()[0],
                }
                for name, vals in buf_components.items():
                    avg = sum(vals) / len(vals)
                    # Convert loss names to metric names for logging
                    if name in ("ssim", "ms_ssim"):
                        log_dict[f"train/{name}_loss"] = avg
                        log_dict[f"train/{name}"] = 1.0 - avg
                    else:
                        log_dict[f"train/{name}"] = avg
                wandb_run.log(log_dict | {"global_step": global_step}, step=global_step)

            if global_step % args.val_every == 0:
                val_metrics = validate(model, val_loader, device, loss_fn)
                val_str = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                logger.info(
                    f"[epoch {epoch}/{args.epochs} step {step + 1}/{steps_per_epoch} "
                    f"global_step {global_step}] {val_str}"
                )
                if wandb_run is not None:
                    wandb_run.log(val_metrics | {"global_step": global_step}, step=global_step)
                if val_metrics["val/loss"] < best_val_loss:
                    best_val_loss = val_metrics["val/loss"]
                    ckpt_best = {
                        "model": model.state_dict(),
                        "epoch": epoch, "step": step + 1, "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "val_metrics": val_metrics,
                        "loss_params": loss_fn.get_params(),
                        "args": vars(args),
                    }
                    if use_gan:
                        ckpt_best["discriminator"] = discriminator.state_dict()
                    save_checkpoint(ckpt_best, run_dir / "best.pt")
                    logger.info(f"best model updated (val loss={best_val_loss:.4f})")

        ckpt_last = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1, "step": 0, "global_step": global_step,
            "best_val_loss": best_val_loss,
            "loss_params": loss_fn.get_params(),
            "args": vars(args),
        }
        if use_gan:
            ckpt_last["discriminator"] = discriminator.state_dict()
            ckpt_last["d_optimizer"] = d_optimizer.state_dict()
            if d_scheduler is not None:
                ckpt_last["d_scheduler"] = d_scheduler.state_dict()
        save_checkpoint(ckpt_last, run_dir / "last.pt")

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